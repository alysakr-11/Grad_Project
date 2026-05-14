import json, os, operator, re, requests, textwrap, subprocess, sys, io, base64
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#f8fafc", size=12),
    title=dict(font=dict(family="Space Grotesk, sans-serif", size=16, color="#f8fafc"), x=0.02, xanchor="left"),
    xaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", linecolor="rgba(148, 163, 184, 0.2)", zerolinecolor="rgba(148, 163, 184, 0.15)",
               tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
               title=dict(font=dict(family="Inter", size=12, color="#cbd5e1"))),
    yaxis=dict(gridcolor="rgba(148, 163, 184, 0.1)", linecolor="rgba(148, 163, 184, 0.2)", zerolinecolor="rgba(148, 163, 184, 0.15)",
               tickfont=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
               title=dict(font=dict(family="Inter", size=12, color="#cbd5e1"))),
    legend=dict(font=dict(family="Inter", size=11, color="#cbd5e1"), bgcolor="rgba(13, 16, 36, 0.6)",
                bordercolor="rgba(148, 163, 184, 0.15)", borderwidth=1),
    margin=dict(l=50, r=20, t=50, b=50),
    hoverlabel=dict(bgcolor="#161938", bordercolor="#00e5ff", font=dict(family="JetBrains Mono", size=12, color="#f8fafc")),
)


def style_fig(fig, title=None):
    layout_args = dict(PLOTLY_LAYOUT)
    if title is not None:
        layout_args = {**layout_args, "title": {**PLOTLY_LAYOUT["title"], "text": title}}
    fig.update_layout(**layout_args)
    return fig


def ask_llama(prompt: str, agent_name: str = "") -> str:
    try:
        res = requests.post(
            LLM_ENDPOINT,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=LLM_TIMEOUT_SEC,
        )
        return res.json().get("response", "No response received.")
    except requests.exceptions.Timeout:
        return f"[{agent_name}] AI Engine timed out after {LLM_TIMEOUT_SEC}s."
    except requests.exceptions.ConnectionError:
        return f"[{agent_name}] AI Engine Offline"
    except Exception as e:
        return f"[{agent_name}] Error: {type(e).__name__}: {e}"


def check_llm_health() -> bool:
    try:
        r = requests.get(LLM_TAGS_ENDPOINT, timeout=LLM_HEALTHCHECK_SEC)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


class RetrieverAgent:
    name = "Retriever Agent"
    icon = "🔍"

    @staticmethod
    def _classify_distribution(skew: float) -> str:
        if abs(skew) < SKEW_NORMAL_THRESHOLD:
            return "roughly normal"
        return "right-skewed" if skew > 0 else "left-skewed"

    def run(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summaries, log_lines = {}, [f"{self.icon} {self.name} activated"]
        for name, df in datasets.items():
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            null_total = int(df.isnull().sum().sum())
            dist_notes = {col: self._classify_distribution(df[col].skew()) for col in num_cols}
            summaries[name] = {
                "shape": df.shape, "num_cols": num_cols, "cat_cols": cat_cols,
                "null_total": null_total, "dist_notes": dist_notes,
                "describe": df[num_cols].describe().to_string() if num_cols else "No numeric columns",
                "sample": df.head(SAMPLE_PREVIEW_ROWS).to_string(),
            }
            log_lines.append(f"  📂 {name}: {df.shape[0]} rows x {df.shape[1]} cols | {len(num_cols)} numeric | {len(cat_cols)} categorical | {null_total} nulls")
        return {"output": summaries, "log": "\n".join(log_lines)}


class PlannerAgent:
    name = "Planner Agent"
    icon = "🎯"

    @staticmethod
    def _build_steps(ctx):
        steps = []
        n_num, n_cat = len(ctx.get("num_cols", [])), len(ctx.get("cat_cols", []))
        if ctx.get("null_total", 0) > 0: steps.append("handle_missing_values")
        if n_num > 0: steps.extend(["descriptive_statistics", "outlier_detection"])
        if n_num >= 2: steps.extend(["correlation_analysis", "anomaly_detection"])
        if n_cat > 0: steps.append("categorical_analysis")
        steps.extend(["visualization", "ai_insight_generation"])
        return steps

    def run(self, retriever_output):
        log_lines = [f"{self.icon} {self.name} activated"]
        plans = {}
        for name, ctx in retriever_output.items():
            steps = self._build_steps(ctx)
            prompt = f"Dataset: shape={ctx['shape']}, numeric={ctx['num_cols']}, categorical={ctx['cat_cols']}, distributions={ctx['dist_notes']}. Briefly explain analytical priority (2-3 sentences)."
            rationale = ask_llama(prompt, self.name)
            plans[name] = {"steps": steps, "rationale": rationale}
            log_lines.append(f"  📂 {name}: {' → '.join(steps)}")
        return {"output": plans, "log": "\n".join(log_lines)}


class StylistAgent:
    name = "Stylist Agent"
    icon = "🎨"
    COLORS = CHART_PALETTE
    VALID_CHARTS = VALID_CHART_TYPES

    @staticmethod
    def _fallback_chart(col_type, dist):
        if col_type == "categorical": return "pie", "fallback: categorical → pie"
        if dist in ("right-skewed", "left-skewed"): return "histogram", f"fallback: {dist} → histogram"
        if dist == "roughly normal": return "box", "fallback: normal → box"
        return "bar", "fallback: default → bar"

    def _ask_llm_for_chart(self, col, dataset_name, col_type, dist, nuniq, n_rows, sample_vals, plan_steps):
        prompt = f"""Choose the BEST chart type for column '{col}' (dataset: '{dataset_name}').
Type: {col_type}, Distribution: {dist}, Unique: {nuniq}/{n_rows}, Samples: {sample_vals}
Available: histogram, box, bar, line, scatter, pie
Reply EXACTLY:
CHART: <type>
REASON: <one sentence>"""
        response = ask_llama(prompt, self.name)
        chart_type, reason = None, "LLM decision"
        for line in response.strip().splitlines():
            ls = line.strip()
            if ls.upper().startswith("CHART:"):
                raw = ls.split(":", 1)[1].strip().lower()
                for v in self.VALID_CHARTS:
                    if v in raw: chart_type = v; break
            elif ls.upper().startswith("REASON:"):
                reason = ls.split(":", 1)[1].strip()
        if chart_type not in self.VALID_CHARTS:
            chart_type, r2 = self._fallback_chart(col_type, dist)
            reason = f"[fallback] {r2}"
        return chart_type, reason

    @staticmethod
    def _real_stats(df, col):
        if df is None or col not in df.columns: return 0, 0, []
        try: return int(df[col].nunique()), len(df), df[col].dropna().head(SAMPLE_PREVIEW_ROWS).tolist()
        except: return 0, 0, []

    def run(self, retriever_output, planner_output, datasets):
        log_lines = [f"{self.icon} {self.name} activated"]
        styles, color_idx = {}, 0
        for name, ctx in retriever_output.items():
            col_styles = {}
            plan_steps = planner_output.get(name, {}).get("steps", [])
            df = datasets.get(name)
            for col in ctx["num_cols"]:
                color = self.COLORS[color_idx % len(self.COLORS)]; color_idx += 1
                dist = ctx["dist_notes"].get(col, "unknown")
                nuniq, n_rows, sv = self._real_stats(df, col)
                ct, reason = self._ask_llm_for_chart(col, name, "numeric", dist, nuniq, n_rows, sv, plan_steps)
                col_styles[col] = {"chart_type": ct, "color": color, "reason": reason, "decided_by": "LLM"}
                log_lines.append(f"  📂 {name} → '{col}' [{dist}]: {ct} — {reason}")
            for col in ctx["cat_cols"]:
                color = self.COLORS[color_idx % len(self.COLORS)]; color_idx += 1
                nuniq, n_rows, sv = self._real_stats(df, col)
                ct, reason = self._ask_llm_for_chart(col, name, "categorical", "N/A", nuniq, n_rows, sv, plan_steps)
                col_styles[col] = {"chart_type": ct, "color": color, "reason": reason, "decided_by": "LLM"}
                log_lines.append(f"  📂 {name} → '{col}' [categorical]: {ct} — {reason}")
            styles[name] = col_styles
        return {"output": styles, "log": "\n".join(log_lines)}


class VisualizerAgent:
    name = "Visualizer Agent"
    icon = "📈"

    @staticmethod
    def _is_time_like(df):
        if isinstance(df.index, pd.DatetimeIndex): return True
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]): return True
        return False

    @classmethod
    def _apply_safety_guards(cls, df, col, ctype):
        try: nuniq = int(df[col].nunique())
        except: return ctype, None
        is_num = is_numeric_dtype(df[col])
        time_ordered = cls._is_time_like(df)
        if ctype == "pie" and nuniq > PIE_MAX_CATEGORIES:
            return "bar", f"Pie with {nuniq} cats unreadable → bar"
        if ctype in ("pie", "bar") and is_num and nuniq > BAR_MAX_CATEGORIES:
            return "histogram", f"{ctype} on numeric with {nuniq} unique → histogram"
        if ctype == "histogram" and nuniq < HISTOGRAM_MIN_UNIQUE:
            return "bar", f"Histogram on {nuniq} unique values → bar"
        if ctype == "line" and LINE_NEEDS_TIME_INDEX and not time_ordered:
            return "histogram" if is_num else "bar", "Line needs time index → distribution chart"
        if ctype == "scatter" and SCATTER_NEEDS_TIME_INDEX and not time_ordered:
            return "box" if is_num else "bar", "Scatter needs time index → distribution chart"
        if ctype in ("histogram", "box") and not is_num:
            return "bar", f"{ctype} needs numeric data → bar"
        return ctype, None

    @staticmethod
    def _make_fig(df, col, style):
        ctype, color = style["chart_type"], style["color"]
        if ctype == "histogram":
            fig = px.histogram(df, x=col, nbins=HISTOGRAM_BINS, color_discrete_sequence=[color])
            fig.update_traces(marker_line_width=0, opacity=0.85)
        elif ctype == "box":
            fig = px.box(df, y=col, color_discrete_sequence=[color])
            fig.update_traces(marker_size=4, line_width=2)
        elif ctype == "pie":
            vc = df[col].value_counts().head(PIE_MAX_CATEGORIES).reset_index()
            vc.columns = [col, "count"]
            fig = px.pie(vc, names=col, values="count", color_discrete_sequence=CHART_PALETTE, hole=0.45)
            fig.update_traces(textfont=dict(family="Inter", size=12, color="#f8fafc"),
                              marker=dict(line=dict(color="#070912", width=2)))
        elif ctype == "line":
            fig = px.line(df, y=col, color_discrete_sequence=[color])
            fig.update_traces(line=dict(width=2.5), mode="lines+markers", marker=dict(size=5))
        elif ctype == "scatter":
            fig = px.scatter(df, x=df.index, y=col, color_discrete_sequence=[color])
            fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="#070912")))
        else:
            vc = df[col].value_counts().head(BAR_TOP_N).reset_index()
            vc.columns = [col, "count"]
            total_cats = int(df[col].nunique())
            suffix = f" (top {BAR_TOP_N} of {total_cats})" if total_cats > BAR_TOP_N else ""
            fig = px.bar(vc, x=col, y="count", color_discrete_sequence=[color])
            fig.update_traces(marker_line_width=0)
            return style_fig(fig, title=f"{col} · {ctype}{suffix}")
        return style_fig(fig, title=f"{col} · {ctype}")

    def run(self, datasets, stylist_output):
        log_lines = [f"{self.icon} {self.name} activated"]
        figures, guard_corrections = {}, 0
        for name, df in datasets.items():
            figures[name] = {}
            for col, style in stylist_output.get(name, {}).items():
                if col not in df.columns: continue
                try:
                    original_type = style["chart_type"]
                    safe_type, gw = self._apply_safety_guards(df, col, original_type)
                    if safe_type != original_type:
                        style = {**style, "chart_type": safe_type}; guard_corrections += 1
                    fig = self._make_fig(df, col, style)
                    figures[name][col] = {"fig": fig, "type": safe_type, "color": style["color"],
                                          "original_type": original_type, "guard_warning": gw}
                    log_lines.append(f"  {'🛡️' if gw else '✅'} {name} → '{col}': {'∼' if gw else ''} {safe_type}{f' (was {original_type})' if gw else ''}")
                except Exception as e:
                    log_lines.append(f"  ⚠️ {name} → '{col}': {type(e).__name__}: {e}")
        return {"output": figures, "log": "\n".join(log_lines)}


class CriticAgent:
    name = "Critic Agent"
    icon = "🧪"
    VALID_CHARTS = VALID_CHART_TYPES

    def _ask_llm_to_critique(self, col, dataset_name, ctype, dist, nuniq, n_rows, sample_vals, stylist_reason):
        prompt = f"""Review chart choice for '{col}' (dataset: '{dataset_name}').
Chosen: {ctype}, Reason: {stylist_reason}, Distribution: {dist}, Unique: {nuniq}/{n_rows}, Samples: {sample_vals}
Reply EXACTLY:
VERDICT: <APPROVE or REJECT>
REPLACEMENT: <chart if REJECT, else same>
CRITIQUE: <one sentence>"""
        response = ask_llama(prompt, self.name)
        verdict, replacement, critique = "APPROVE", ctype, "Approved as-is."
        for line in response.strip().splitlines():
            ls = line.strip()
            if ls.upper().startswith("VERDICT:"): verdict = "REJECT" if "REJECT" in ls.upper() else "APPROVE"
            elif ls.upper().startswith("REPLACEMENT:"):
                raw = ls.split(":", 1)[1].strip().lower()
                for v in self.VALID_CHARTS:
                    if v in raw: replacement = v; break
            elif ls.upper().startswith("CRITIQUE:"): critique = ls.split(":", 1)[1].strip()
        if verdict == "REJECT" and replacement == ctype: replacement = "bar"
        return verdict, replacement, critique

    def run(self, retriever_output, stylist_output, visualizer_output, datasets):
        log_lines = [f"{self.icon} {self.name} activated"]
        critiques, overrides = {}, {}
        for name, ctx in retriever_output.items():
            critiques[name], overrides[name] = {}, {}
            col_styles = stylist_output.get(name, {})
            for col, style in col_styles.items():
                if col not in datasets[name].columns: continue
                df = datasets[name]
                ctype, sr = style["chart_type"], style.get("reason", "")
                dist = ctx["dist_notes"].get(col, "N/A") if is_numeric_dtype(df[col]) else "categorical"
                nuniq, n_rows, sv = int(df[col].nunique()), len(df), df[col].dropna().head(SAMPLE_PREVIEW_ROWS).tolist()
                verdict, replacement, critique = self._ask_llm_to_critique(col, name, ctype, dist, nuniq, n_rows, sv, sr)
                critiques[name][col] = [f"Stylist: {ctype} — {sr}", f"Verdict: {verdict}", f"Critique: {critique}"]
                if verdict == "REJECT":
                    overrides[name][col] = replacement
                    log_lines.append(f"  🔄 {name} → '{col}': REJECTED '{ctype}' → '{replacement}'")
                else:
                    log_lines.append(f"  ✅ {name} → '{col}': APPROVED")
        total = sum(len(v) for v in overrides.values())
        return {"output": critiques, "overrides": overrides, "log": "\n".join(log_lines)}


class InsightAgent:
    name = "Insight Agent"
    icon = "🧠"

    def run(self, retriever_output, planner_output, critic_output, datasets):
        log_lines = [f"{self.icon} {self.name} activated"]
        parts = []
        for name, ctx in retriever_output.items():
            parts.append(f"\n=== {name} === Shape: {ctx['shape']}")
            parts.append(f"Numeric: {ctx['num_cols']}, Distributions: {ctx['dist_notes']}")
            parts.append(f"Stats: {ctx['describe']}")
            plan = planner_output.get(name, {})
            parts.append(f"Plan: {plan.get('steps', [])}, Rationale: {plan.get('rationale', '')}")
            parts.append(f"Critic overrides: {len(critic_output.get('overrides', {}).get(name, {}))}")
        context = "\n".join(parts)
        prompt = f"""Senior business consultant. Context from multi-agent AI system:

{context}

Produce unified strategic report:
1. Cross-Dataset Patterns
2. Key Business Risks (top 3)
3. Growth Opportunities
4. Data Quality Assessment
5. Recommended Actions (3 specific)

Treat ALL datasets as one business picture. Use bullet points. Be concise but strategic."""
        log_lines.append("  🔄 Generating unified report...")
        report = ask_llama(prompt, self.name)
        log_lines.append("  ✅ Report generated")
        return {"output": report, "log": "\n".join(log_lines)}


class RecommenderAgent:
    name = "Recommender Agent"
    icon = "💡"

    def run(self, insight_report):
        log_lines = [f"{self.icon} {self.name} activated"]
        prompt = f"""Dual-Role Consultant (CDO + COO). Based on this insight report:

{insight_report}

=== SECTION 1: Data & Technical Strategy (For IT Teams) ===
2 specific recommendations on data quality, normalization, or pipeline improvements.

=== SECTION 2: Business & Operational Directives (For Management) ===
2 specific operational recommendations (cost, efficiency, process).

Format each: - 📌 [Title] \\n- 🎯 Why: [reason] \\n- 🚀 Action Step: [step]"""
        log_lines.append("  🔄 Generating recommendations...")
        recs = ask_llama(prompt, self.name)
        log_lines.append("  ✅ Recommendations generated")
        return {"output": recs, "log": "\n".join(log_lines)}


class AgentState(TypedDict):
    datasets: Dict[str, pd.DataFrame]
    results: Dict[str, Any]
    overrides_applied: bool
    _has_re_rendered: bool


class Orchestrator:
    def __init__(self):
        self.retriever = RetrieverAgent()
        self.planner = PlannerAgent()
        self.stylist = StylistAgent()
        self.visualizer = VisualizerAgent()
        self.critic = CriticAgent()
        self.insight = InsightAgent()

    def run(self, datasets):
        results = {}
        results["retriever"] = self.retriever.run(datasets)
        results["planner"] = self.planner.run(results["retriever"]["output"])
        results["stylist"] = self.stylist.run(results["retriever"]["output"], results["planner"]["output"], datasets)
        results["visualizer"] = self.visualizer.run(datasets, results["stylist"]["output"])
        results["critic"] = self.critic.run(results["retriever"]["output"], results["stylist"]["output"],
                                              results["visualizer"]["output"], datasets)

        has_overrides = False
        stylist_out = results["stylist"]["output"]
        for ds_name, ovs in results["critic"]["overrides"].items():
            for col, new_type in ovs.items():
                if ds_name in stylist_out and col in stylist_out[ds_name]:
                    stylist_out[ds_name][col]["chart_type"] = new_type
                    has_overrides = True
        results["stylist"]["output"] = stylist_out

        if has_overrides:
            results["visualizer"] = self.visualizer.run(datasets, stylist_out)

        return results

    def run_insight(self, results, datasets):
        return self.insight.run(results["retriever"]["output"], results["planner"]["output"], results["critic"], datasets)


# ─── Triple-View Builder ───
VIEW_A = "view_a"
VIEW_B = "view_b"
VIEW_INTERSECTION = "view_intersection"

VIEW_LABELS = {
    VIEW_A: "Dataset A",
    VIEW_B: "Dataset B",
    VIEW_INTERSECTION: "Intersection (A ∩ B)",
}

def build_views(preprocessed: Dict[str, pd.DataFrame]):
    names = list(preprocessed.keys())
    views: Dict[str, Dict[str, pd.DataFrame]] = {}
    view_titles: Dict[str, str] = {}
    shared_cols: List[str] = []

    if len(names) == 2:
        name_a, name_b = names
        df_a, df_b = preprocessed[name_a], preprocessed[name_b]
        shared_cols = sorted(set(df_a.columns) & set(df_b.columns))
        short_a = name_a.rsplit(".", 1)[0] if "." in name_a else name_a
        short_b = name_b.rsplit(".", 1)[0] if "." in name_b else name_b

        views[VIEW_A] = {name_a: df_a}
        view_titles[VIEW_A] = f"Dataset A — {short_a}"
        views[VIEW_B] = {name_b: df_b}
        view_titles[VIEW_B] = f"Dataset B — {short_b}"

        if shared_cols:
            df_a_s = df_a[shared_cols].copy()
            df_b_s = df_b[shared_cols].copy()
            df_a_s["_source"] = short_a
            df_b_s["_source"] = short_b
            combined = pd.concat([df_a_s, df_b_s], ignore_index=True)
            views[VIEW_INTERSECTION] = {"Intersection (A ∩ B)": combined}
            view_titles[VIEW_INTERSECTION] = (
                f"Intersection — {short_a} ∩ {short_b} "
                f"({len(shared_cols)} shared col{'s' if len(shared_cols) != 1 else ''})"
            )
        return views, view_titles, shared_cols

    views["unified"] = preprocessed
    view_titles["unified"] = "Unified Analysis"
    return views, view_titles, shared_cols


def is_triple_view(views):
    return VIEW_A in views or VIEW_B in views


def get_view_keys(views):
    order = [VIEW_A, VIEW_B, VIEW_INTERSECTION]
    return [v for v in order if v in views]


def smart_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df_p, report = df.copy(), []
    df_p.columns = [str(c).strip().lower().replace(" ", "_") for c in df_p.columns]
    report.append("Standardized column names")
    dups = int(df_p.duplicated().sum())
    if dups > 0: df_p = df_p.drop_duplicates(); report.append(f"Removed {dups} duplicate rows")
    else: report.append("No duplicates found")
    converted = []
    for col in df_p.columns:
        od = df_p[col].dtype
        try: df_p[col] = pd.to_numeric(df_p[col], errors="raise")
        except: pass
        if df_p[col].dtype != od: converted.append(col)
    if converted: report.append(f"Converted to numeric: {', '.join(converted)}")
    for col in df_p.columns:
        mc = int(df_p[col].isnull().sum())
        if mc == 0: continue
        if is_numeric_dtype(df_p[col]):
            fv = df_p[col].median(); df_p[col] = df_p[col].fillna(fv)
            report.append(f"'{col}': {mc} nulls → median ({fv:.2f})")
        else:
            mv = df_p[col].mode()[0] if not df_p[col].mode().empty else "Unknown"
            df_p[col] = df_p[col].fillna(mv)
            report.append(f"'{col}': {mc} nulls → mode ('{mv}')")
    capped = []
    for col in df_p.select_dtypes(include=np.number).columns:
        q1, q3 = df_p[col].quantile(0.25), df_p[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0: continue
        lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
        n = int(((df_p[col] < lo) | (df_p[col] > hi)).sum())
        if n > 0: df_p[col] = df_p[col].clip(lower=lo, upper=hi); capped.append(f"'{col}' ({n})")
    if capped: report.append(f"Outliers capped: {', '.join(capped)}")
    else: report.append("No outliers needed capping")
    report.append(f"Final: {df_p.shape[0]} rows x {df_p.shape[1]} cols, nulls: {int(df_p.isnull().sum().sum())}")
    return df_p, report


def compute_unified_outliers(datasets):
    result = {}
    for name, df in datasets.items():
        result[name] = {}
        for col in df.select_dtypes(include=np.number).columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0: continue
            lo = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            hi = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n > 0: result[name][col] = n
    return result


def run_unified_anomaly(datasets):
    frames = []
    for name, df in datasets.items():
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols: continue
        sub = df[num_cols].dropna().copy()
        sub["_source"] = name
        frames.append(sub)
    if not frames: return None, []
    common_cols = list(set.intersection(*[set(f.columns) - {"_source"} for f in frames]))
    if not common_cols:
        all_results = []
        for f in frames:
            fc = [c for c in f.columns if c != "_source"]
            iso = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=ANOMALY_RANDOM_STATE)
            f["Status"] = np.where(iso.fit_predict(f[fc]) == -1, "Anomaly", "Normal")
            all_results.append(f)
        return pd.concat(all_results, ignore_index=True), []
    combined = pd.concat(frames, ignore_index=True)
    iso = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=ANOMALY_RANDOM_STATE)
    combined["Status"] = np.where(iso.fit_predict(combined[common_cols]) == -1, "Anomaly", "Normal")
    return combined, common_cols


def run_similarity_recommender(df, row_index, top_n=RECOMMENDER_TOP_N):
    num_df = df.select_dtypes(include=np.number).dropna()
    if num_df.empty or row_index not in num_df.index: return None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num_df)
    target_pos = num_df.index.get_loc(row_index)
    target_vector = scaled[target_pos].reshape(1, -1)
    sims = cosine_similarity(target_vector, scaled).flatten()
    order = sims.argsort()[::-1]
    order = [i for i in order if i != target_pos][:top_n]
    result = df.loc[num_df.index[order]].copy()
    result.insert(0, "similarity_score", [round(float(sims[i]), 4) for i in order])
    return result


def fig_to_json(fig):
    return json.loads(fig.to_json())

def fig_to_html(fig):
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})


def generate_pdf(insight_text, recommender_text, charts_dir):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(15, 12, 41)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_y(8)
    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, text="Smart Business Analytics", ln=True, align="C")
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, text="Executive AI Report & Visualizations", ln=True, align="C")
    pdf.ln(15)
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, text="1. Unified Strategic Insight Report", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    clean = str(insight_text).encode("latin-1", "ignore").decode("latin-1") if insight_text else "No data."
    pdf.multi_cell(0, 7, text=clean)
    pdf.ln(10)
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, text="2. Executive Recommendations", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    clean2 = str(recommender_text).encode("latin-1", "ignore").decode("latin-1") if recommender_text else "No data."
    pdf.multi_cell(0, 7, text=clean2)
    pdf.ln(10)
    pdf.add_page()
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, text="3. Key Data Visualizations", ln=True)
    pdf.ln(5)
    charts_found = False
    for i in range(PDF_MAX_CHARTS):
        img_path = os.path.join(charts_dir, f"chart_{i}.png")
        if not os.path.exists(img_path): continue
        charts_found = True
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, text=f"Chart {i+1}", ln=True)
        pdf.image(img_path, x=15, w=180)
        pdf.ln(5)
    if not charts_found:
        pdf.set_font("Arial", "I", 12)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, text="No charts generated.", ln=True)
    pdf_path = os.path.join(charts_dir, "Executive_Report.pdf")
    pdf.output(pdf_path)
    return pdf_path
