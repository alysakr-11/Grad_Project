import json, os, operator, re, requests, textwrap, subprocess, sys, io, base64, math
import warnings
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
        data = res.json()
        if "error" in data:
            err = data["error"]
            if "memory" in err.lower() or "available" in err.lower():
                return f"[{agent_name}] Model '{LLM_MODEL}' needs more memory than available ({err}). Try a smaller model: run `ollama pull llama3.2:1b` then set LLM_MODEL=llama3.2:1b in .env"
            return f"[{agent_name}] API Error: {err}"
        return data.get("response", "No response received.")
    except requests.exceptions.Timeout:
        return f"[{agent_name}] AI Engine timed out after {LLM_TIMEOUT_SEC}s."
    except requests.exceptions.ConnectionError:
        return f"[{agent_name}] AI Engine Offline"
    except Exception as e:
        return f"[{agent_name}] Error: {type(e).__name__}: {e}"


def check_llm_health() -> str:
    """Returns 'online' if Ollama is reachable, else 'offline'."""
    try:
        r = requests.get(LLM_TAGS_ENDPOINT, timeout=LLM_HEALTHCHECK_SEC)
        return "online" if r.status_code == 200 else "offline"
    except requests.exceptions.RequestException:
        return "offline"


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
        if ctype == "line" and not time_ordered and not is_num:
            return "bar", "Line on categorical without time index → bar"
        # scatter: always allow when user picks it — no time restriction
        if ctype in ("histogram", "box") and not is_num:
            return "bar", f"{ctype} needs numeric data → bar"
        return ctype, None

    @staticmethod
    def _make_fig(df, col, style, y_col=None, agg_func="count"):
        """
        Build a Plotly figure for a single column (auto-pipeline) or an X/Y pair
        (interactive chart builder).

        Parameters
        ----------
        df        : DataFrame
        col       : X-axis column (always required)
        style     : dict with keys chart_type / color
        y_col     : optional Y-axis column; when supplied the chart shows a real
                    numeric value instead of a count
        agg_func  : aggregation applied when grouping X → Y
                    one of "count" | "sum" | "mean" | "median" | "min" | "max"
                    Only used when y_col is set and chart type groups data (bar, pie, line).
        """
        ctype, color = style["chart_type"], style["color"]

        # ── helpers ──────────────────────────────────────────────────────────
        AGG_MAP = {
            "count":  "count",
            "sum":    "sum",
            "mean":   "mean",
            "median": "median",
            "min":    "min",
            "max":    "max",
        }
        safe_agg = AGG_MAP.get(agg_func, "mean")

        def _agg_df(group_col, value_col, limit=None):
            """Group group_col, aggregate value_col, return sorted DataFrame."""
            grp = df.groupby(group_col)[value_col]
            if safe_agg == "count":
                agg = grp.count()
            elif safe_agg == "sum":
                agg = grp.sum()
            elif safe_agg == "mean":
                agg = grp.mean()
            elif safe_agg == "median":
                agg = grp.median()
            elif safe_agg == "min":
                agg = grp.min()
            else:  # max
                agg = grp.max()
            agg = agg.sort_values(ascending=False)
            if limit:
                agg = agg.head(limit)
            return agg.reset_index()

        def _y_label():
            if not y_col:
                return "Count"
            return f"{safe_agg.capitalize()} of {y_col}"

        # ── histogram ────────────────────────────────────────────────────────
        if ctype == "histogram":
            if y_col and is_numeric_dtype(df[y_col]):
                # Overlay two numeric distributions
                fig = px.histogram(df, x=col, y=y_col, nbins=HISTOGRAM_BINS,
                                   histfunc=safe_agg if safe_agg != "median" else "avg",
                                   color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=_y_label())
            else:
                fig = px.histogram(df, x=col, nbins=HISTOGRAM_BINS,
                                   color_discrete_sequence=[color])
            fig.update_traces(marker_line_width=0, opacity=0.85)

        # ── box ──────────────────────────────────────────────────────────────
        elif ctype == "box":
            if y_col and is_numeric_dtype(df[y_col]):
                # Group distribution: X = category, Y = numeric values
                fig = px.box(df, x=col, y=y_col, color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=y_col)
            else:
                fig = px.box(df, y=col, color_discrete_sequence=[color])
            fig.update_traces(marker_size=4, line_width=2)

        # ── pie ──────────────────────────────────────────────────────────────
        elif ctype == "pie":
            if y_col and is_numeric_dtype(df[y_col]):
                agg_data = _agg_df(col, y_col, PIE_MAX_CATEGORIES)
                fig = px.pie(agg_data, names=col, values=y_col,
                             color_discrete_sequence=CHART_PALETTE, hole=0.45)
                fig.update_layout(title_text=f"{_y_label()} by {col}")
            else:
                vc = df[col].value_counts().head(PIE_MAX_CATEGORIES).reset_index()
                vc.columns = [col, "count"]
                fig = px.pie(vc, names=col, values="count",
                             color_discrete_sequence=CHART_PALETTE, hole=0.45)
            fig.update_traces(textfont=dict(family="Inter", size=12, color="#f8fafc"),
                              marker=dict(line=dict(color="#070912", width=2)))

        # ── line ─────────────────────────────────────────────────────────────
        elif ctype == "line":
            if y_col and is_numeric_dtype(df[y_col]):
                if is_numeric_dtype(df[col]):
                    # Both numeric: simple X vs Y line
                    sorted_df = df[[col, y_col]].dropna().sort_values(col)
                    fig = px.line(sorted_df, x=col, y=y_col,
                                  color_discrete_sequence=[color])
                else:
                    # Categorical X: aggregate then plot
                    agg_data = _agg_df(col, y_col)
                    fig = px.line(agg_data, x=col, y=y_col,
                                  color_discrete_sequence=[color])
                    fig.update_layout(yaxis_title=_y_label())
            else:
                fig = px.line(df, y=col, color_discrete_sequence=[color])
            fig.update_traces(line=dict(width=2.5), mode="lines+markers",
                              marker=dict(size=5))

        # ── scatter ──────────────────────────────────────────────────────────
        elif ctype == "scatter":
            if y_col and is_numeric_dtype(df[y_col]):
                fig = px.scatter(df, x=col, y=y_col,
                                 color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=y_col)
            else:
                fig = px.scatter(df, x=df.index, y=col,
                                 color_discrete_sequence=[color])
            fig.update_traces(marker=dict(size=7, opacity=0.75,
                              line=dict(width=0.5, color="#070912")))

        # ── bar (default) ─────────────────────────────────────────────────────
        else:
            if y_col and is_numeric_dtype(df[y_col]):
                agg_data = _agg_df(col, y_col, BAR_TOP_N)
                total_cats = int(df[col].nunique())
                suffix = f" (top {BAR_TOP_N} of {total_cats})" if total_cats > BAR_TOP_N else ""
                fig = px.bar(agg_data, x=col, y=y_col,
                             color_discrete_sequence=[color])
                fig.update_layout(yaxis_title=_y_label())
                fig.update_traces(marker_line_width=0)
                return style_fig(fig, title=f"{col} · {ctype}{suffix}")
            else:
                vc = df[col].value_counts().head(BAR_TOP_N).reset_index()
                vc.columns = [col, "count"]
                total_cats = int(df[col].nunique())
                suffix = f" (top {BAR_TOP_N} of {total_cats})" if total_cats > BAR_TOP_N else ""
                fig = px.bar(vc, x=col, y="count", color_discrete_sequence=[color])
                fig.update_traces(marker_line_width=0)
                return style_fig(fig, title=f"{col} · {ctype}{suffix}")

        # Build title
        title_y = f" vs {y_col}" if y_col else ""
        return style_fig(fig, title=f"{col}{title_y} · {ctype}")

    def run(self, datasets, stylist_output):
        log_lines = [f"{self.icon} {self.name} activated"]
        figures, guard_corrections = {}, 0
        for name, df in datasets.items():
            figures[name] = {}

            # Build a list of numeric columns to use as Y candidates for categorical X
            num_cols = [c for c in df.columns if is_numeric_dtype(df[c]) and c != "_source"]
            cat_cols = [c for c in df.columns if not is_numeric_dtype(df[c]) and c != "_source"]

            for col, style in stylist_output.get(name, {}).items():
                if col not in df.columns:
                    continue
                try:
                    original_type = style["chart_type"]
                    safe_type, gw = self._apply_safety_guards(df, col, original_type)
                    if safe_type != original_type:
                        style = {**style, "chart_type": safe_type}
                        guard_corrections += 1

                    # --- Smart Y pairing ---
                    # For categorical X columns, pick the best numeric Y so charts show
                    # real values instead of defaulting to value_counts().
                    # For numeric X columns that work better with another numeric Y
                    # (scatter/line), also pair them if possible.
                    y_col = None
                    agg_func = "mean"
                    col_is_cat = not is_numeric_dtype(df[col])

                    if col_is_cat and num_cols and safe_type in ("bar", "line", "pie"):
                        # Pick the numeric column with the highest variance as Y
                        best = max(num_cols, key=lambda c: df[c].std() if df[c].std() > 0 else 0)
                        y_col = best
                        agg_func = "mean"
                    elif not col_is_cat and safe_type in ("scatter",) and num_cols:
                        # For scatter on a numeric col, pair with the next best numeric col
                        candidates = [c for c in num_cols if c != col]
                        if candidates:
                            y_col = max(candidates, key=lambda c: abs(df[col].corr(df[c])) if df[c].std() > 0 else 0)
                            agg_func = "mean"

                    fig = self._make_fig(df, col, style, y_col=y_col, agg_func=agg_func)
                    figures[name][col] = {
                        "fig": fig, "type": safe_type, "color": style["color"],
                        "original_type": original_type, "guard_warning": gw,
                        "y_col": y_col, "agg_func": agg_func,
                    }
                    y_info = f" vs {y_col} [{agg_func}]" if y_col else ""
                    log_lines.append(
                        f"  {'🛡️' if gw else '✅'} {name} → '{col}'{y_info}: "
                        f"{'∼' if gw else ''}{safe_type}"
                        f"{f' (was {original_type})' if gw else ''}"
                    )
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


# ─── Helper: Domain Detection ───
DOMAIN_KEYWORDS = {
    "Finance": ["revenue", "profit", "cost", "expense", "income", "budget", "financial", "account", "audit", "tax", "invoice", "payment", "transaction", "balance", "cash", "asset", "liability", "equity", "depreciation", "amortization"],
    "Energy": ["toe", "consumption", "energy", "power", "fuel", "gas", "electricity", "emission", "carbon", "solar", "wind", "thermal", "kwh", "mw", "gwh", "steam", "coal", "oil", "renewable", "efficiency", "heat", "cooling", "generation", "load", "demand"],
    "Retail": ["sales", "customer", "product", "inventory", "order", "sku", "store", "price", "quantity", "category", "brand", "supplier", "warehouse", "shipment", "promotion", "discount", "margin", "revenue"],
    "HR": ["employee", "salary", "hiring", "turnover", "attrition", "department", "manager", "payroll", "bonus", "benefit", "promotion", "tenure", "headcount", "recruit", "training", "performance"],
    "Manufacturing": ["production", "defect", "quality", "machine", "downtime", "throughput", "yield", "batch", "assembly", "maintenance", "waste", "oee", "cycle", "output", "capacity"],
    "Healthcare": ["patient", "diagnosis", "treatment", "hospital", "doctor", "prescription", "surgery", "admission", "discharge", "bmi", "blood", "heart", "cancer", "diabetes", "symptom"],
    "Supply Chain": ["logistics", "supplier", "warehouse", "shipment", "carrier", "lead", "delivery", "route", "freight", "transport", "inventory", "stock", "dispatch", "container"],
}

# Directionality rules: tells the LLM whether a higher or lower value is desirable per domain
def _is_timestamp_col(series: pd.Series) -> bool:
    """Detect if a numeric column is actually a Unix timestamp (epoch nanoseconds/milliseconds)."""
    if not is_numeric_dtype(series):
        return False
    vals = series.dropna()
    if vals.empty:
        return False
    # Unix timestamps in seconds: ~1.5e9 (year 2018), ms: ~1.5e12, ns: ~1.5e18
    mean_val = vals.mean()
    if mean_val > 1e15:  # nanosecond timestamps
        return True
    if mean_val > 1e11:  # millisecond timestamps
        return True
    return False

def _is_conversion_rate_col(col_name: str) -> bool:
    """Detect if a column name suggests it's a fixed conversion rate/ratio."""
    name = col_name.lower().replace(" ", "_").replace("-", "_")
    keywords = ["conversion", "conversion_rate", "conversion_factor", "rate", "multiplier", "coefficient", "factor"]
    return any(kw == name or name.endswith("_" + kw) or name.startswith(kw + "_") or kw in name for kw in keywords)

def detect_domain(datasets: Dict[str, pd.DataFrame]) -> str:
    scores = {d: 0 for d in DOMAIN_KEYWORDS}
    for df in datasets.values():
        all_cols = " ".join(str(c).lower() for c in df.columns)
        all_vals = " ".join(str(df[c].dropna().astype(str).str.cat(sep=" ")).lower()[:2000] for c in df.columns if not is_numeric_dtype(df[c]))
        text = all_cols + " " + all_vals
        for domain, kws in DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw.lower() in text:
                    scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General Business"


# ═══════════════════════════════════════════════════════════════════════════
# 5. INSIGHT AGENT  — LLM-Powered Strategic Analysis
# Extracts data statistics programmatically, then uses LLM to generate
# a comprehensive strategic report. Adapts to ANY domain/dataset.
# ═══════════════════════════════════════════════════════════════════════════
class InsightAgent:
    """
    Agent 5 — Insight (robust, raw-accurate, domain-aware).

    The report is generated deterministically from the RAW data so every figure is
    exact and auditable. The local LLM is OFF by default; when enabled it only adds a
    short narrative that is validated and silently dropped if it fails.
    """

    name = "Insight Agent"
    icon = "🧠"

    DOMAIN_FRAMING = {
        "Energy": {
            "lens": "fuel mix, consumption volatility and cost/carbon exposure",
            "unit_note": "TOE (Tons of Oil Equivalent) is a standardised measure of energy "
                         "*consumed* to run operations — it is not energy lost or wasted.",
        },
        "Finance": {"lens": "revenue quality, cost efficiency and margin stability", "unit_note": ""},
        "Retail": {"lens": "sales performance, product mix and inventory health", "unit_note": ""},
        "HR": {"lens": "workforce composition, attrition and productivity", "unit_note": ""},
        "Healthcare": {"lens": "patient outcomes and operational efficiency", "unit_note": ""},
        "Manufacturing": {"lens": "throughput, yield and downtime", "unit_note": ""},
        "Supply Chain": {"lens": "lead times, fill rates and logistics performance", "unit_note": ""},
        "General Business": {"lens": "operational KPIs, trends and data quality", "unit_note": ""},
    }

    @staticmethod
    def _safe(val, default=0.0):
        if val is None:
            return default
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return default
        return val

    @staticmethod
    def _fmt(x):
        """Auditable full numbers with thousands separators (no M/B abbreviation)."""
        try:
            x = float(x)
        except (TypeError, ValueError):
            return str(x)
        if x != x or x in (float("inf"), float("-inf")):
            return "n/a"
        r = round(x, 2)
        if r == int(r):
            return f"{int(r):,}"
        return f"{r:,.2f}"

    @staticmethod
    def _get_direction(col, domain):
        cl = str(col).lower().replace(" ", "_")
        good = ["revenue", "profit", "income", "gain", "return", "yield", "efficiency",
                "productivity", "throughput", "output", "savings", "growth", "bonus",
                "satisfaction", "retention", "quality", "recovery", "survival", "conversion"]
        bad = ["cost", "expense", "debt", "loss", "liability", "waste", "defect", "error",
               "failure", "downtime", "consumption", "emission", "turnover", "attrition",
               "mortality", "complication", "readmission", "infection", "lead_time",
               "delay", "shortage", "toe"]
        if any(kw in cl for kw in good):
            return "higher_is_good"
        if any(kw in cl for kw in bad):
            return "higher_is_bad"
        return "neutral"

    @classmethod
    def _dir_text(cls, direction, col=None, domain=None):
        if domain == "Energy" and direction == "higher_is_bad":
            return "higher means more energy consumed (greater cost and emissions)"
        return {"higher_is_good": "higher is better",
                "higher_is_bad": "higher is worse"}.get(direction, "context-dependent")

    # ====================================================================
    def run(self, retriever_output, planner_output, critic_output, datasets,
            data_context=None, raw_datasets=None, use_llm=False):
        log_lines = [f"{self.icon} {self.name} activated"]
        try:
            source, used_raw = self._select_source(datasets, raw_datasets)
            if not source:
                return {"output": "## Strategic Report\n\nNo datasets were available to analyse.",
                        "log": "\n".join(log_lines + ["  No datasets provided."])}
            log_lines.append(f"  Analysing {'RAW' if used_raw else 'provided'} data ({len(source)} dataset(s)).")

            analysis = {name: self._prepare_analysis_frame(df) for name, df in source.items()}
            domain = detect_domain(analysis)
            self._domain = domain
            log_lines.append(f"  Domain: {domain}")

            dataset_info, dataset_findings, segments_map, dupes_map = {}, {}, {}, {}
            for name, df in analysis.items():
                ro = (retriever_output or {}).get(name, {})
                info = self._build_dataset_info(df, name, ro, domain)
                dataset_info[name] = info
                dataset_findings[name] = self._compute_findings(df, info, domain)
                pk = self._pick_primary_kpi(dataset_findings[name])
                segments_map[name] = self._compute_segments(df, info, domain, primary_kpi=pk)
                dupes_map[name] = self._near_duplicate_labels(df, info)
                log_lines.append(f"  '{name}': {info['rows']} rows, {len(info['kpi_cols'])} KPIs, "
                                 f"{len(info['cat_cols'])} categories"
                                 + (f", segmented by `{segments_map[name]['dimension']}`"
                                    if segments_map[name] else ""))

            cross = (self._compute_cross_comparison(dataset_findings, domain)
                     if len(analysis) >= 2 else {})
            # cross-source (A vs B) comparison for combined / intersection views — from RAW
            source_cmp = {}
            for name, df in analysis.items():
                pk = self._pick_primary_kpi(dataset_findings[name])
                sc = self._compute_source_comparison(df, pk, domain, raw_datasets)
                if sc:
                    source_cmp[name] = sc
            cap_note = self._capping_note(source, datasets, raw_datasets) if used_raw else ""

            headline = self._build_headline(dataset_findings, segments_map, cross, domain)
            narrative = self._build_narrative_det(domain, dataset_info, dataset_findings, segments_map, cross)
            if use_llm:
                ctx = self._build_llm_context(domain, dataset_info, dataset_findings, segments_map, cross)
                llm = self._try_llm_narrative(domain, ctx, dataset_info, log_lines)
                if llm:
                    narrative = llm

            sec1 = self._build_section1_overview(dataset_info, dataset_findings, segments_map, domain, analysis)
            sec2 = self._build_section2_key_findings(dataset_findings, segments_map, cross, domain, source_cmp)
            sec3 = self._build_section3_data_quality(dataset_info, dataset_findings, cap_note, dupes_map)
            sec4 = self._build_section4_summary(dataset_findings, segments_map, cross, domain, analysis, dupes_map)

            report = self._assemble(domain, headline, narrative, sec1, sec2, sec3, sec4)
            log_lines.append(f"  Report assembled ({len(report)} chars).")
            return {"output": report, "log": "\n".join(log_lines)}
        except Exception as e:
            import traceback
            log_lines.append(f"  InsightAgent error (safe fallback): {e}")
            log_lines.append("  " + traceback.format_exc().replace("\n", "\n  "))
            try:
                safe = self._emergency_report((raw_datasets or datasets or {}))
            except Exception:
                safe = "## Strategic Report\n\nThe analysis engine hit an error. Please check server logs."
            return {"output": safe, "log": "\n".join(log_lines)}

    # ---- source + cleaning ----
    @staticmethod
    def _select_source(datasets, raw_datasets):
        datasets = datasets or {}
        raw_datasets = raw_datasets or {}
        if not raw_datasets:
            return dict(datasets), False
        src, used_raw = {}, False
        for name, df in datasets.items():
            if name in raw_datasets:
                src[name] = raw_datasets[name]; used_raw = True
            else:
                src[name] = df
        if not src:
            return dict(raw_datasets), True
        return src, used_raw

    @staticmethod
    def _prepare_analysis_frame(df):
        out = df.copy()
        for col in out.columns:
            # Skip only columns that are ALREADY numeric or datetime; attempt coercion on
            # everything else — object AND the pandas-3 'str' dtype (which is != object).
            if is_numeric_dtype(out[col]) or pd.api.types.is_datetime64_any_dtype(out[col]):
                continue
            s = out[col].astype(str).str.strip()
            cleaned = (s.str.replace(",", "", regex=False)
                        .str.replace(r"\s+", "", regex=True)
                        .str.replace(r"[$€£¥%]", "", regex=True))
            num = pd.to_numeric(cleaned, errors="coerce")
            real = s.replace({"nan": np.nan, "None": np.nan, "": np.nan, "NaN": np.nan})
            denom = max(int(real.notna().sum()), 1)
            if real.notna().sum() > 0 and (num.notna().sum() / denom) >= 0.8:
                out[col] = num
        return out

    @staticmethod
    def _capping_note(source, provided, raw_datasets):
        if not raw_datasets or not provided:
            return ""
        capped_total, cols = 0, []
        for name, raw_df0 in source.items():
            prov = provided.get(name)
            if prov is None:
                continue
            raw_df = InsightAgent._prepare_analysis_frame(raw_df0)
            prov_map = {str(c).strip().lower().replace(" ", "_"): c for c in prov.columns}
            for c in raw_df.select_dtypes(include=np.number).columns:
                pc = prov_map.get(str(c).strip().lower().replace(" ", "_"))
                if pc is None or not is_numeric_dtype(prov[pc]):
                    continue
                a = pd.to_numeric(raw_df[c], errors="coerce")
                b = pd.to_numeric(prov[pc], errors="coerce")
                n = min(len(a), len(b))
                if n == 0:
                    continue
                diff = int((~np.isclose(a.iloc[:n].fillna(0).values, b.iloc[:n].fillna(0).values,
                                        rtol=1e-6, atol=1e-6)).sum())
                if diff > 0:
                    capped_total += diff; cols.append(str(c))
        if capped_total > 0:
            return (f"The dashboard's charts/KPIs run on a cleaned copy where **{capped_total:,}** extreme "
                    f"value(s) were capped (IQR clipping) in {', '.join('`%s`' % c for c in cols[:5])}"
                    f"{' …' if len(cols) > 5 else ''}. The figures in **this** report use the **raw** data, "
                    f"so those extremes remain visible below.")
        return ""

    # ---- optional LLM narrative ----
    def _try_llm_narrative(self, domain, ctx, dataset_info, log_lines):
        prompt = (f"You are a senior {domain} analyst. Using ONLY the verified statistics below, "
                  f"write a 2-paragraph executive narrative (max 180 words). Do NOT invent numbers, "
                  f"columns or concepts. For Energy, treat TOE as energy consumed, never 'loss'. "
                  f"No headings or bullets.\n\nVERIFIED STATS:\n{ctx}")
        text = ask_llama(prompt, self.name)
        ok, reason = self._validate_narrative(text, dataset_info, domain)
        if ok:
            log_lines.append("  LLM narrative accepted.")
            t = text.strip()
            while t[:1] == "#":
                t = "\n".join(t.splitlines()[1:]).strip()
            return t
        log_lines.append(f"  LLM narrative rejected ({reason}) — using deterministic narrative.")
        return ""

    @staticmethod
    def _validate_narrative(text, dataset_info, domain):
        if not isinstance(text, str):
            return False, "non-string"
        t = text.strip(); low = t.lower()
        if len(t) < 180 or t.startswith("["):
            return False, "too short / error marker"
        bad = ["ai engine offline", "timed out", "api error", "needs more memory",
               "no response received", "as an ai", "i cannot", "i'm sorry", "i am unable"]
        if any(b in low for b in bad):
            return False, "error/refusal"
        if domain == "Energy" and ("energy loss" in low or "energy lost" in low or
                                   ("consumption" in low and "loss" in low and "correlat" in low)):
            return False, "energy-loss hallucination"
        real_cols = []
        for info in dataset_info.values():
            real_cols += [str(c).lower() for c in info.get("kpi_cols", []) + info.get("cat_cols", [])]
        if real_cols and not any(c in low for c in set(real_cols)) and not any(ch.isdigit() for ch in t):
            return False, "ungrounded"
        return True, "ok"

    def _build_llm_context(self, domain, dataset_info, dataset_findings, segments_map, cross):
        p = [f"Domain: {domain}", self.DOMAIN_FRAMING.get(domain, {}).get("unit_note", "")]
        for name, info in dataset_info.items():
            p.append(f"[{name}] {info['rows']:,} rows")
            f = dataset_findings.get(name, {})
            for col, s in list(f.get("col_stats", {}).items())[:5]:
                cv = (abs(s["std"] / s["mean"]) * 100) if s["mean"] else 0
                p.append(f"  {col}: mean={self._fmt(s['mean'])}, median={self._fmt(s['median'])}, "
                         f"max={self._fmt(s['max'])}, CV={cv:.0f}%")
            for col, sp in list(f.get("spikes", {}).items())[:3]:
                p.append(f"  OUTLIER {col}: {self._fmt(sp['value'])} ({sp['distance_iqr']}x IQR)")
            seg = segments_map.get(name)
            if seg:
                tops = ", ".join(f"{r['value']} ({r['share']:.0f}%)" for r in seg["rows"][:4])
                p.append(f"  segments of {seg['kpi']} by {seg['dimension']}: {tops}")
        return "\n".join(x for x in p if x)

    # ---- metadata ----
    def _build_dataset_info(self, df, name, ro, domain):
        info = {"name": name, "rows": len(df), "cols": len(df.columns), "columns": list(df.columns)}
        date_col = self._detect_date_col(df)
        info["date_col"] = date_col
        info["date_range"] = "no date column"
        if date_col:
            try:
                s = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
                if not s.empty:
                    info["date_range"] = f"{s.min().date()} to {s.max().date()}"
            except Exception:
                pass
        all_num = [c for c in df.columns if is_numeric_dtype(df[c])]
        if date_col in all_num:
            all_num.remove(date_col)
        info["timestamp_cols"] = [c for c in all_num if _is_timestamp_col(df[c])]
        info["rate_cols"] = [c for c in all_num if _is_conversion_rate_col(c)]
        excluded = set(info["timestamp_cols"] + info["rate_cols"])
        info["kpi_cols"] = [c for c in all_num if c not in excluded]
        info["cat_cols"] = [c for c in df.columns if not is_numeric_dtype(df[c]) and c != date_col]
        info["nulls"] = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().sum() > 0}
        return info

    @staticmethod
    def _detect_date_col(df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                continue
            name = str(col).lower()
            datey = any(k in name for k in ["date", "time", "day", "month", "year", "timestamp", "period"])
            sample = df[col].dropna().astype(str).head(50)
            if sample.empty:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= (0.6 if datey else 0.9):
                return col
        return None

    # ---- analytics ----
    @staticmethod
    def _detect_spike_value(vals):
        if len(vals) < 10:
            return None
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return None
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        anomalies = vals[(vals < lo) | (vals > hi)]
        if anomalies.empty:
            return None
        median_v = vals.median()
        worst = anomalies.loc[anomalies.sub(median_v).abs().idxmax()]
        return {"value": round(float(worst), 2), "median": round(float(median_v), 2),
                "iqr": round(float(iqr), 2),
                "distance_iqr": round(float(abs(worst - median_v) / iqr), 1) if iqr else 0,
                "count": int(len(anomalies)), "is_high": bool(worst > median_v)}

    def _compute_findings(self, df, info, domain):
        f, kpi_cols, date_col = {}, info["kpi_cols"], info["date_col"]
        col_stats = {}
        for col in kpi_cols:
            vals = df[col].dropna()
            if len(vals) < 3:
                continue
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            col_stats[col] = {
                "mean": self._safe(round(float(vals.mean()), 2)),
                "median": self._safe(round(float(vals.median()), 2)),
                "std": self._safe(round(float(vals.std()), 2)),
                "min": self._safe(round(float(vals.min()), 2)),
                "max": self._safe(round(float(vals.max()), 2)),
                "q1": self._safe(round(float(q1), 2)), "q3": self._safe(round(float(q3), 2)),
                "iqr": self._safe(round(float(q3 - q1), 2)),
                "skew": self._safe(round(float(vals.skew()), 2), 0.0), "n": int(len(vals)),
                "sum": self._safe(round(float(vals.sum()), 2))}
        f["col_stats"] = col_stats
        f["spikes"] = {c: s for c in kpi_cols for s in [self._detect_spike_value(df[c].dropna())] if s}
        trends = {}
        if date_col and kpi_cols:
            try:
                sort_df = df.sort_values(date_col)
                half = len(sort_df) // 2
                if half > 0:
                    first, second = sort_df[kpi_cols].iloc[:half], sort_df[kpi_cols].iloc[half:]
                    for col in kpi_cols:
                        fm, sm = first[col].mean(), second[col].mean()
                        if pd.notna(fm) and pd.notna(sm) and fm != 0:
                            chg = ((sm - fm) / abs(fm)) * 100
                            if abs(chg) >= 0.5:
                                trends[col] = {"first_avg": round(float(fm), 2),
                                               "second_avg": round(float(sm), 2),
                                               "change_pct": round(float(chg), 1),
                                               "direction": "increase" if chg > 0 else "decrease"}
            except Exception:
                pass
        f["trends"] = trends
        return f

    def _pick_primary_dimension(self, info, df):
        cats = [c for c in info["cat_cols"] if str(c).lower() != "_source"]
        if not cats:
            return None
        prefer = ["sub_category", "subcategory", "category", "type", "segment", "fuel", "product",
                  "region", "item", "class", "group", "department", "channel", "sector", "source", "name"]
        cap = min(50, max(2, len(df) // 2))
        scored = []
        for c in cats:
            n = df[c].nunique(dropna=True)
            if n < 2 or n > cap:
                continue
            scored.append((any(p in str(c).lower() for p in prefer), -abs(n - 8), c))
        if scored:
            scored.sort(reverse=True)
            return scored[0][2]
        for c in cats:
            if 2 <= df[c].nunique(dropna=True) <= 50:
                return c
        return None

    def _compute_segments(self, df, info, domain, primary_kpi=None):
        dim = self._pick_primary_dimension(info, df)
        kpis = info["kpi_cols"]
        if not dim or not kpis:
            return {}
        sums = {c: abs(df[c].dropna().sum()) for c in kpis if len(df[c].dropna())}
        if not sums:
            return {}
        # anchor the breakdown to the metric the narrative discusses
        kpi = primary_kpi if (primary_kpi in sums) else max(sums, key=sums.get)
        try:
            g = df.groupby(dim)[kpi].agg(["count", "sum", "mean", "median"])
        except Exception:
            return {}
        total = float(df[kpi].sum())
        g = g.sort_values("sum", ascending=False).head(8)
        rows = [{"value": str(idx), "count": int(r["count"]), "sum": float(r["sum"]),
                 "mean": float(r["mean"]), "median": float(r["median"]),
                 "share": (float(r["sum"]) / total * 100) if total else 0.0}
                for idx, r in g.iterrows()]
        return {"dimension": dim, "kpi": kpi, "rows": rows,
                "n_categories": int(df[dim].nunique(dropna=True)), "total": total}

    @staticmethod
    def _lev(s, t):
        """Levenshtein edit distance with early exit (returns 3 when clearly > 2)."""
        if s == t:
            return 0
        m, n = len(s), len(t)
        if abs(m - n) > 2:
            return 3
        d = list(range(n + 1))
        for i in range(1, m + 1):
            prev, d[0] = d[0], i
            for j in range(1, n + 1):
                cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (s[i - 1] != t[j - 1]))
                prev = cur
            if min(d) > 2:
                return 3
        return d[n]

    @classmethod
    def _near_duplicate_labels(cls, df, info):
        """Flag categorical labels that differ by a tiny edit (<=2 characters) — almost certainly the
        same value mis-keyed (e.g. a dropped letter, 'افران' vs 'افرن'). Uses EDIT DISTANCE, not
        string-overlap ratio, so genuinely distinct labels that merely share a long prefix
        (e.g. '...الضواغط' = compressors vs '...المضخات' = pumps, edit distance 4) are NOT flagged."""
        out = []
        cats = [c for c in info["cat_cols"] if str(c).lower() != "_source"]
        for c in cats:
            vc = df[c].dropna().astype(str).value_counts()
            labels = list(vc.index)
            if not (2 <= len(labels) <= 80):
                continue
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a, b = labels[i], labels[j]
                    if a == b or min(len(a), len(b)) < 5:
                        continue
                    ed = cls._lev(a, b)
                    if 1 <= ed <= 2:
                        out.append({"col": str(c), "a": a, "b": b,
                                    "na": int(vc[a]), "nb": int(vc[b]), "ed": ed})
        return out[:5]

    def _compute_source_comparison(self, df, primary_kpi, domain, raw_datasets):
        """For a combined / intersection view (frame carries a `_source` column), compare the
        contributing datasets on the primary metric AND on a TOE/energy metric — read from each
        source's RAW data so figures are uncapped and the energy metric isn't lost to a name clash."""
        if "_source" not in df.columns or not primary_kpi:
            return {}
        srcs = [s for s in df["_source"].dropna().astype(str).unique()]
        if len(srcs) < 2:
            return {}

        def find_raw(sv):
            if not raw_datasets:
                return None
            for k, v in raw_datasets.items():
                ks = str(k); base = ks.rsplit(".", 1)[0]
                if ks == sv or ks.startswith(sv) or sv in ks or base == sv or sv.startswith(base):
                    return v
            return None

        def match_col(rdf, target):
            tn = str(target).strip().lower().replace(" ", "_")
            for c in rdf.columns:
                if str(c).strip().lower().replace(" ", "_") == tn:
                    return c
            return None

        def toe_col(rdf):
            for c in rdf.columns:
                cl = str(c).lower()
                if "toe" in cl and not _is_conversion_rate_col(c) and is_numeric_dtype(rdf[c]):
                    return c
            return None

        per, used_raw_any = {}, False
        for sv in srcs:
            rdf0 = find_raw(sv)
            if rdf0 is not None:
                used_raw_any = True
                rdf = self._prepare_analysis_frame(rdf0)
                pc = match_col(rdf, primary_kpi)
                v = rdf[pc].dropna() if pc is not None else pd.Series(dtype=float)
                tcol = toe_col(rdf)
                tv = rdf[tcol].dropna() if tcol is not None else None
                per[sv] = {"raw": True, "n": int(len(v)),
                           "mean": float(v.mean()) if len(v) else 0.0,
                           "median": float(v.median()) if len(v) else 0.0,
                           "sum": float(v.sum()) if len(v) else 0.0,
                           "max": float(v.max()) if len(v) else 0.0,
                           "toe_col": str(tcol) if tcol is not None else None,
                           "toe_mean": (float(tv.mean()) if tv is not None and len(tv) else None),
                           "toe_sum": (float(tv.sum()) if tv is not None and len(tv) else None)}
            else:
                sub = df[df["_source"].astype(str) == sv][primary_kpi].dropna()
                per[sv] = {"raw": False, "n": int(len(sub)),
                           "mean": float(sub.mean()) if len(sub) else 0.0,
                           "median": float(sub.median()) if len(sub) else 0.0,
                           "sum": float(sub.sum()) if len(sub) else 0.0,
                           "max": float(sub.max()) if len(sub) else 0.0,
                           "toe_col": None, "toe_mean": None, "toe_sum": None}
        grand = sum(p["sum"] for p in per.values())
        for p in per.values():
            p["share"] = (p["sum"] / grand * 100) if grand else 0.0
        has_toe = any(p["toe_mean"] is not None for p in per.values())
        return {"primary_kpi": primary_kpi, "per": per, "from_raw": used_raw_any, "has_toe": has_toe}

    def _compute_cross_comparison(self, dataset_findings, domain):
        names = list(dataset_findings.keys())
        if len(names) < 2:
            return {}
        shared = set(dataset_findings[names[0]]["col_stats"].keys())
        for n in names[1:]:
            shared &= set(dataset_findings[n]["col_stats"].keys())
        cross = {}
        for col in sorted(shared):
            vals = {n: dataset_findings[n]["col_stats"][col]["mean"] for n in names}
            lo, hi = min(vals.values()), max(vals.values())
            cross[col] = {"averages": vals, "gap_ratio": round(hi / lo, 2) if lo else None}
        return cross

    # ---- deterministic report ----
    def _pick_primary_kpi(self, findings):
        """Choose the metric the narrative, Finding 1, and the segment breakdown all anchor to.

        In **Energy**, prefer the standardised energy unit (TOE) when present — raw volumetric
        values (kWh, m³, litres …) are NOT comparable across fuels, so the largest-magnitude
        column is usually just the one measured in the biggest unit, not the most important one.
        Elsewhere, use the most *material* metric (largest absolute total)."""
        cs = findings.get("col_stats", {})
        if not cs:
            return None
        domain = getattr(self, "_domain", None)
        if domain == "Energy":
            toe = [c for c in cs if "toe" in str(c).lower()]
            if toe:
                return max(toe, key=lambda c: abs(cs[c].get("sum", cs[c]["mean"] * cs[c]["n"])))
        mag = {c: abs(s.get("sum", s["mean"] * s["n"])) for c, s in cs.items()}
        if any(mag.values()):
            return max(mag, key=mag.get)
        return max(cs, key=lambda c: abs(cs[c]["mean"]))

    @staticmethod
    def _is_volumetric(col, domain):
        """An Energy KPI that is a raw volumetric value (not the normalised TOE unit)."""
        return domain == "Energy" and "toe" not in str(col).lower()

    def _rank_kpis(self, findings, limit=3):
        """Primary first (by magnitude), then the remaining KPIs by risk signal
        (trend size, then outlier distance, then volatility)."""
        cs = findings.get("col_stats", {})
        if not cs:
            return []
        primary = self._pick_primary_kpi(findings)
        rest = [c for c in cs if c != primary]
        def sig(c):
            t = findings.get("trends", {}).get(c)
            sp = findings.get("spikes", {}).get(c)
            cv = abs(cs[c]["std"] / cs[c]["mean"]) if cs[c]["mean"] else 0
            return (abs(t["change_pct"]) if t else 0, sp["distance_iqr"] if sp else 0, cv)
        rest.sort(key=sig, reverse=True)
        return ([primary] + rest)[:limit]

    def _build_headline(self, dataset_findings, segments_map, cross, domain):
        best = None
        for dname, f in dataset_findings.items():
            for col, sp in f.get("spikes", {}).items():
                score = sp["distance_iqr"] * 8
                txt = (f"`{col}` in *{dname}* contains a "
                       f"{'severe' if sp['distance_iqr'] > 5 else 'moderate'} outlier of "
                       f"**{self._fmt(sp['value'])}** — **{sp['distance_iqr']}× the IQR** from the median of "
                       f"{self._fmt(sp['median'])} ({sp['count']} flagged).")
                if best is None or score > best[0]:
                    best = (score, txt)
            for col, t in f.get("trends", {}).items():
                d = self._get_direction(col, domain)
                unfav = (t["direction"] == "increase" and d == "higher_is_bad") or \
                        (t["direction"] == "decrease" and d == "higher_is_good")
                score = abs(t["change_pct"]) + (15 if unfav else 0)
                txt = (f"`{col}` in *{dname}* {t['direction']}d **{abs(t['change_pct']):.1f}%** "
                       f"({self._fmt(t['first_avg'])} → {self._fmt(t['second_avg'])}) — "
                       f"{'a concern' if unfav else 'favourable'}.")
                if best is None or score > best[0]:
                    best = (score, txt)
            seg = segments_map.get(dname)
            if seg and seg["rows"]:
                top = seg["rows"][0]
                if top["share"] >= 50 and len(seg["rows"]) > 1:
                    score = top["share"]
                    txt = (f"`{top['value']}` accounts for **{top['share']:.0f}%** of total "
                           f"`{seg['kpi']}` in *{dname}* — a single-segment concentration risk.")
                    if best is None or score > best[0]:
                        best = (score, txt)
        if best is None and cross:
            gaps = [(c, v["gap_ratio"]) for c, v in cross.items() if v["gap_ratio"]]
            if gaps:
                c, g = max(gaps, key=lambda x: x[1])
                best = (g, f"shared metric `{c}` differs **{g:.1f}×** between datasets.")
        if best is None:
            for dname, f in dataset_findings.items():
                pk = self._pick_primary_kpi(f)
                if pk:
                    s = f["col_stats"][pk]
                    best = (1, f"`{pk}` in *{dname}* averages **{self._fmt(s['mean'])}** "
                               f"(median {self._fmt(s['median'])}).")
                    break
        return best[1] if best else "No numeric KPIs were available to derive a headline finding."

    def _build_narrative_det(self, domain, dataset_info, dataset_findings, segments_map, cross):
        framing = self.DOMAIN_FRAMING.get(domain, self.DOMAIN_FRAMING["General Business"])
        total_rows = sum(i["rows"] for i in dataset_info.values())
        n_ds = len(dataset_info)
        primary_ds = max(dataset_findings, key=lambda d: len(dataset_findings[d].get("col_stats", {})),
                         default=None)
        para1 = (f"This {domain} diagnostic covers **{total_rows:,} record(s)** across {n_ds} dataset(s), "
                 f"examined through the lens of {framing['lens']}.")
        if framing.get("unit_note"):
            para1 += " " + framing["unit_note"]
        if primary_ds and dataset_findings[primary_ds].get("col_stats"):
            f = dataset_findings[primary_ds]
            pk = self._pick_primary_kpi(f)
            s = f["col_stats"][pk]
            cv = abs(s["std"] / s["mean"]) * 100 if s["mean"] else 0
            vol = ("extreme" if cv > 100 else "high" if cv > 50 else "moderate" if cv > 20 else "low")
            para1 += (f" The primary metric `{pk}` averages **{self._fmt(s['mean'])}** against a median of "
                      f"**{self._fmt(s['median'])}**, with **{vol} volatility (CV {cv:.0f}%)** and a range up to "
                      f"**{self._fmt(s['max'])}**.")
            seg = segments_map.get(primary_ds)
            if seg and (seg["rows"][0]["share"] >= 50 or cv > 100):
                top = seg["rows"][0]
                vol = (" (a volumetric figure, not energy-normalised — see TOE)"
                       if self._is_volumetric(seg["kpi"], domain) else "")
                para1 += (f" The aggregate blends heterogeneous segments, so it is best read per "
                          f"`{seg['dimension']}`: **{top['value']}** alone accounts for **{top['share']:.0f}%** "
                          f"of total `{seg['kpi']}`{vol}.")
        head = self._build_headline(dataset_findings, segments_map, {}, domain)
        total_severe = sum(1 for f in dataset_findings.values()
                           for s in f.get("spikes", {}).values() if s["distance_iqr"] > 5)
        any_outliers = sum(len(f.get("spikes", {})) for f in dataset_findings.values()) > 0
        if total_severe:
            para2 = (f"The dominant signal is a data-governance one: {head} Such multi-sigma readings are either "
                     f"genuine large-scale events or unit-entry errors, and must be triaged before these figures "
                     f"feed any {'sustainability/carbon' if domain == 'Energy' else 'executive'} reporting. "
                     f"Leadership should reject any 'stable portfolio' reading, isolate and validate the top "
                     f"outliers, then re-baseline on the cleansed, segment-level data.")
        elif any_outliers:
            para2 = (f"The headline is structural rather than a single break: {head} The flagged records and the "
                     f"segment mix below should be reviewed before averages are quoted to leadership, but no single "
                     f"reading is severe enough to invalidate the dataset.")
        else:
            para2 = (f"No multi-sigma outliers were detected: {head} The figures are internally consistent; focus "
                     f"on the trends and segment concentration highlighted below.")
        return para1 + "\n\n" + para2

    @staticmethod
    def _year_span(df, info):
        """Detect a year-like column (numeric or the date col) and return 'YYYY–YYYY'."""
        for c in df.columns:
            if "year" in str(c).lower() and is_numeric_dtype(df[c]):
                v = df[c].dropna()
                if len(v) and v.between(1900, 2100).mean() > 0.8:
                    return f"{int(v.min())}–{int(v.max())}"
        return None

    def _build_section1_overview(self, dataset_info, dataset_findings, segments_map, domain, stats_data):
        L = ["## 1. Domain & Operational Profile", ""]
        note = self.DOMAIN_FRAMING.get(domain, {}).get("unit_note", "")
        L.append(f"Classified as a **{domain}** problem from its column vocabulary and contents."
                 + (f" {note}" if note else ""))
        L.append("")
        total_rows = sum(len(df) for df in stats_data.values())
        total_kpis = sum(len(i["kpi_cols"]) for i in dataset_info.values())
        for name, info in dataset_info.items():
            L.append(f"**Dataset `{name}`** — {info['rows']:,} rows × {info['cols']} columns")
            if info["kpi_cols"]:
                L.append(f"- Numeric metrics ({len(info['kpi_cols'])}): "
                         + ", ".join(f"`{c}`" for c in info["kpi_cols"][:8])
                         + (" …" if len(info["kpi_cols"]) > 8 else ""))
            seg = segments_map.get(name)
            if seg:
                L.append(f"- Primary dimension `{seg['dimension']}` with **{seg['n_categories']}** categories "
                         f"(e.g. " + ", ".join(f"`{r['value']}`" for r in seg["rows"][:3]) + ")")
            elif info["cat_cols"]:
                L.append("- Categorical: " + ", ".join(f"`{c}`" for c in info["cat_cols"][:6]))
            if info["date_col"]:
                L.append(f"- Time column `{info['date_col']}` covering **{info['date_range']}**")
            else:
                span = self._year_span(stats_data[name], info)
                if span:
                    L.append(f"- Period covered: **{span}**")
            L.append("")
        L.append(f"**Scope:** {len(dataset_info)} dataset(s), **{total_rows:,}** records, "
                 f"**{total_kpis}** numeric KPIs.")
        return "\n".join(L)

    def _build_section2_key_findings(self, dataset_findings, segments_map, cross, domain, source_cmp=None):
        source_cmp = source_cmp or {}
        L = ["## 2. Definitive Statistical Findings", ""]
        if not any(f["col_stats"] for f in dataset_findings.values()):
            L.append("No numeric KPI columns were available for statistical analysis.")
            return "\n".join(L)
        n = 1
        for dname, f in dataset_findings.items():
            cs = f["col_stats"]
            if not cs:
                continue
            ranked = self._rank_kpis(f, limit=3)
            primary = ranked[0] if ranked else None
            for col in ranked:
                s = cs[col]
                cv = abs(s["std"] / s["mean"]) * 100 if s["mean"] else 0
                tag = " (primary metric)" if col == primary else ""
                L.append(f"**Finding {n} — `{col}`{tag} (dataset *{dname}*)**")
                L.append(f"- Centre: mean **{self._fmt(s['mean'])}**, median **{self._fmt(s['median'])}** (n={s['n']:,})")
                sk = s.get("skew", 0.0)
                if abs(sk) >= 0.5:
                    side = "right" if sk > 0 else "left"
                    tail = "upper tail of large values" if sk > 0 else "lower tail of small values"
                    L.append(f"- **{side.capitalize()}-skewed** (skew {sk:+.1f}): a long {tail} — "
                             f"the simple average is not representative of the typical record")
                L.append(f"- Spread: **{self._fmt(s['min'])} – {self._fmt(s['max'])}**, "
                         f"IQR {self._fmt(s['iqr'])}, σ {self._fmt(s['std'])}")
                band = ("**extreme**" if cv > 100 else "high" if cv > 50 else "moderate" if cv > 20 else "low")
                L.append(f"- Volatility: {band} (**CV {cv:.0f}%**) — "
                         f"{self._dir_text(self._get_direction(col, domain), col, domain)}")
                if self._is_volumetric(col, domain):
                    L.append("- ⚠️ *Volumetric metric:* this is a raw quantity that is **not unit-normalised "
                             "across fuels** (e.g. kWh vs m³), so its average and any cross-fuel share are not "
                             "energy-comparable — use **TOE** for energy comparisons")
                sp = f.get("spikes", {}).get(col)
                if sp:
                    L.append(f"- Outlier: extreme value **{self._fmt(sp['value'])}** "
                             f"({sp['distance_iqr']}× IQR from median; {sp['count']} record(s) flagged)")
                t = f.get("trends", {}).get(col)
                if t:
                    L.append(f"- Trend: **{self._fmt(t['first_avg'])} → {self._fmt(t['second_avg'])}** "
                             f"({t['direction']} of {abs(t['change_pct']):.1f}%)")
                L.append("")
                n += 1

            # cross-source (A vs B) comparison — the point of a combined view
            sc = source_cmp.get(dname)
            if sc:
                pk = sc["primary_kpi"]
                L.append(f"**Cross-source comparison — `{pk}` by dataset**"
                         + (" *(from each source's raw data)*" if sc["from_raw"] else "") + ":")
                for sv, p in sorted(sc["per"].items(), key=lambda x: -x[1]["sum"]):
                    L.append(f"- *{sv}*: {p['n']:,} records, total **{self._fmt(p['sum'])}** "
                             f"(**{p['share']:.1f}%**), mean {self._fmt(p['mean'])}, max {self._fmt(p['max'])}")
                if sc["has_toe"]:
                    toe_bits = []
                    for sv, p in sc["per"].items():
                        if p["toe_mean"] is not None:
                            toe_bits.append(f"*{sv}* mean {self._fmt(p['toe_mean'])} ({p['toe_col']})")
                    if toe_bits:
                        L.append("- Energy (TOE) per source: " + " | ".join(toe_bits))
                L.append("")

            # segment breakdown (anchored to the primary metric)
            seg = segments_map.get(dname)
            if seg and seg["rows"]:
                L.append(f"**Segment breakdown — `{seg['kpi']}` by `{seg['dimension']}`** "
                         f"({seg['n_categories']} categories):")
                for r in seg["rows"][:6]:
                    L.append(f"- `{r['value']}`: {r['count']:,} records, total **{self._fmt(r['sum'])}** "
                             f"(**{r['share']:.1f}%**), mean {self._fmt(r['mean'])}")
                top = seg["rows"][0]
                if top["share"] >= 50 and len(seg["rows"]) > 1:
                    vol = (" *(volumetric — not energy-normalised; see TOE)*"
                           if self._is_volumetric(seg["kpi"], domain) else "")
                    L.append(f"- ⚠️ **Concentration:** `{top['value']}` alone is **{top['share']:.0f}%** of total "
                             f"`{seg['kpi']}`{vol} — a single-segment dependency worth managing as a risk")
                L.append("")
        if cross:
            gaps = [(c, v["gap_ratio"]) for c, v in cross.items() if v["gap_ratio"]]
            if gaps:
                c, g = max(gaps, key=lambda x: x[1])
                avgs = " | ".join(f"*{nm}*: **{self._fmt(v)}**" for nm, v in cross[c]["averages"].items())
                L.append(f"**Finding {n} — cross-dataset gap on `{c}`**")
                L.append(f"- Disparity **{g:.1f}×** ({avgs})")
                L.append("")
        L.append("_Figures computed from the raw data using robust IQR-based methods._")
        return "\n".join(L)

    def _build_section3_data_quality(self, dataset_info, dataset_findings, cap_note, dupes_map=None):
        dupes_map = dupes_map or {}
        L = ["## 3. Critical Data Governance & Anomaly Flagging", ""]
        if cap_note:
            L += [f"> ⚙️ **Pipeline note:** {cap_note}", ""]
        # combined/intersection view: contributing datasets were preprocessed (and may have been capped)
        combined = [n for n, info in dataset_info.items() if "_source" in info.get("columns", [])]
        if combined and not cap_note:
            L += ["> ⚙️ **Combined-view note:** this view pools the contributing datasets and is built on the "
                  "preprocessing-cleaned copy, so extreme values from either source may have been capped "
                  "upstream and are not all visible here. Only metrics common to both datasets are included "
                  "(differently-named metrics, e.g. the TOE columns, are analysed in their per-dataset reports "
                  "and in the cross-source comparison above). Treat pooled figures as indicative, not exact.", ""]
        for name, info in dataset_info.items():
            L.append(f"**Dataset `{name}`**")
            concerns = []
            if info["nulls"]:
                tot = sum(info["nulls"].values())
                worst = sorted(info["nulls"].items(), key=lambda x: -x[1])[:4]
                concerns.append(f"Missing values ({tot:,}): " + ", ".join(f"`{c}`={v:,}" for c, v in worst)
                                + (" …" if len(info["nulls"]) > 4 else ""))
            dups = dupes_map.get(name) or []
            for d in dups:
                concerns.append(f"**Likely duplicate category** in `{d['col']}`: “{d['a']}” ({d['na']} rows) vs "
                                f"“{d['b']}” ({d['nb']} rows) differ by one or two characters — almost certainly the "
                                f"same value mis-keyed; reconcile before segmenting or reporting")
            f = dataset_findings.get(name, {})
            spikes = f.get("spikes", {})
            severe = {c: s for c, s in spikes.items() if s["distance_iqr"] > 5}
            moderate = [c for c, s in spikes.items() if s["distance_iqr"] <= 5]
            for c, s in sorted(severe.items(), key=lambda x: -x[1]["distance_iqr"]):
                concerns.append(f"**Severe outlier** in `{c}`: max **{self._fmt(s['value'])}** vs median "
                                f"{self._fmt(s['median'])} (**{s['distance_iqr']}× IQR**, {s['count']} record(s) flagged) — "
                                f"verify for unit-entry errors before regulatory/exec reporting")
            if moderate:
                tot_mod = sum(spikes[c]["count"] for c in moderate)
                concerns.append(f"Moderate outliers (1.5–5× IQR): {tot_mod} record(s) across "
                                + ", ".join(f"`{c}`" for c in moderate))
            hv = [c for c, s in f.get("col_stats", {}).items()
                  if s["mean"] and abs(s["std"] / s["mean"]) * 100 > 50]
            if hv:
                concerns.append("High-variability columns (CV>50%): " + ", ".join(f"`{c}`" for c in hv[:6]))
            issues = len(info["nulls"]) + len(severe) + (1 if moderate else 0) + len(dups)
            if severe:
                rating = "**Low** — severe multi-sigma outliers must be cleansed before relying on conclusions"
            elif dups:
                rating = "**Low** — duplicate category labels fragment the data; reconcile before relying on segment figures"
            elif issues == 0:
                rating = "**High** — no material concerns"
            elif issues <= 2:
                rating = "**Medium** — minor issues, analysis remains reliable"
            else:
                rating = "**Low** — multiple concerns; cleanse before relying on conclusions"
            L += [f"- {c}" for c in concerns] or ["- No data-quality concerns detected"]
            L.append(f"- Overall reliability: {rating}")
            L.append("")
        return "\n".join(L)

    def _build_section4_summary(self, dataset_findings, segments_map, cross, domain, stats_data, dupes_map=None):
        dupes_map = dupes_map or {}
        L = ["## 4. Strategic Recommendation for Leadership", ""]
        total_rows = sum(len(df) for df in stats_data.values())
        total_kpis = sum(len(f.get("col_stats", {})) for f in dataset_findings.values())
        metrics_with_outliers = sum(len(f.get("spikes", {})) for f in dataset_findings.values())
        records_flagged = sum(sp["count"] for f in dataset_findings.values()
                              for sp in f.get("spikes", {}).values())
        total_trends = sum(len(f.get("trends", {})) for f in dataset_findings.values())
        severe = sum(1 for f in dataset_findings.values()
                     for s in f.get("spikes", {}).values() if s["distance_iqr"] > 5)
        # top concentration across datasets
        top_conc = None
        for dn, seg in segments_map.items():
            if seg and seg["rows"]:
                r = seg["rows"][0]
                if top_conc is None or r["share"] > top_conc[1]:
                    top_conc = (r["value"], r["share"], seg["kpi"], seg["dimension"])
        n_dupes = sum(len(v or []) for v in dupes_map.values())

        L.append(f"- **Scope:** {len(stats_data)} dataset(s), {total_rows:,} records, {total_kpis} KPIs.")
        L.append("- **Trends:** " + (f"{total_trends} significant directional trend(s)."
                                     if total_trends else "no significant directional trends."))
        if metrics_with_outliers:
            tone = ("isolate and validate the severe one(s) first" if severe
                    else "material — investigate the flagged records before relying on averages")
            L.append(f"- **Anomalies:** **{records_flagged:,}** record-level outlier(s) across "
                     f"{metrics_with_outliers} metric(s) ({severe} severe) — {tone}.")
        else:
            L.append("- **Anomalies:** none detected — data is internally consistent.")
        if top_conc and top_conc[1] >= 50:
            vol = (" (volumetric — see TOE)" if self._is_volumetric(top_conc[2], domain) else "")
            L.append(f"- **Concentration:** `{top_conc[0]}` is **{top_conc[1]:.0f}%** of total "
                     f"`{top_conc[2]}`{vol} — single-segment dependency.")
        if n_dupes:
            L.append(f"- **Data quality:** {n_dupes} likely duplicate category label(s) detected — see Section 3.")
        if cross:
            mg = max((v["gap_ratio"] for v in cross.values() if v["gap_ratio"]), default=0)
            if mg > 3:
                L.append(f"- **Cross-dataset:** max metric gap **{mg:.1f}×** — "
                         f"{'significant' if mg > 10 else 'moderate'} structural difference.")
        L.append("")

        actions = []
        if severe:
            seg_hint = ""
            for dn, seg in segments_map.items():
                if seg:
                    seg_hint = (f" Then segment by `{seg['dimension']}` to localise which categories "
                                f"(e.g. `{seg['rows'][0]['value']}`) drive it.")
                    break
            actions.append("Reject any 'stable portfolio' reading and implement a data-cleansing filter that "
                           "isolates the top outliers for engineering review (unit-entry vs genuine event), then "
                           "re-baseline KPIs on the cleansed data." + seg_hint)
        if n_dupes:
            actions.append("Reconcile the duplicate category labels (standardise to a single spelling) before any "
                           "segment-level figures or carbon/sustainability reporting are published.")
        if top_conc and top_conc[1] >= 50:
            if self._is_volumetric(top_conc[2], domain):
                actions.append(f"Note that `{top_conc[0]}` is **{top_conc[1]:.0f}%** of total `{top_conc[2]}` in "
                               f"*volumetric* terms only — confirm the picture on a normalised energy (TOE) basis "
                               f"before treating it as a real dependency, since volumetric shares conflate units.")
            else:
                actions.append(f"Treat the **{top_conc[1]:.0f}% concentration** in `{top_conc[0]}` as a strategic "
                               f"dependency — assess supply/operational risk and whether diversification or a "
                               f"dedicated efficiency programme for that segment is warranted.")
        if not actions:
            actions.append("Stand up a recurring KPI review with documented targets for the primary metrics, and "
                           "monitor the highlighted trends and segment mix.")
        L.append("**Recommended action:** " + " ".join(actions))
        L.append("")
        L.append(f"_Automated {domain} diagnostic — figures computed from the raw data with robust statistics._")
        return "\n".join(L)

    def _emergency_report(self, datasets):
        L = ["## Strategic Report", ""]
        for name, df in datasets.items():
            df = self._prepare_analysis_frame(df)
            num = df.select_dtypes(include=np.number)
            L.append(f"**`{name}`** — {len(df):,} rows × {len(df.columns)} cols, "
                     f"{len(num.columns)} numeric, {int(df.isnull().sum().sum()):,} missing.")
            for c in list(num.columns)[:5]:
                v = df[c].dropna()
                if len(v):
                    L.append(f"- `{c}`: mean {self._fmt(v.mean())}, median {self._fmt(v.median())}, "
                             f"max {self._fmt(v.max())}")
            L.append("")
        return "\n".join(L)

    def _assemble(self, domain, headline, narrative, sec1, sec2, sec3, sec4):
        title = "📊 Strategic Energy Audit Brief" if domain == "Energy" else "🧠 Strategic Diagnostic Report"
        parts = [f"# {title} — {domain}", "", f"> **Headline:** {headline}", ""]
        parts += ["## Executive Narrative", "", narrative, ""]
        parts += [sec1, "", sec2, "", sec3, "", sec4]
        parts.append("\n---\n*Report generated deterministically from your raw data — every figure above "
                     "is computed directly from the file, not estimated.*")
        return "\n".join(parts)


class RecommenderAgent:
    """Agent 6 — data-grounded recommendations from RAW data (LLM optional, validated)."""

    name = "Recommender Agent"
    icon = "💡"

    @staticmethod
    def _compute_data_summary(datasets):
        ia = InsightAgent()
        parts = []
        for name, df in datasets.items():
            df = ia._prepare_analysis_frame(df)
            num = df.select_dtypes(include=np.number).columns.tolist()
            parts.append(f"Dataset '{name}': {df.shape[0]} rows, {len(num)} numeric")
            for col in num[:5]:
                v = df[col].dropna()
                if len(v) >= 3:
                    parts.append(f"  - {col}: mean={ia._fmt(v.mean())}, median={ia._fmt(v.median())}, "
                                 f"max={ia._fmt(v.max())}")
        return "\n".join(parts)

    def _build_deterministic_recs(self, datasets, domain):
        ia = InsightAgent()
        recs = []
        for name, raw in datasets.items():
            df = ia._prepare_analysis_frame(raw)
            info = ia._build_dataset_info(df, name, {}, domain)
            findings = ia._compute_findings(df, info, domain)
            seg = ia._compute_segments(df, info, domain)
            for col, sp in findings.get("spikes", {}).items():
                if sp["distance_iqr"] > 3:
                    recs.append((sp["distance_iqr"] * 10,
                        f"Triage the outlier in `{col}`",
                        f"`{col}` in *{name}* shows {sp['count']} outlier(s); the extreme is "
                        f"{ia._fmt(sp['value'])} ({sp['distance_iqr']}× IQR from median {ia._fmt(sp['median'])}).",
                        "Have data engineering review these records within 1 week to confirm genuine event vs "
                        "unit-entry error, then correct or annotate before any regulatory/exec reporting.",
                        f"A trustworthy `{col}` series; downstream KPIs stop being distorted by {sp['count']} record(s)."))
            for col, t in findings.get("trends", {}).items():
                d = ia._get_direction(col, domain)
                unfav = (t["direction"] == "increase" and d == "higher_is_bad") or \
                        (t["direction"] == "decrease" and d == "higher_is_good")
                if unfav and abs(t["change_pct"]) >= 5:
                    recs.append((abs(t["change_pct"]) + 20,
                        f"Address the {t['direction']} in `{col}`",
                        f"`{col}` moved {ia._fmt(t['first_avg'])} → {ia._fmt(t['second_avg'])} "
                        f"({abs(t['change_pct']):.1f}%), which is unfavourable.",
                        "Assign an owner to root-cause the shift this month with weekly tracking.",
                        f"Bring `{col}` back toward {ia._fmt(t['first_avg'])} next period."))
            for col, s in findings.get("col_stats", {}).items():
                if s["mean"] and abs(s["std"] / s["mean"]) * 100 > 80:
                    cv = abs(s["std"] / s["mean"]) * 100
                    extra = ""
                    if seg and seg["kpi"] == col:
                        extra = f" Segment by `{seg['dimension']}` (top: `{seg['rows'][0]['value']}`, {seg['rows'][0]['share']:.0f}%)."
                    recs.append((cv / 2,
                        f"Stabilise volatility in `{col}`",
                        f"`{col}` is highly volatile (CV {cv:.0f}%, σ {ia._fmt(s['std'])} around mean {ia._fmt(s['mean'])}).",
                        "Identify the drivers of the spread and tighten the high-variance process this quarter." + extra,
                        f"Reduce the coefficient of variation for `{col}` below 40%."))
            if info["nulls"]:
                tot = sum(info["nulls"].values())
                worst = max(info["nulls"], key=info["nulls"].get)
                recs.append((15 + min(tot, 100) / 10,
                    f"Close data gaps in *{name}*",
                    f"{tot:,} missing value(s) across {len(info['nulls'])} column(s); `{worst}` is worst.",
                    "Fix collection at source for the top fields and backfill within 2 weeks.",
                    "Raise completeness toward 100% so analyses aren't biased by gaps."))
        recs.sort(key=lambda r: r[0], reverse=True)
        seen, top = set(), []
        for r in recs:
            if r[1] in seen:
                continue
            seen.add(r[1]); top.append(r)
            if len(top) == 4:
                break
        fillers = [
            ("Stand up a KPI monitoring cadence", "Clear KPIs exist but no recurring review is implied.",
             "Create a weekly dashboard of the top metrics and assign an owner.",
             "Faster drift detection; decisions tied to current numbers."),
            ("Set baselines and targets", "Metrics lack documented targets to measure against.",
             "Define numeric targets for each primary KPI this quarter.",
             "Every metric becomes actionable against a goal."),
            ("Validate data definitions", "Column meanings are inferred and should be confirmed.",
             "Run a 1-hour data-dictionary review with stakeholders.", "Higher trust in every report."),
            ("Automate the analytics refresh", "Analysis appears ad-hoc rather than scheduled.",
             "Schedule the pipeline to refresh automatically.", "Always-current insights, no manual effort."),
        ]
        i = 0
        while len(top) < 4 and i < len(fillers):
            t, w, a, im = fillers[i]
            if t not in seen:
                top.append((0, t, w, a, im)); seen.add(t)
            i += 1
        out = []
        for i, (_, title, why, action, impact) in enumerate(top, 1):
            out += [f"**Recommendation {i}: {title}**", f"- **Why:** {why}",
                    f"- **Action:** {action}", f"- **Expected Impact:** {impact}", ""]
        return "\n".join(out).strip()

    def run(self, insight_report, datasets=None, raw_datasets=None, use_llm=False):
        log_lines = [f"{self.icon} {self.name} activated"]
        source, used_raw = InsightAgent._select_source(datasets, raw_datasets)
        domain = detect_domain({k: InsightAgent._prepare_analysis_frame(v) for k, v in source.items()}) \
            if source else "General Business"
        deterministic = self._build_deterministic_recs(source, domain)
        log_lines.append(f"  Built {deterministic.count('**Recommendation')} recs from "
                         f"{'RAW' if used_raw else 'provided'} data.")
        final = deterministic
        if use_llm:
            prompt = (f"You are a {domain} advisor. Improve ONLY the wording of the 4 recommendations below. "
                      f"Keep the same 4 items, numbers and structure. No new facts.\n\n{deterministic}")
            polished = ask_llama(prompt, self.name)
            if self._polished_is_valid(polished):
                final = polished.strip(); log_lines.append("  Using LLM-polished wording.")
            else:
                log_lines.append("  LLM polish rejected — using deterministic recs.")
        return {"output": final, "log": "\n".join(log_lines)}

    @staticmethod
    def _polished_is_valid(text):
        if not isinstance(text, str):
            return False
        t = text.strip()
        if t.startswith("[") or len(t) < 120:
            return False
        low = t.lower()
        if any(b in low for b in ["ai engine offline", "timed out", "api error", "as an ai", "i'm sorry"]):
            return False
        return t.count("**Recommendation") >= 3


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

    def run_insight(self, results, datasets, raw_datasets=None):
        return self.insight.run(results["retriever"]["output"], results["planner"]["output"], results["critic"], datasets, raw_datasets=raw_datasets)


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