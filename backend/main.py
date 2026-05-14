import os, json, uuid, shutil, glob
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents_core import (
    Orchestrator, smart_preprocess, compute_unified_outliers,
    run_unified_anomaly, run_similarity_recommender,
    fig_to_json, fig_to_html, generate_pdf, check_llm_health,
    ask_llama, RecommenderAgent, SUPPORTED_FILE_TYPES,
    UPLOAD_DIR, CHART_DIR, PDF_MAX_CHARTS,
    build_views, is_triple_view, get_view_keys,
    VIEW_A, VIEW_B, VIEW_INTERSECTION
)

app = FastAPI(title="Smart Business Analytics Dashboard", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

sessions: Dict[str, Dict[str, Any]] = {}


def _load_file(file_path: str) -> Optional[pd.DataFrame]:
    lower = file_path.lower()
    try:
        if lower.endswith((".csv", ".txt")): return pd.read_csv(file_path)
        if lower.endswith((".xlsx", ".xls")): return pd.read_excel(file_path)
        if lower.endswith(".json"): return pd.read_json(file_path)
        if lower.endswith(".xml"): return pd.read_xml(file_path)
        if lower.endswith(".parquet"): return pd.read_parquet(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load {file_path}: {str(e)}")
    return None


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": check_llm_health()}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    uploaded = {}
    for file in files:
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in SUPPORTED_FILE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        uploaded[file.filename] = file_path

    sessions[session_id] = {
        "files": uploaded, "datasets": {}, "preprocessed": {},
        "prep_reports": {}, "views": {}, "view_titles": {},
        "shared_cols": [], "results_by_view": {}
    }
    return {"session_id": session_id, "files": list(uploaded.keys())}


@app.post("/api/preprocess")
def preprocess(session_id: str = Form(...)):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    raw = {}
    for name, path in sess["files"].items():
        df = _load_file(path)
        if df is not None: raw[name] = df
    if not raw: raise HTTPException(400, "No valid datasets")

    preprocessed, reports = {}, {}
    for name, df in raw.items():
        df_c, report = smart_preprocess(df)
        preprocessed[name] = df_c
        reports[name] = report
    sess["datasets"] = raw
    sess["preprocessed"] = preprocessed
    sess["prep_reports"] = reports

    # Build triple-views
    views, view_titles, shared_cols = build_views(preprocessed)
    sess["views"] = views
    sess["view_titles"] = view_titles
    sess["shared_cols"] = shared_cols

    summary = {}
    for name, df in preprocessed.items():
        summary[name] = {"rows": df.shape[0], "cols": df.shape[1], "columns": list(df.columns)}
    return {
        "summary": summary,
        "reports": reports,
        "views": {k: list(v.keys()) for k, v in views.items()},
        "view_titles": view_titles,
        "shared_cols": shared_cols,
        "triple_mode": len(preprocessed) == 2,
    }


@app.get("/api/views/{session_id}")
def get_views(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    return {
        "views": {k: list(v.keys()) for k, v in views.items()},
        "view_titles": sess.get("view_titles", {}),
        "shared_cols": sess.get("shared_cols", []),
        "triple_mode": is_triple_view(views),
        "view_keys": get_view_keys(views),
    }


@app.get("/api/data/{session_id}/preview")
def data_preview(session_id: str, dataset: str = "", rows: int = 10):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    datasets = sess.get("preprocessed") or sess.get("datasets", {})
    if dataset:
        df = datasets.get(dataset)
        if df is None: raise HTTPException(404, f"Dataset {dataset} not found")
        return {"columns": list(df.columns), "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "data": json.loads(df.head(rows).to_json(orient="records")),
                "shape": list(df.shape), "name": dataset}
    return {name: {"columns": list(df.columns), "shape": list(df.shape), "data": json.loads(df.head(rows).to_json(orient="records"))}
            for name, df in datasets.items()}


@app.post("/api/pipeline/run")
def run_pipeline(session_id: str = Form(...), run_insight: bool = Form(False)):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    if not views: raise HTTPException(400, "No views found. Upload and preprocess first.")
    view_titles = sess.get("view_titles", {})

    results_by_view = {}
    view_keys = list(views.keys())
    orch = Orchestrator()

    for vk in view_keys:
        view_datasets = views[vk]
        results = orch.run(view_datasets)
        if run_insight:
            insight_result = orch.run_insight(results, view_datasets)
            results["insight"] = insight_result
            rec = RecommenderAgent().run(insight_result["output"])
            results["recommender"] = rec
        results_by_view[vk] = results

    sess["results_by_view"] = results_by_view
    return {
        "status": "complete",
        "views": list(results_by_view.keys()),
        "view_titles": view_titles,
        "triple_mode": is_triple_view(views),
    }


@app.get("/api/results/{session_id}")
def get_results(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    if not results_by_view: return {"status": "no_results"}

    views = sess.get("views", {})
    view_titles = sess.get("view_titles", {})

    out = {}
    for vk, results in results_by_view.items():
        logs = {}
        for agent in ["retriever", "planner", "stylist", "visualizer", "critic"]:
            if agent in results:
                logs[agent] = results[agent].get("log", "")

        # Get datasets for this view
        view_ds = views.get(vk, {})
        kpi = {}
        for name, df in view_ds.items():
            num = df.select_dtypes(include=np.number)
            kpi[name] = {
                "rows": df.shape[0], "cols": df.shape[1],
                "numeric_cols": len(num.columns),
                "nulls": int(df.isnull().sum().sum()),
            }

        outliers = compute_unified_outliers(view_ds)
        out[vk] = {
            "kpi": kpi,
            "logs": logs,
            "outliers": outliers,
            "has_insight": "insight" in results,
            "has_recommender": "recommender" in results,
            "title": view_titles.get(vk, vk),
        }
    return {
        "status": "complete",
        "views": out,
        "view_keys": list(results_by_view.keys()),
        "view_titles": view_titles,
        "triple_mode": is_triple_view(views),
        "shared_cols": sess.get("shared_cols", []),
    }


@app.get("/api/charts/{session_id}")
def get_charts(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    view_titles = sess.get("view_titles", {})
    out = {}

    for vk, results in results_by_view.items():
        viz = results.get("visualizer", {}).get("output", {})
        charts_data = {}
        for ds_name, col_figs in viz.items():
            charts_data[ds_name] = {}
            for col, fig_data in col_figs.items():
                charts_data[ds_name][col] = {
                    "json": fig_to_json(fig_data["fig"]),
                    "type": fig_data["type"],
                    "original_type": fig_data.get("original_type", fig_data["type"]),
                    "guard_warning": fig_data.get("guard_warning"),
                }
        out[vk] = {"charts": charts_data, "title": view_titles.get(vk, vk)}

    return {"views": out, "view_keys": list(results_by_view.keys())}


@app.get("/api/insight/{session_id}")
def get_insight(session_id: str):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    out = {}
    for vk, results in results_by_view.items():
        insight = results.get("insight", {}).get("output", "")
        recommender = results.get("recommender", {}).get("output", "")
        out[vk] = {"insight": insight, "recommender": recommender}
    return {"views": out}


@app.post("/api/anomaly/run")
def run_anomaly(session_id: str = Form(...), view_key: str = Form("")):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    datasets = views.get(view_key, {}) if view_key else (sess.get("preprocessed") or sess.get("datasets", {}))
    if not datasets: raise HTTPException(400, "No data for this view")
    combined, common_cols = run_unified_anomaly(datasets)
    if combined is None: return {"error": "No numeric data"}
    anomaly_count = int((combined["Status"] == "Anomaly").sum())
    return {
        "total": len(combined), "anomalies": anomaly_count,
        "common_columns": common_cols,
        "breakdown": json.loads(combined.groupby(["_source", "Status"]).size().reset_index(name="count").to_json(orient="records"))
    }


@app.post("/api/anomaly/explain")
def explain_anomaly(session_id: str = Form(...), view_key: str = Form("")):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    views = sess.get("views", {})
    datasets = views.get(view_key, {}) if view_key else (sess.get("preprocessed") or sess.get("datasets", {}))
    if not datasets: return {"error": "No data"}
    combined, common_cols = run_unified_anomaly(datasets)
    if combined is None: return {"explanation": "No numeric data."}
    normal = combined[combined["Status"] == "Normal"]
    anomalies = combined[combined["Status"] == "Anomaly"]
    if anomalies.empty: return {"explanation": "No anomalies found."}
    feat = common_cols if common_cols else [c for c in combined.columns if c not in ("Status", "_source") and np.issubdtype(combined[c].dtype, np.number)]
    prompt = f"""Explain Isolation Forest anomalies.
Normal stats: {normal[feat].describe().to_string() if feat else 'N/A'}
Sample anomalies: {anomalies[feat].head(5).to_string() if feat else 'N/A'}
Explain: (1) why flagged, (2) which features deviate, (3) 3 business causes."""
    explanation = ask_llama(prompt, "Anomaly Explainer")
    return {"explanation": explanation}


@app.post("/api/recommender/similar")
def find_similar(session_id: str = Form(...), dataset: str = Form(...), row_index: int = Form(0), top_n: int = Form(5)):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    datasets = sess.get("preprocessed") or sess.get("datasets", {})
    df = datasets.get(dataset)
    if df is None: raise HTTPException(404, f"Dataset {dataset} not found")
    result = run_similarity_recommender(df, row_index, top_n)
    if result is None: return {"error": "Not enough numeric data"}
    return {"matches": json.loads(result.to_json(orient="records"))}


@app.post("/api/pdf/generate")
def generate_pdf_report(session_id: str = Form(...), view_key: str = Form("")):
    sess = sessions.get(session_id)
    if not sess: raise HTTPException(404, "Session not found")
    results_by_view = sess.get("results_by_view", {})
    if view_key and view_key in results_by_view:
        results = results_by_view[view_key]
    else:
        # Use first view with results
        results = next(iter(results_by_view.values())) if results_by_view else {}

    insight = results.get("insight", {}).get("output", "") if results else ""
    rec = results.get("recommender", {}).get("output", "") if results else ""
    viz = results.get("visualizer", {}).get("output", {}) if results else {}

    chart_save_dir = os.path.join(CHART_DIR, session_id)
    os.makedirs(chart_save_dir, exist_ok=True)
    count = 0
    for ds_name, col_figs in viz.items():
        for col, fig_data in col_figs.items():
            if count >= PDF_MAX_CHARTS: break
            try:
                fig_data["fig"].write_image(os.path.join(chart_save_dir, f"chart_{count}.png"),
                                            width=800, height=450, engine="kaleido")
                count += 1
            except: pass
        if count >= PDF_MAX_CHARTS: break

    pdf_path = generate_pdf(insight, rec, chart_save_dir)
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename="Executive_Report.pdf")
    raise HTTPException(500, "PDF generation failed")


@app.get("/api/llm/status")
def llm_status():
    return {"online": check_llm_health()}


@app.post("/api/ask")
def ask_llm(prompt: str = Form(...)):
    return {"response": ask_llama(prompt, "API")}


# Serve frontend as static files — mount AFTER all API routes so they take priority
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
