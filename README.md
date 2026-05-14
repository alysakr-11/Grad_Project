# Smart Business Analytics Dashboard

Multi-agent AI analytics system with triple-view analysis (Dataset A, B, Intersection). Upload CSV/Excel files and get automated KPIs, visualizations, outlier detection, ML anomaly detection, and AI-powered business insights.

Built with FastAPI + vanilla HTML/CSS/JS — no build tools required.

## Features

- **6 AI Agents** — Retriever, Planner, Stylist, Visualizer, Critic, Insight + Recommender
- **Triple-View Analysis** — When 2 datasets share columns, analyzes A, B, and Intersection
- **Automated Visualizations** — Histograms, box plots, bar charts, pie charts, line/scatter via Plotly
- **Anomaly Detection** — Isolation Forest with per-view results and AI explanations
- **Smart Recommender** — Cosine similarity to find similar records across datasets
- **Executive PDF Reports** — AI insight + chart export
- **Ollama Integration** — Local LLM for agents (falls back gracefully when offline)

## Stack

| Layer | Tech |
|---|---|
| Backend | Python 3, FastAPI, Uvicorn |
| ML/Science | scikit-learn, pandas, numpy, Plotly |
| AI Engine | Ollama (llama3) |
| Frontend | Vanilla HTML, CSS, JavaScript |
| PDF | fpdf2 + kaleido |
| Auth/Env | python-dotenv |

## Quick Start

```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Start the app
python launch.py
# → Opens http://localhost:8000
```

Or manually:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Then visit http://localhost:8000
```

### Optional: AI Engine

```bash
ollama pull llama3
ollama serve
```

The app works without Ollama — agents return "AI Engine Offline" fallback messages.

## Usage

1. Drop CSV/Excel files on the upload zone
2. Click **Run Agent Pipeline**
3. Browse tabs: KPIs, Visualizations, Outliers, Anomaly Detection, AI Insights

Upload **2 files with shared columns** to activate triple-view mode.

## Project Structure

```
├── backend/
│   ├── main.py           # FastAPI entry point + all routes
│   ├── agents_core.py    # Orchestrator, 6 agents, anomaly, recommender, PDF
│   └── config.py         # Constants, env vars, palette, thresholds
├── frontend/
│   ├── index.html        # Dashboard UI
│   ├── css/style.css     # Aurora Executive dark theme
│   └── js/app.js         # All frontend logic
├── dataset/              # Sample CSV files
├── launch.py             # Convenience launcher
├── Procfile              # Render/Railway deployment
├── requirements.txt      # Root deps (includes backend/)
└── .env.example          # Env variable docs
```

## Deployment

### Render (free)

1. Push to GitHub
2. New Web Service → Connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### Railway / Heroku

Auto-detects `Procfile` and `requirements.txt`.

> Note: Ollama requires local GPU — AI insights show offline fallback on cloud deployments. All analytics features work without it.

## License

Graduation Project 2026
