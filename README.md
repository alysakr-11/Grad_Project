# Smart Business Analytics Dashboard

Multi-agent AI analytics system with interactive visualizations, anomaly detection, and auto-generated business insights.

## Features

- **Interactive Chart Builder** — Custom bar, line, scatter, pie, histogram, box, and area charts with user-selected axes, aggregation functions, and color picker
- **Multi-Agent Pipeline** — Retriever → Planner → Visualizer → Critic agents that auto-generate charts and reports from uploaded data
- **Anomaly Detection** — Isolation Forest-based ML anomaly scanner with AI-powered explanations
- **Outlier Reporting** — IQR-based statistical outlier detection per column
- **AI Insights** — Ollama-powered business insight reports and recommendations
- **PDF Export** — Compile charts, insights, and recommendations into an executive PDF report
- **Triple-View Mode** — When two datasets share columns, analyzes them individually and at their intersection
- **Multi-Format Upload** — CSV, Excel, JSON, XML, Parquet, TXT

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Frontend | Vanilla JS, Plotly.js, CSS |
| ML/AI | scikit-learn (Isolation Forest), Ollama (LLM) |
| Data | pandas, NumPy |
| PDF | fpdf2, Kaleido |

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) (optional — for AI features)

### Setup

```bash
# Clone the repo
git clone https://github.com/alysakr-11/Grad_Project.git
cd Grad_Project

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate      # Windows
source venv/bin/activate    # macOS / Linux

# Install dependencies
pip install -r backend/requirements.txt
```

### Run

```bash
python launch.py
```

Or directly with Uvicorn:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `LLM_MODEL` | `llama3.2:1b` | Model name for AI insights |
| `LLM_TIMEOUT_SEC` | `300` | LLM request timeout |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Enable AI Features

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull llama3.2:1b`
3. Restart the dashboard — the AI Engine status shows **ONLINE**

## Usage

1. **Upload data** — Drag & drop or click to upload CSV/Excel/JSON/etc.
2. **Build charts** — Go to the Visualizations tab, pick a dataset, X/Y axes, and chart type
3. **Run pipeline** — Click "Run Agent Pipeline" to auto-generate charts, KPIs, outliers, and insights
4. **Explore** — Anomaly Detection, AI Insights, Recommender, and Data Preview tabs

## Project Structure

```
Grad_Project/
├── backend/
│   ├── main.py            # FastAPI app & API routes
│   ├── agents_core.py     # Multi-agent pipeline logic
│   ├── config.py          # Configuration constants
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── index.html         # Main dashboard page
│   ├── css/style.css      # Styles
│   └── js/app.js          # Frontend application logic
├── launch.py              # One-click launcher
├── start.bat              # Windows start script
├── .env                   # Environment configuration
└── .gitignore
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload files |
| POST | `/api/preprocess` | Preprocess uploaded data |
| POST | `/api/pipeline/run` | Run multi-agent pipeline |
| POST | `/api/chart/build` | Build interactive chart |
| GET | `/api/data/{session}/columns` | Get column info per dataset |
| GET | `/api/data/{session}/preview` | Data preview |
| POST | `/api/anomaly/run` | Run anomaly detection |
| POST | `/api/anomaly/explain` | AI explanation of anomalies |
| POST | `/api/pdf/generate` | Generate PDF report |
| GET | `/api/llm/status` | Check AI engine status |
| GET | `/api/health` | Health check |
