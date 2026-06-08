import os
from pathlib import Path

# Load .env from project root (works regardless of how the server is started)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path))
except ImportError:
    pass

_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_ENDPOINT = f"{_OLLAMA_HOST}/api/generate"
LLM_TAGS_ENDPOINT = f"{_OLLAMA_HOST}/api/tags"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "300"))
LLM_HEALTHCHECK_SEC = int(os.getenv("LLM_HEALTHCHECK_SEC", "5"))

ANOMALY_CONTAMINATION = 0.05
ANOMALY_RANDOM_STATE = 42
OUTLIER_IQR_MULTIPLIER = 1.5
SKEW_NORMAL_THRESHOLD = 0.5
HISTOGRAM_BINS = 40
SAMPLE_PREVIEW_ROWS = 5
RECOMMENDER_TOP_N = 5

PIE_MAX_CATEGORIES = 8
BAR_MAX_CATEGORIES = 20
HISTOGRAM_MIN_UNIQUE = 6
SCATTER_NEEDS_TIME_INDEX = True
LINE_NEEDS_TIME_INDEX = True
BAR_TOP_N = 20
PDF_MAX_CHARTS = 3
PDF_CHART_WIDTH = 800
PDF_CHART_HEIGHT = 450

BRAND_CYAN = "#00e5ff"
BRAND_VIOLET = "#8b5cf6"
BRAND_PINK = "#ec4899"
BRAND_RED = "#ff5470"
BRAND_YELLOW = "#fbbf24"
BRAND_GREEN = "#10f5a4"
CHART_PALETTE = [BRAND_CYAN, BRAND_VIOLET, BRAND_PINK, BRAND_GREEN, BRAND_YELLOW,
                 "#f97316", "#3b82f6", "#ef4444", "#14b8a6", "#a855f7"]

VALID_CHART_TYPES = {"histogram", "box", "bar", "line", "scatter", "pie"}
SUPPORTED_FILE_TYPES = {"csv", "xlsx", "xls", "json", "xml", "parquet", "txt"}

UPLOAD_DIR = "uploads"
CHART_DIR = "charts"
