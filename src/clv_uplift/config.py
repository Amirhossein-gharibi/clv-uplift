# src/clv_uplift/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]
DATA_DIR      = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR   = ARTIFACTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data constants
UCI_FILENAME      = "online_retail_II.xlsx"
SNAPSHOT_DATE_STR = "2011-12-10"

# Feature engineering
RFM_QUANTILES = 4
CLV_SEGMENTS  = ["Champions", "Loyal", "At-Risk", "Lost"]

# Treatment design
TREATMENT_RECENCY_DAYS    = 90
TREATMENT_SPEND_THRESHOLD = 500.0

# Model
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# Features — canonical model feature set (single source of truth).
# Every module that needs these (uplift.py, api/schemas.py, api/routers/predict.py)
# imports FEATURE_COLS from here; no module defines its own copy.
# Excluded by design: rfm_score (collinear r+f+m), treatment, outcome (leakage),
# CustomerID (identifier), clv_segment (redundant encoding of r/f/m scores).
FEATURE_COLS = [
    "recency_days",
    "frequency",
    "monetary_value",
    "r_score",
    "f_score",
    "m_score",
    "cancel_rate",
]

# Policy economics — treat a customer iff incremental conversion prob * AOV > contact cost,
# i.e. CATE > cost / AOV = 0.50 / 20 = 0.025. Used by Phase 2.4 (prospective targeting-rate
# analysis) and any future /policy endpoint. NOT used to deploy a policy when the RATE test
# fails its gate.
BREAKEVEN_CATE = 0.025

# API
API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("API_PORT", 8000))
MODEL_PATH = ARTIFACTS_DIR / "uplift_model.pkl"