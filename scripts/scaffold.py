# scripts/scaffold.py
"""
Run from the project root with:  python scripts/scaffold.py

This script creates the entire src-layout skeleton for the clv-uplift project.
It is idempotent — running it twice will not overwrite files that already exist.
"""

from pathlib import Path

# Path(__file__) is the absolute path to THIS script file (scaffold.py).
# .resolve() makes it absolute even if called from a different directory.
# .parent is the scripts/ folder.
# .parent again is the project root (clv-uplift/).
ROOT = Path(__file__).resolve().parent.parent
print(f"Project root detected as: {ROOT}")

# ── 1. Create all directories ─────────────────────────────────────────────────
# These are every folder the project needs. mkdir(parents=True) creates
# intermediate folders automatically. exist_ok=True means no error if they
# already exist — making the script safe to run multiple times.

DIRS = [
    "src/clv_uplift/data",
    "src/clv_uplift/features",
    "src/clv_uplift/models",
    "src/clv_uplift/api",
    "streamlit_app",
    "notebooks",
    "tests",
    "data/raw",
    "artifacts",
    "scripts",
]

for d in DIRS:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
    print(f"  created  {d}/")

# ── 2. Write __init__.py files ────────────────────────────────────────────────
# An __init__.py file is what makes a folder a Python package.
# Without it, Python will not recognize the folder as importable.
# The top-level package gets a version string; sub-packages get a minimal docstring.

INITS = [
    "src/clv_uplift/__init__.py",
    "src/clv_uplift/data/__init__.py",
    "src/clv_uplift/features/__init__.py",
    "src/clv_uplift/models/__init__.py",
    "src/clv_uplift/api/__init__.py",
    "tests/__init__.py",
]

PKG_INIT  = '"""clv_uplift — CLV + Uplift Modeling Service."""\n__version__ = "0.1.0"\n'
EMPTY_INIT = '"""Sub-package."""\n'

for f in INITS:
    path = ROOT / f
    if not path.exists():   # never overwrite existing files
        content = PKG_INIT if f == "src/clv_uplift/__init__.py" else EMPTY_INIT
        path.write_text(content, encoding="utf-8")
        print(f"  wrote    {f}")

# ── 3. Write config.py ────────────────────────────────────────────────────────
# All magic numbers, paths, and environment-dependent settings live here.
# Any module that needs a threshold or a path imports from config, never
# hardcodes the value inline. This is the single source of truth principle.

CONFIG_PY = '''\
# src/clv_uplift/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# load_dotenv() reads the .env file (if present) and populates os.environ.
# This must happen before any os.getenv() calls below.
load_dotenv()

ROOT          = Path(__file__).resolve().parents[2]   # project root
DATA_DIR      = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Data constants ────────────────────────────────────────────────────────────
UCI_FILENAME      = "online_retail_II.xlsx"
SNAPSHOT_DATE_STR = "2011-12-10"          # one day after the last transaction

# ── Feature engineering ───────────────────────────────────────────────────────
RFM_QUANTILES = 4
CLV_SEGMENTS  = ["Champions", "Loyal", "At-Risk", "Lost"]

# ── Treatment design ──────────────────────────────────────────────────────────
TREATMENT_RECENCY_DAYS    = 90      # customer must have bought within 90 days
TREATMENT_SPEND_THRESHOLD = 500.0   # AND spent >= £500 total

# ── Model ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.2

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST   = os.getenv("API_HOST", "0.0.0.0")
API_PORT   = int(os.getenv("API_PORT", 8000))
MODEL_PATH = ARTIFACTS_DIR / "uplift_model.pkl"
'''

cfg_path = ROOT / "src/clv_uplift/config.py"
if not cfg_path.exists():
    cfg_path.write_text(CONFIG_PY, encoding="utf-8")
    print("  wrote    src/clv_uplift/config.py")

# ── 4. Write .env.example ─────────────────────────────────────────────────────
# .env.example is committed to git — it documents what variables exist.
# The actual .env file (with real values) is gitignored and never committed.

ENV_EXAMPLE = "API_PORT=\nSHAP_MAX_DISPLAY=\n"
env_ex = ROOT / ".env.example"
if not env_ex.exists():
    env_ex.write_text(ENV_EXAMPLE, encoding="utf-8")
    print("  wrote    .env.example")

# ── 5. Write pyproject.toml ───────────────────────────────────────────────────
# Modern Python packaging: one file replaces setup.py + requirements.txt.
# The [tool.setuptools.packages.find] where = ["src"] line is critical —
# it tells pip to look for packages inside the src/ directory.

PYPROJECT = '''\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "clv-uplift"
version = "0.1.0"
description = "Customer Lifetime Value + Uplift Modeling Service"
requires-python = ">=3.10"

dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pandas>=2.1",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "shap>=0.45",
    "python-dotenv>=1.0",
    "openpyxl>=3.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
    "pytest-asyncio>=0.23",
]

[tool.setuptools.packages.find]
where = ["src"]
'''

pp = ROOT / "pyproject.toml"
if not pp.exists():
    pp.write_text(PYPROJECT, encoding="utf-8")
    print("  wrote    pyproject.toml")

# ── 6. Write .gitignore ───────────────────────────────────────────────────────

GITIGNORE = """\
.env
.venv/
__pycache__/
*.pyc
*.pkl
*.egg-info/
dist/
.pytest_cache/
data/raw/
artifacts/
"""
gi = ROOT / ".gitignore"
if not gi.exists():
    gi.write_text(GITIGNORE, encoding="utf-8")
    print("  wrote    .gitignore")

print("\n✓ Scaffold complete.")
print("  Next step:  pip install -e '.[dev]'")