from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
DEFAULT_RANDOM_STATE = 42
TARGET_COLUMN = "Class"

