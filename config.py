"""Central project paths and reproducibility settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"

RANDOM_STATE = 42
DEFAULT_DATASET_NAME = "ML-EdgeIIoT-dataset.csv"
DEFAULT_DATASET_PATH = DATA_RAW_DIR / DEFAULT_DATASET_NAME

# Create project output folders when the application/modules are imported.
for directory in (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODEL_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    BENCHMARKS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
