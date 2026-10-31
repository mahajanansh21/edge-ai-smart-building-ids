from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
)


def test_project_root_exists():
    assert PROJECT_ROOT.is_dir()


def test_required_workspace_directories_exist():
    for path in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODEL_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ):
        assert path.is_dir()
