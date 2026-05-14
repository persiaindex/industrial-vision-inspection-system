"""Project configuration paths.

This module keeps important project paths in one place.
Later steps will reuse these paths for loading images, saving outputs,
writing reports, and organizing evaluation results.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREPROCESSING_OUTPUT_DIR = OUTPUTS_DIR / "preprocessing"
PREDICTIONS_OUTPUT_DIR = OUTPUTS_DIR / "predictions"
EVALUATION_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
ERROR_ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "error_analysis"


def ensure_project_directories() -> None:
    """Create the standard project directories if they do not exist."""

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_IMAGES_DIR,
        OUTPUTS_DIR,
        PREPROCESSING_OUTPUT_DIR,
        PREDICTIONS_OUTPUT_DIR,
        EVALUATION_OUTPUT_DIR,
        ERROR_ANALYSIS_OUTPUT_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
