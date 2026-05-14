"""Smoke tests for the industrial vision project setup."""

from industrial_vision.config import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT, ensure_project_directories


def test_project_directories_are_created() -> None:
    """The project should create its standard data and output directories."""

    ensure_project_directories()

    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert OUTPUTS_DIR.exists()
