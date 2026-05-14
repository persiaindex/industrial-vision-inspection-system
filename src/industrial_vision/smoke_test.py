"""First executable check for the industrial vision project."""

from industrial_vision.config import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT, ensure_project_directories


def main() -> None:
    """Run a simple project setup check."""

    ensure_project_directories()

    print("Industrial Vision Inspection System - smoke test OK")
    print(f"Project root exists: {PROJECT_ROOT.exists()}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    print(f"Outputs directory exists: {OUTPUTS_DIR.exists()}")


if __name__ == "__main__":
    main()
