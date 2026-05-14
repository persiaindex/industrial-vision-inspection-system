"""Step 16 command script: portfolio release report."""

from pathlib import Path

from industrial_vision.release_report import run_release_report_workflow


def main() -> None:
    """Generate a portfolio release readiness report."""

    project_root = Path.cwd()
    output_path = project_root / "docs" / "STEP_16_RELEASE_REPORT.md"

    report, saved_path = run_release_report_workflow(
        project_root=project_root,
        output_path=output_path,
    )

    print("Step 16 — Project Polish, README, Screenshots, and Portfolio Release")
    print(f"Project root: {report.project_root}")
    print(f"Total checked files: {report.total_checked}")
    print(f"Existing files: {report.existing_count}")
    print(f"Missing files: {report.missing_count}")
    print(f"Saved release report to: {saved_path}")

    if report.missing_count > 0:
        print("\nMissing files:")
        for status in report.files:
            if not status.exists:
                print(f"- {status.path}")


if __name__ == "__main__":
    main()
