"""Portfolio release report helpers for the industrial vision project."""

from dataclasses import asdict, dataclass
from pathlib import Path


IMPORTANT_PROJECT_FILES = [
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "src/industrial_vision/dashboard.py",
    "src/industrial_vision/inference_api.py",
    "src/industrial_vision/inference_service.py",
    "src/industrial_vision/model_persistence.py",
    "src/industrial_vision/model_evaluation.py",
    "src/industrial_vision/ml_baseline.py",
    "src/industrial_vision/feature_extraction.py",
]


@dataclass(frozen=True)
class ReleaseFileStatus:
    """Status of one release file."""

    path: str
    exists: bool


@dataclass(frozen=True)
class ReleaseReport:
    """Portfolio release readiness report."""

    project_root: str
    total_checked: int
    existing_count: int
    missing_count: int
    files: list[ReleaseFileStatus]

    def to_dict(self) -> dict[str, object]:
        """Convert report into a dictionary."""

        return asdict(self)


def check_release_files(
    project_root: str | Path,
    relative_paths: list[str] | None = None,
) -> ReleaseReport:
    """Check whether important release files exist."""

    root = Path(project_root)
    paths = relative_paths if relative_paths is not None else IMPORTANT_PROJECT_FILES

    statuses = [
        ReleaseFileStatus(
            path=relative_path,
            exists=(root / relative_path).exists(),
        )
        for relative_path in paths
    ]

    existing_count = sum(1 for status in statuses if status.exists)
    total_checked = len(statuses)
    missing_count = total_checked - existing_count

    return ReleaseReport(
        project_root=str(root),
        total_checked=total_checked,
        existing_count=existing_count,
        missing_count=missing_count,
        files=statuses,
    )


def create_release_report_markdown(report: ReleaseReport) -> str:
    """Create a Markdown release readiness report."""

    rows = [
        f"| `{status.path}` | {'yes' if status.exists else 'no'} |"
        for status in report.files
    ]

    return f"""# Industrial Vision Portfolio Release Report

## Summary

| Metric | Value |
|---|---:|
| Total checked files | {report.total_checked} |
| Existing files | {report.existing_count} |
| Missing files | {report.missing_count} |

## Checked Files

| File | Exists |
|---|---|
{chr(10).join(rows)}

## Interpretation

If `missing_count` is `0`, the main portfolio files are present.

Before publishing the repository, also check:

- README is clear and professional
- screenshots are included
- commands are tested
- Docker section is correct
- no large generated files are accidentally committed
- no private credentials are included
"""


def save_release_report(
    report_markdown: str,
    output_path: str | Path,
) -> Path:
    """Save the release report as Markdown."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown, encoding="utf-8")

    return path


def run_release_report_workflow(
    project_root: str | Path,
    output_path: str | Path,
) -> tuple[ReleaseReport, Path]:
    """Run the release report workflow and save the Markdown result."""

    report = check_release_files(project_root)
    markdown = create_release_report_markdown(report)
    saved_path = save_release_report(markdown, output_path)

    return report, saved_path
