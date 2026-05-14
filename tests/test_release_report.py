"""Tests for Step 16 portfolio release report."""

from industrial_vision.release_report import (
    ReleaseFileStatus,
    ReleaseReport,
    check_release_files,
    create_release_report_markdown,
    run_release_report_workflow,
    save_release_report,
)


def test_check_release_files_counts_existing_and_missing_files(tmp_path) -> None:
    """Release file checker should count existing and missing files."""

    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim", encoding="utf-8")

    report = check_release_files(
        tmp_path,
        relative_paths=["README.md", "Dockerfile", "missing.txt"],
    )

    assert report.total_checked == 3
    assert report.existing_count == 2
    assert report.missing_count == 1


def test_check_release_files_returns_file_status_objects(tmp_path) -> None:
    """Release file checker should return per-file status objects."""

    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")

    report = check_release_files(
        tmp_path,
        relative_paths=["README.md"],
    )

    assert len(report.files) == 1
    assert isinstance(report.files[0], ReleaseFileStatus)
    assert report.files[0].path == "README.md"
    assert report.files[0].exists is True


def test_release_report_to_dict_contains_summary() -> None:
    """Release report should be convertible to a dictionary."""

    report = ReleaseReport(
        project_root=".",
        total_checked=1,
        existing_count=1,
        missing_count=0,
        files=[ReleaseFileStatus(path="README.md", exists=True)],
    )

    data = report.to_dict()

    assert data["total_checked"] == 1
    assert data["missing_count"] == 0
    assert data["files"][0]["path"] == "README.md"


def test_create_release_report_markdown_contains_checked_files() -> None:
    """Markdown release report should contain summary and file table."""

    report = ReleaseReport(
        project_root=".",
        total_checked=2,
        existing_count=1,
        missing_count=1,
        files=[
            ReleaseFileStatus(path="README.md", exists=True),
            ReleaseFileStatus(path="missing.txt", exists=False),
        ],
    )

    markdown = create_release_report_markdown(report)

    assert "Industrial Vision Portfolio Release Report" in markdown
    assert "`README.md`" in markdown
    assert "`missing.txt`" in markdown
    assert "Missing files" in markdown or "Missing" in markdown


def test_save_release_report_writes_markdown_file(tmp_path) -> None:
    """Release report markdown should be saved to disk."""

    output_path = tmp_path / "release_report.md"

    saved_path = save_release_report("# Report", output_path)

    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == "# Report"


def test_run_release_report_workflow_saves_report(tmp_path) -> None:
    """Full release report workflow should save a Markdown report."""

    (tmp_path / "README.md").write_text("# Demo", encoding="utf-8")
    output_path = tmp_path / "docs" / "release_report.md"

    report, saved_path = run_release_report_workflow(
        project_root=tmp_path,
        output_path=output_path,
    )

    assert report.total_checked > 0
    assert saved_path.exists()
    assert "Industrial Vision Portfolio Release Report" in saved_path.read_text(encoding="utf-8")


def test_release_report_marks_nested_files_as_existing(tmp_path) -> None:
    """Nested release files should be checked correctly."""

    nested_file = tmp_path / "src" / "industrial_vision" / "dashboard.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text('"""Dashboard."""', encoding="utf-8")

    report = check_release_files(
        tmp_path,
        relative_paths=["src/industrial_vision/dashboard.py"],
    )

    assert report.existing_count == 1
    assert report.missing_count == 0


def test_release_report_marks_nested_files_as_missing(tmp_path) -> None:
    """Missing nested release files should be reported correctly."""

    report = check_release_files(
        tmp_path,
        relative_paths=["src/industrial_vision/dashboard.py"],
    )

    assert report.existing_count == 0
    assert report.missing_count == 1
