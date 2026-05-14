"""Tests for Step 07 inspection result export helpers."""

import csv
import json

from industrial_vision.inspection import DefectCandidate, InspectionResult
from industrial_vision.result_export import (
    result_to_summary_row,
    save_defect_candidates_csv,
    save_inspection_result_json,
    save_inspection_summary_csv,
)


def create_sample_result() -> InspectionResult:
    """Create a sample failed inspection result for export tests."""

    candidates = [
        DefectCandidate(
            x=10,
            y=20,
            width=30,
            height=40,
            area=900.12345,
            mean_intensity=35.6789,
        )
    ]

    return InspectionResult(
        filename="sample.png",
        status="FAIL",
        passed=False,
        defect_count=1,
        total_defect_area=900.12345,
        max_defect_area=900.12345,
        candidates=candidates,
    )


def test_result_to_summary_row_returns_flat_dictionary() -> None:
    """A result should be converted into a flat CSV-friendly row."""

    result = create_sample_result()

    row = result_to_summary_row(result)

    assert row["filename"] == "sample.png"
    assert row["status"] == "FAIL"
    assert row["passed"] is False
    assert row["defect_count"] == 1
    assert row["total_defect_area"] == 900.123


def test_save_inspection_result_json_writes_full_result(tmp_path) -> None:
    """The JSON export should write a full inspection result."""

    result = create_sample_result()
    output_path = tmp_path / "result.json"

    saved_path = save_inspection_result_json(result, output_path)

    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert data["filename"] == "sample.png"
    assert data["status"] == "FAIL"
    assert data["defect_count"] == 1
    assert len(data["candidates"]) == 1


def test_save_inspection_summary_csv_writes_summary_row(tmp_path) -> None:
    """The summary CSV should contain one row per inspection result."""

    result = create_sample_result()
    output_path = tmp_path / "summary.csv"

    saved_path = save_inspection_summary_csv([result], output_path)

    with saved_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert saved_path.exists()
    assert len(rows) == 1
    assert rows[0]["filename"] == "sample.png"
    assert rows[0]["status"] == "FAIL"
    assert rows[0]["defect_count"] == "1"


def test_save_defect_candidates_csv_writes_candidate_rows(tmp_path) -> None:
    """The candidate CSV should contain one row per defect candidate."""

    result = create_sample_result()
    output_path = tmp_path / "candidates.csv"

    saved_path = save_defect_candidates_csv(result, output_path)

    with saved_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert saved_path.exists()
    assert len(rows) == 1
    assert rows[0]["filename"] == "sample.png"
    assert rows[0]["candidate_index"] == "1"
    assert rows[0]["x"] == "10"


def test_save_defect_candidates_csv_handles_clean_result(tmp_path) -> None:
    """A clean result should still produce a candidates CSV with only a header."""

    result = InspectionResult(
        filename="clean.png",
        status="PASS",
        passed=True,
        defect_count=0,
        total_defect_area=0.0,
        max_defect_area=0.0,
        candidates=[],
    )
    output_path = tmp_path / "clean_candidates.csv"

    saved_path = save_defect_candidates_csv(result, output_path)

    with saved_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert saved_path.exists()
    assert rows == []
