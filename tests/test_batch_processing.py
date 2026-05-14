"""Tests for Step 08 batch image processing."""

import pytest

from industrial_vision.batch_processing import (
    count_statuses,
    create_demo_batch_images,
    list_image_files,
    normalize_extensions,
    run_batch_inspection,
)
from industrial_vision.inspection import InspectionResult


def test_normalize_extensions_adds_dots_and_lowercase() -> None:
    """Extensions should be normalized for robust file filtering."""

    normalized = normalize_extensions(["PNG", ".JPG", "jpeg"])

    assert normalized == {".png", ".jpg", ".jpeg"}


def test_list_image_files_finds_supported_images(tmp_path) -> None:
    """The image listing helper should find supported image files."""

    (tmp_path / "a.png").write_bytes(b"fake")
    (tmp_path / "b.JPG").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    image_files = list_image_files(tmp_path)

    assert len(image_files) == 2
    assert image_files[0].suffix.lower() in {".png", ".jpg"}


def test_list_image_files_raises_for_missing_directory(tmp_path) -> None:
    """A missing input directory should produce a clear error."""

    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        list_image_files(missing_dir)


def test_create_demo_batch_images_creates_four_images(tmp_path) -> None:
    """The demo image generator should create a small batch of images."""

    created_images = create_demo_batch_images(tmp_path)

    assert len(created_images) == 4
    assert all(path.exists() for path in created_images)


def test_run_batch_inspection_processes_demo_images(tmp_path) -> None:
    """Batch inspection should process every demo image and save summary outputs."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    created_images = create_demo_batch_images(input_dir)
    results = run_batch_inspection(input_dir, output_dir)

    assert len(results) == len(created_images)
    assert all(isinstance(result, InspectionResult) for result in results)
    assert any(result.status == "PASS" for result in results)
    assert any(result.status == "FAIL" for result in results)
    assert (output_dir / "batch_inspection_summary.csv").exists()
    assert (output_dir / "json").exists()
    assert (output_dir / "visuals").exists()


def test_count_statuses_counts_pass_and_fail() -> None:
    """PASS and FAIL results should be counted correctly."""

    pass_result = InspectionResult(
        filename="clean.png",
        status="PASS",
        passed=True,
        defect_count=0,
        total_defect_area=0.0,
        max_defect_area=0.0,
        candidates=[],
    )
    fail_result = InspectionResult(
        filename="defective.png",
        status="FAIL",
        passed=False,
        defect_count=1,
        total_defect_area=100.0,
        max_defect_area=100.0,
        candidates=[],
    )

    counts = count_statuses([pass_result, fail_result, fail_result])

    assert counts["PASS"] == 1
    assert counts["FAIL"] == 2
