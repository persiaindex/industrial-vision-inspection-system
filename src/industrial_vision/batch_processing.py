"""Batch image processing helpers for industrial vision inspection."""

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from industrial_vision.image_io import load_image, save_image
from industrial_vision.inspection import InspectionResult, draw_inspection_result, inspect_image_rule_based
from industrial_vision.result_export import (
    save_defect_candidates_csv,
    save_inspection_result_json,
    save_inspection_summary_csv,
)


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def normalize_extensions(extensions: Iterable[str]) -> set[str]:
    """Normalize image extensions to lowercase values starting with a dot."""

    normalized_extensions: set[str] = set()

    for extension in extensions:
        extension = extension.lower().strip()

        if not extension.startswith("."):
            extension = f".{extension}"

        normalized_extensions.add(extension)

    return normalized_extensions


def list_image_files(
    input_dir: str | Path,
    extensions: Iterable[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> list[Path]:
    """List supported image files recursively from an input directory."""

    directory = Path(input_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {directory}")

    normalized_extensions = normalize_extensions(extensions)

    image_files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    ]

    return sorted(image_files)


def create_synthetic_inspection_image(
    output_path: str | Path,
    with_defect: bool,
    defect_center: tuple[int, int] = (330, 175),
    defect_radius: int = 28,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Create one synthetic inspection image.

    The image simulates a bright product on a dark background.
    Optional dark spots simulate defect candidates.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.rectangle(image, (140, 80), (500, 280), color=(210, 210, 210), thickness=-1)
    cv2.rectangle(image, (140, 80), (500, 280), color=(255, 255, 255), thickness=3)

    if with_defect:
        cv2.circle(
            image,
            center=defect_center,
            radius=defect_radius,
            color=(35, 35, 35),
            thickness=-1,
        )

    label = "DEFECT" if with_defect else "CLEAN"

    cv2.putText(
        image,
        label,
        (255, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    return save_image(image, path)


def create_demo_batch_images(output_dir: str | Path) -> list[Path]:
    """Create a small demo batch with clean and defective sample images."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    image_specs = [
        ("part_clean_001.png", False, (330, 175), 28),
        ("part_defective_001.png", True, (330, 175), 28),
        ("part_defective_002.png", True, (250, 145), 20),
        ("part_clean_002.png", False, (330, 175), 28),
    ]

    created_paths: list[Path] = []

    for filename, with_defect, defect_center, defect_radius in image_specs:
        image_path = directory / filename
        created_paths.append(
            create_synthetic_inspection_image(
                image_path,
                with_defect=with_defect,
                defect_center=defect_center,
                defect_radius=defect_radius,
            )
        )

    return created_paths


def inspect_single_image_file(
    image_path: str | Path,
    output_dir: str | Path,
    save_outputs: bool = True,
) -> InspectionResult:
    """Inspect one image file and optionally save per-image outputs."""

    path = Path(image_path)
    output_directory = Path(output_dir)
    image = load_image(path)

    result, product_mask, defect_mask = inspect_image_rule_based(
        image,
        filename=path.name,
        product_threshold=100,
        dark_defect_threshold=80,
        min_product_area=1000.0,
        min_defect_area=50.0,
    )

    if save_outputs:
        visual_result = draw_inspection_result(image, result)

        json_dir = output_directory / "json"
        candidates_dir = output_directory / "candidates"
        masks_dir = output_directory / "masks"
        visuals_dir = output_directory / "visuals"

        save_inspection_result_json(result, json_dir / f"{path.stem}.json")
        save_defect_candidates_csv(result, candidates_dir / f"{path.stem}_candidates.csv")
        save_image(product_mask, masks_dir / f"{path.stem}_product_mask.png")
        save_image(defect_mask, masks_dir / f"{path.stem}_defect_mask.png")
        save_image(visual_result, visuals_dir / f"{path.stem}_inspection.png")

    return result


def run_batch_inspection(
    input_dir: str | Path,
    output_dir: str | Path,
    extensions: Iterable[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> list[InspectionResult]:
    """Run rule-based inspection for all supported images in a folder."""

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    image_files = list_image_files(input_dir, extensions=extensions)

    results: list[InspectionResult] = []

    for image_path in image_files:
        result = inspect_single_image_file(
            image_path,
            output_directory,
            save_outputs=True,
        )
        results.append(result)

    save_inspection_summary_csv(
        results,
        output_directory / "batch_inspection_summary.csv",
    )

    return results


def count_statuses(results: Iterable[InspectionResult]) -> dict[str, int]:
    """Count PASS and FAIL results."""

    counts = {"PASS": 0, "FAIL": 0}

    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    return counts
