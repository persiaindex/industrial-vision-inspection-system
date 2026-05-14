"""Feature extraction helpers for industrial vision inspection."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from industrial_vision.batch_processing import list_image_files
from industrial_vision.edge_detection import apply_canny_edges, find_external_contours
from industrial_vision.image_io import load_image
from industrial_vision.inspection import inspect_image_rule_based
from industrial_vision.preprocessing import apply_gaussian_blur, convert_to_grayscale


@dataclass(frozen=True)
class ImageFeatureRow:
    """One row of numerical features extracted from an image."""

    filename: str
    label: str
    width: int
    height: int
    image_area: int
    mean_intensity: float
    std_intensity: float
    min_intensity: int
    max_intensity: int
    edge_pixel_count: int
    edge_density: float
    contour_count: int
    product_area_pixels: int
    defect_count: int
    total_defect_area: float
    max_defect_area: float
    defect_area_ratio: float
    largest_defect_width: int
    largest_defect_height: int
    largest_defect_mean_intensity: float

    def to_dict(self) -> dict[str, object]:
        """Convert the feature row into a dictionary."""

        return asdict(self)


def infer_label_from_filename(filename: str) -> str:
    """Infer a simple label from a filename.

    This is only for demo data. Real projects should use labels from
    an annotation file, database, or production system.
    """

    lowered = filename.lower()

    if "defect" in lowered or "fail" in lowered:
        return "defective"

    if "clean" in lowered or "pass" in lowered or "ok" in lowered:
        return "clean"

    return "unknown"


def count_nonzero_pixels(image: np.ndarray) -> int:
    """Count non-zero pixels in a single-channel image."""

    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got shape: {image.shape}")

    return int(cv2.countNonZero(image))


def extract_image_features(
    image: np.ndarray,
    filename: str,
    label: str | None = None,
) -> ImageFeatureRow:
    """Extract numerical features from one image."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a BGR image with 3 channels, got shape: {image.shape}")

    height, width, _ = image.shape
    image_area = int(width * height)

    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale, kernel_size=(5, 5))

    edges = apply_canny_edges(
        blurred,
        low_threshold=50,
        high_threshold=150,
    )
    contours = find_external_contours(edges)

    result, product_mask, _ = inspect_image_rule_based(
        image,
        filename=filename,
        product_threshold=100,
        dark_defect_threshold=80,
        min_product_area=1000.0,
        min_defect_area=50.0,
    )

    product_area_pixels = count_nonzero_pixels(product_mask)
    edge_pixel_count = count_nonzero_pixels(edges)

    edge_density = edge_pixel_count / image_area if image_area else 0.0
    defect_area_ratio = (
        result.total_defect_area / product_area_pixels if product_area_pixels else 0.0
    )

    largest_candidate = result.candidates[0] if result.candidates else None

    final_label = label if label is not None else infer_label_from_filename(filename)

    return ImageFeatureRow(
        filename=filename,
        label=final_label,
        width=width,
        height=height,
        image_area=image_area,
        mean_intensity=float(grayscale.mean()),
        std_intensity=float(grayscale.std()),
        min_intensity=int(grayscale.min()),
        max_intensity=int(grayscale.max()),
        edge_pixel_count=edge_pixel_count,
        edge_density=float(edge_density),
        contour_count=len(contours),
        product_area_pixels=product_area_pixels,
        defect_count=result.defect_count,
        total_defect_area=float(result.total_defect_area),
        max_defect_area=float(result.max_defect_area),
        defect_area_ratio=float(defect_area_ratio),
        largest_defect_width=largest_candidate.width if largest_candidate else 0,
        largest_defect_height=largest_candidate.height if largest_candidate else 0,
        largest_defect_mean_intensity=(
            float(largest_candidate.mean_intensity) if largest_candidate else 0.0
        ),
    )


def extract_features_from_image_file(image_path: str | Path) -> ImageFeatureRow:
    """Load one image file and extract its feature row."""

    path = Path(image_path)
    image = load_image(path)

    return extract_image_features(
        image,
        filename=path.name,
        label=infer_label_from_filename(path.name),
    )


def feature_rows_to_dataframe(rows: Iterable[ImageFeatureRow]) -> pd.DataFrame:
    """Convert feature rows into a Pandas DataFrame."""

    return pd.DataFrame([row.to_dict() for row in rows])


def extract_feature_dataset(
    input_dir: str | Path,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract features for all images in a folder.

    If output_csv_path is provided, the feature table is saved as CSV.
    """

    image_files = list_image_files(input_dir)
    rows = [extract_features_from_image_file(image_path) for image_path in image_files]
    dataframe = feature_rows_to_dataframe(rows)

    if output_csv_path is not None:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)

    return dataframe
