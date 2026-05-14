"""Tests for Step 09 feature extraction."""

import cv2
import numpy as np
import pandas as pd

from industrial_vision.batch_processing import create_demo_batch_images
from industrial_vision.feature_extraction import (
    count_nonzero_pixels,
    extract_feature_dataset,
    extract_image_features,
    extract_features_from_image_file,
    feature_rows_to_dataframe,
    infer_label_from_filename,
)


def create_test_image(with_defect: bool = True) -> np.ndarray:
    """Create a synthetic BGR image for feature extraction tests."""

    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (170, 120), color=(210, 210, 210), thickness=-1)

    if with_defect:
        cv2.circle(image, center=(110, 80), radius=12, color=(30, 30, 30), thickness=-1)

    return image


def test_infer_label_from_filename() -> None:
    """Labels should be inferred from simple demo filenames."""

    assert infer_label_from_filename("part_clean_001.png") == "clean"
    assert infer_label_from_filename("part_defective_001.png") == "defective"
    assert infer_label_from_filename("part_unknown_001.png") == "unknown"


def test_count_nonzero_pixels_counts_white_pixels() -> None:
    """Non-zero pixel counting should work for binary masks."""

    image = np.zeros((10, 10), dtype=np.uint8)
    image[2:5, 2:5] = 255

    assert count_nonzero_pixels(image) == 9


def test_extract_image_features_returns_expected_basic_fields() -> None:
    """Feature extraction should return a complete feature row."""

    image = create_test_image(with_defect=True)

    features = extract_image_features(
        image,
        filename="part_defective_001.png",
    )

    assert features.filename == "part_defective_001.png"
    assert features.label == "defective"
    assert features.width == 220
    assert features.height == 160
    assert features.image_area == 35200
    assert features.defect_count >= 1
    assert features.total_defect_area > 0


def test_extract_image_features_detects_clean_image() -> None:
    """A clean image should have zero detected defects."""

    image = create_test_image(with_defect=False)

    features = extract_image_features(
        image,
        filename="part_clean_001.png",
    )

    assert features.label == "clean"
    assert features.defect_count == 0
    assert features.total_defect_area == 0.0


def test_feature_rows_to_dataframe_creates_dataframe() -> None:
    """Feature rows should convert into a Pandas DataFrame."""

    image = create_test_image(with_defect=True)
    features = extract_image_features(image, filename="part_defective_001.png")

    dataframe = feature_rows_to_dataframe([features])

    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 1
    assert "filename" in dataframe.columns
    assert "defect_count" in dataframe.columns


def test_extract_features_from_image_file(tmp_path) -> None:
    """Feature extraction should work from an image file path."""

    image = create_test_image(with_defect=True)
    image_path = tmp_path / "part_defective_001.png"
    cv2.imwrite(str(image_path), image)

    features = extract_features_from_image_file(image_path)

    assert features.filename == "part_defective_001.png"
    assert features.label == "defective"
    assert features.defect_count >= 1


def test_extract_feature_dataset_saves_csv(tmp_path) -> None:
    """A feature dataset should be extracted from a folder and saved as CSV."""

    input_dir = tmp_path / "input"
    output_csv = tmp_path / "features.csv"

    create_demo_batch_images(input_dir)
    dataframe = extract_feature_dataset(input_dir, output_csv)

    assert output_csv.exists()
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 4
    assert "label" in dataframe.columns
    assert "defect_area_ratio" in dataframe.columns
