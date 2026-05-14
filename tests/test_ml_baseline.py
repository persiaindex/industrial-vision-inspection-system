"""Tests for Step 10 classical ML baseline."""

import json

import pandas as pd
import pytest

from industrial_vision.ml_baseline import (
    FEATURE_COLUMNS,
    create_synthetic_ml_dataset,
    prepare_ml_dataframe,
    run_classical_ml_baseline_workflow,
    save_ml_metrics,
    save_predictions_csv,
    train_classical_ml_baseline,
)


def create_small_feature_dataframe() -> pd.DataFrame:
    """Create a small feature dataframe for baseline tests."""

    rows = []

    for index in range(8):
        rows.append(
            {
                "filename": f"part_clean_{index:03d}.png",
                "label": "clean",
                "width": 640,
                "height": 360,
                "image_area": 230400,
                "mean_intensity": 70.0,
                "std_intensity": 90.0,
                "min_intensity": 0,
                "max_intensity": 255,
                "edge_pixel_count": 1200,
                "edge_density": 0.005,
                "contour_count": 2,
                "product_area_pixels": 72000,
                "defect_count": 0,
                "total_defect_area": 0.0,
                "max_defect_area": 0.0,
                "defect_area_ratio": 0.0,
                "largest_defect_width": 0,
                "largest_defect_height": 0,
                "largest_defect_mean_intensity": 0.0,
            }
        )

    for index in range(8):
        rows.append(
            {
                "filename": f"part_defective_{index:03d}.png",
                "label": "defective",
                "width": 640,
                "height": 360,
                "image_area": 230400,
                "mean_intensity": 68.0,
                "std_intensity": 92.0,
                "min_intensity": 0,
                "max_intensity": 255,
                "edge_pixel_count": 1500,
                "edge_density": 0.007,
                "contour_count": 4,
                "product_area_pixels": 72000,
                "defect_count": 1,
                "total_defect_area": 900.0,
                "max_defect_area": 900.0,
                "defect_area_ratio": 0.0125,
                "largest_defect_width": 30,
                "largest_defect_height": 30,
                "largest_defect_mean_intensity": 40.0,
            }
        )

    return pd.DataFrame(rows)


def test_create_synthetic_ml_dataset_creates_requested_images(tmp_path) -> None:
    """Synthetic ML dataset generation should create clean and defective images."""

    created_paths = create_synthetic_ml_dataset(
        tmp_path,
        clean_count=3,
        defective_count=2,
        random_seed=123,
    )

    assert len(created_paths) == 5
    assert all(path.exists() for path in created_paths)
    assert any("clean" in path.name for path in created_paths)
    assert any("defective" in path.name for path in created_paths)


def test_prepare_ml_dataframe_returns_x_and_y() -> None:
    """The ML dataframe preparation helper should return feature matrix and labels."""

    dataframe = create_small_feature_dataframe()

    x, y = prepare_ml_dataframe(dataframe)

    assert list(x.columns) == FEATURE_COLUMNS
    assert len(x) == len(y)
    assert set(y.unique()) == {"clean", "defective"}


def test_prepare_ml_dataframe_rejects_missing_label() -> None:
    """A dataframe without a label column should be rejected."""

    dataframe = create_small_feature_dataframe().drop(columns=["label"])

    with pytest.raises(ValueError):
        prepare_ml_dataframe(dataframe)


def test_train_classical_ml_baseline_returns_metrics_and_predictions() -> None:
    """Training should return a model, metrics, and prediction rows."""

    dataframe = create_small_feature_dataframe()

    model, result, predictions = train_classical_ml_baseline(
        dataframe,
        random_seed=42,
        test_size=0.25,
    )

    assert hasattr(model, "predict")
    assert result.train_rows > 0
    assert result.test_rows > 0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.labels == ["clean", "defective"]
    assert "true_label" in predictions.columns
    assert "predicted_label" in predictions.columns


def test_save_ml_metrics_writes_json(tmp_path) -> None:
    """ML metrics should be saved as JSON."""

    dataframe = create_small_feature_dataframe()
    _, result, _ = train_classical_ml_baseline(dataframe)
    output_path = tmp_path / "metrics.json"

    saved_path = save_ml_metrics(result, output_path)

    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert "accuracy" in data
    assert "confusion_matrix" in data


def test_save_predictions_csv_writes_csv(tmp_path) -> None:
    """Prediction rows should be saved as CSV."""

    dataframe = create_small_feature_dataframe()
    _, _, predictions = train_classical_ml_baseline(dataframe)
    output_path = tmp_path / "predictions.csv"

    saved_path = save_predictions_csv(predictions, output_path)

    loaded = pd.read_csv(saved_path)

    assert saved_path.exists()
    assert len(loaded) == len(predictions)
    assert "predicted_label" in loaded.columns


def test_run_classical_ml_baseline_workflow_creates_outputs(tmp_path) -> None:
    """The full workflow should create feature, metric, and prediction outputs."""

    image_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"

    create_synthetic_ml_dataset(
        image_dir,
        clean_count=6,
        defective_count=6,
        random_seed=42,
    )

    result, feature_dataframe, prediction_dataframe = run_classical_ml_baseline_workflow(
        image_dir,
        output_dir,
        random_seed=42,
    )

    assert result.train_rows > 0
    assert result.test_rows > 0
    assert len(feature_dataframe) == 12
    assert len(prediction_dataframe) == result.test_rows
    assert (output_dir / "step10_ml_features.csv").exists()
    assert (output_dir / "step10_ml_metrics.json").exists()
    assert (output_dir / "step10_ml_predictions.csv").exists()
