"""Tests for Step 12 model persistence."""

import json

import pandas as pd
import pytest

from industrial_vision.model_persistence import (
    create_model_bundle,
    create_model_metadata,
    load_model_bundle,
    predict_with_model_bundle,
    save_loaded_model_predictions_csv,
    save_model_bundle,
    save_model_metadata_json,
    train_save_load_predict_workflow,
    validate_feature_dataframe,
)
from industrial_vision.ml_baseline import FEATURE_COLUMNS, train_classical_ml_baseline


def create_feature_dataframe() -> pd.DataFrame:
    """Create a small separable feature dataframe for persistence tests."""

    rows = []

    for index in range(10):
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

    for index in range(10):
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


def test_create_model_metadata_contains_training_information() -> None:
    """Model metadata should contain training and feature information."""

    dataframe = create_feature_dataframe()
    _, training_result, _ = train_classical_ml_baseline(dataframe)

    metadata = create_model_metadata(training_result)

    assert metadata.model_type == "RandomForestClassifier"
    assert metadata.feature_columns == FEATURE_COLUMNS
    assert metadata.train_rows > 0
    assert metadata.test_rows > 0


def test_create_model_bundle_contains_model_and_metadata() -> None:
    """Model bundle should contain the trained model and metadata."""

    dataframe = create_feature_dataframe()
    model, training_result, _ = train_classical_ml_baseline(dataframe)

    bundle = create_model_bundle(model, training_result)

    assert "model" in bundle
    assert "metadata" in bundle
    assert bundle["metadata"]["model_type"] == "RandomForestClassifier"


def test_save_and_load_model_bundle_roundtrip(tmp_path) -> None:
    """A saved model bundle should load back with model and metadata."""

    dataframe = create_feature_dataframe()
    model, training_result, _ = train_classical_ml_baseline(dataframe)
    model_path = tmp_path / "model.joblib"

    save_model_bundle(model, training_result, model_path)
    loaded_bundle = load_model_bundle(model_path)

    assert model_path.exists()
    assert "model" in loaded_bundle
    assert "metadata" in loaded_bundle


def test_load_model_bundle_raises_for_missing_file(tmp_path) -> None:
    """Loading a missing model file should produce a clear error."""

    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError):
        load_model_bundle(missing_path)


def test_save_model_metadata_json_writes_readable_metadata(tmp_path) -> None:
    """Metadata should also be saved as readable JSON."""

    dataframe = create_feature_dataframe()
    _, training_result, _ = train_classical_ml_baseline(dataframe)
    metadata_path = tmp_path / "metadata.json"

    saved_path = save_model_metadata_json(training_result, metadata_path)

    data = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_path.exists()
    assert data["model_type"] == "RandomForestClassifier"
    assert "feature_columns" in data


def test_validate_feature_dataframe_rejects_missing_columns() -> None:
    """Prediction dataframe validation should reject missing feature columns."""

    dataframe = create_feature_dataframe().drop(columns=["defect_count"])

    with pytest.raises(ValueError):
        validate_feature_dataframe(dataframe, FEATURE_COLUMNS)


def test_predict_with_model_bundle_returns_predictions() -> None:
    """Loaded model prediction helper should return predicted labels and probabilities."""

    dataframe = create_feature_dataframe()
    model, training_result, _ = train_classical_ml_baseline(dataframe)
    bundle = create_model_bundle(model, training_result)

    predictions = predict_with_model_bundle(bundle, dataframe)

    assert len(predictions) == len(dataframe)
    assert "predicted_label" in predictions.columns
    assert "probability_clean" in predictions.columns
    assert "probability_defective" in predictions.columns
    assert "correct" in predictions.columns


def test_save_loaded_model_predictions_csv_writes_file(tmp_path) -> None:
    """Loaded-model predictions should be saved as CSV."""

    dataframe = create_feature_dataframe()
    model, training_result, _ = train_classical_ml_baseline(dataframe)
    bundle = create_model_bundle(model, training_result)
    predictions = predict_with_model_bundle(bundle, dataframe)

    output_path = tmp_path / "predictions.csv"
    saved_path = save_loaded_model_predictions_csv(predictions, output_path)

    loaded = pd.read_csv(saved_path)

    assert saved_path.exists()
    assert len(loaded) == len(predictions)
    assert "predicted_label" in loaded.columns


def test_train_save_load_predict_workflow_creates_outputs(tmp_path) -> None:
    """The full persistence workflow should save model, metadata, and predictions."""

    dataframe = create_feature_dataframe()
    output_dir = tmp_path / "model_outputs"

    model_path, metadata_path, predictions, metadata = train_save_load_predict_workflow(
        dataframe,
        output_dir,
        random_seed=42,
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert (output_dir / "step12_loaded_model_predictions.csv").exists()
    assert len(predictions) == len(dataframe)
    assert metadata.model_type == "RandomForestClassifier"
