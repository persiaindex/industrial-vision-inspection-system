"""Tests for Step 11 model evaluation and error analysis."""

import pandas as pd
import pytest

from industrial_vision.ml_baseline import MLTrainingResult, train_classical_ml_baseline
from industrial_vision.model_evaluation import (
    build_confusion_matrix_dataframe,
    count_prediction_outcomes,
    create_evaluation_report_markdown,
    get_error_rows,
    get_feature_importance_dataframe,
    run_model_evaluation_workflow,
    save_confusion_matrix_csv,
    save_dataframe_csv,
    validate_prediction_dataframe,
)


def create_feature_dataframe() -> pd.DataFrame:
    """Create a small but separable feature dataframe for evaluation tests."""

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


def test_validate_prediction_dataframe_rejects_missing_columns() -> None:
    """Prediction dataframe validation should reject missing required columns."""

    dataframe = pd.DataFrame({"true_label": ["clean"]})

    with pytest.raises(ValueError):
        validate_prediction_dataframe(dataframe)


def test_get_error_rows_returns_only_incorrect_predictions() -> None:
    """Error rows should include only incorrect predictions."""

    dataframe = pd.DataFrame(
        {
            "true_label": ["clean", "defective", "clean"],
            "predicted_label": ["clean", "clean", "defective"],
            "correct": [True, False, False],
        }
    )

    errors = get_error_rows(dataframe)

    assert len(errors) == 2
    assert errors["correct"].sum() == 0


def test_count_prediction_outcomes_counts_correct_and_incorrect() -> None:
    """Prediction outcome counts should be calculated correctly."""

    dataframe = pd.DataFrame(
        {
            "true_label": ["clean", "defective", "clean"],
            "predicted_label": ["clean", "clean", "defective"],
            "correct": [True, False, False],
        }
    )

    counts = count_prediction_outcomes(dataframe)

    assert counts["total"] == 3
    assert counts["correct"] == 1
    assert counts["incorrect"] == 2


def test_build_confusion_matrix_dataframe_labels_rows_and_columns() -> None:
    """Confusion matrix dataframe should have readable row and column labels."""

    result = MLTrainingResult(
        accuracy=0.8,
        train_rows=10,
        test_rows=5,
        feature_columns=["a", "b"],
        labels=["clean", "defective"],
        confusion_matrix=[[2, 1], [0, 2]],
        classification_report={},
    )

    matrix_dataframe = build_confusion_matrix_dataframe(result)

    assert list(matrix_dataframe.index) == ["true_clean", "true_defective"]
    assert list(matrix_dataframe.columns) == ["pred_clean", "pred_defective"]
    assert matrix_dataframe.loc["true_clean", "pred_clean"] == 2


def test_get_feature_importance_dataframe_returns_sorted_table() -> None:
    """Feature importance dataframe should be sorted by importance."""

    feature_dataframe = create_feature_dataframe()
    model, result, _ = train_classical_ml_baseline(feature_dataframe)

    importance_dataframe = get_feature_importance_dataframe(
        model,
        result.feature_columns,
    )

    assert len(importance_dataframe) == len(result.feature_columns)
    assert list(importance_dataframe.columns) == ["feature", "importance"]
    assert importance_dataframe.iloc[0]["importance"] >= importance_dataframe.iloc[-1]["importance"]


def test_save_dataframe_csv_writes_file(tmp_path) -> None:
    """Generic dataframe CSV saving should create a file."""

    dataframe = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    output_path = tmp_path / "data.csv"

    saved_path = save_dataframe_csv(dataframe, output_path)

    assert saved_path.exists()


def test_save_confusion_matrix_csv_writes_file(tmp_path) -> None:
    """Confusion matrix CSV saving should create a file."""

    result = MLTrainingResult(
        accuracy=1.0,
        train_rows=10,
        test_rows=4,
        feature_columns=["a", "b"],
        labels=["clean", "defective"],
        confusion_matrix=[[2, 0], [0, 2]],
        classification_report={},
    )
    output_path = tmp_path / "confusion_matrix.csv"

    saved_path = save_confusion_matrix_csv(result, output_path)

    assert saved_path.exists()


def test_create_evaluation_report_markdown_contains_key_sections() -> None:
    """Markdown report should include the main evaluation sections."""

    result = MLTrainingResult(
        accuracy=1.0,
        train_rows=10,
        test_rows=4,
        feature_columns=["a", "b"],
        labels=["clean", "defective"],
        confusion_matrix=[[2, 0], [0, 2]],
        classification_report={},
    )
    outcome_counts = {"total": 4, "correct": 4, "incorrect": 0}
    error_dataframe = pd.DataFrame(columns=["true_label", "predicted_label", "correct"])
    importance_dataframe = pd.DataFrame(
        {"feature": ["defect_count"], "importance": [0.9]}
    )

    report = create_evaluation_report_markdown(
        result,
        outcome_counts,
        error_dataframe,
        importance_dataframe,
    )

    assert "Model Evaluation and Error Analysis Report" in report
    assert "Confusion Matrix" in report
    assert "Top Feature Importances" in report


def test_run_model_evaluation_workflow_creates_outputs(tmp_path) -> None:
    """Full evaluation workflow should save all expected artifacts."""

    feature_dataframe = create_feature_dataframe()
    output_dir = tmp_path / "evaluation"

    result, predictions, errors, importances = run_model_evaluation_workflow(
        feature_dataframe,
        output_dir,
        random_seed=42,
    )

    assert result.test_rows == len(predictions)
    assert len(importances) == len(result.feature_columns)
    assert isinstance(errors, pd.DataFrame)
    assert (output_dir / "step11_confusion_matrix.csv").exists()
    assert (output_dir / "step11_misclassified_samples.csv").exists()
    assert (output_dir / "step11_feature_importance.csv").exists()
    assert (output_dir / "step11_evaluation_report.md").exists()
