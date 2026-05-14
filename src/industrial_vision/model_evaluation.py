"""Model evaluation and error analysis helpers for industrial vision ML."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from industrial_vision.ml_baseline import (
    FEATURE_COLUMNS,
    MLTrainingResult,
    train_classical_ml_baseline,
)


REQUIRED_PREDICTION_COLUMNS = ["true_label", "predicted_label", "correct"]


def validate_prediction_dataframe(prediction_dataframe: pd.DataFrame) -> None:
    """Validate that a prediction dataframe contains required columns."""

    missing_columns = [
        column for column in REQUIRED_PREDICTION_COLUMNS if column not in prediction_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")


def get_error_rows(prediction_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return only misclassified rows from a prediction dataframe."""

    validate_prediction_dataframe(prediction_dataframe)

    return prediction_dataframe[prediction_dataframe["correct"] == False].copy()


def count_prediction_outcomes(prediction_dataframe: pd.DataFrame) -> dict[str, int]:
    """Count correct and incorrect predictions."""

    validate_prediction_dataframe(prediction_dataframe)

    correct_count = int(prediction_dataframe["correct"].sum())
    total_count = int(len(prediction_dataframe))
    incorrect_count = total_count - correct_count

    return {
        "total": total_count,
        "correct": correct_count,
        "incorrect": incorrect_count,
    }


def build_confusion_matrix_dataframe(
    training_result: MLTrainingResult,
) -> pd.DataFrame:
    """Convert a confusion matrix into a labeled DataFrame."""

    labels = training_result.labels

    return pd.DataFrame(
        training_result.confusion_matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def get_feature_importance_dataframe(
    model: RandomForestClassifier,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a sorted feature-importance table from a trained Random Forest model."""

    columns = feature_columns if feature_columns is not None else FEATURE_COLUMNS

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    if len(model.feature_importances_) != len(columns):
        raise ValueError("Feature importance length does not match feature column length.")

    dataframe = pd.DataFrame(
        {
            "feature": columns,
            "importance": model.feature_importances_,
        }
    )

    return dataframe.sort_values("importance", ascending=False).reset_index(drop=True)


def save_dataframe_csv(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a DataFrame as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)

    return path


def save_confusion_matrix_csv(
    training_result: MLTrainingResult,
    output_path: str | Path,
) -> Path:
    """Save confusion matrix as a CSV file."""

    matrix_dataframe = build_confusion_matrix_dataframe(training_result)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_dataframe.to_csv(path)

    return path


def create_evaluation_report_markdown(
    training_result: MLTrainingResult,
    outcome_counts: dict[str, int],
    error_dataframe: pd.DataFrame,
    feature_importance_dataframe: pd.DataFrame,
) -> str:
    """Create a Markdown evaluation and error-analysis report."""

    top_features = feature_importance_dataframe.head(10)

    top_feature_lines = [
        f"| {row.feature} | {row.importance:.6f} |"
        for row in top_features.itertuples(index=False)
    ]

    if not top_feature_lines:
        top_feature_lines = ["| No features available | 0.000000 |"]

    if error_dataframe.empty:
        error_summary = "No misclassified samples were found in the test set."
    else:
        error_summary = f"{len(error_dataframe)} misclassified sample(s) were found in the test set."

    report = f"""# Step 11 — Model Evaluation and Error Analysis Report

## Summary

| Metric | Value |
|---|---:|
| Accuracy | {training_result.accuracy:.4f} |
| Train rows | {training_result.train_rows} |
| Test rows | {training_result.test_rows} |
| Total predictions | {outcome_counts["total"]} |
| Correct predictions | {outcome_counts["correct"]} |
| Incorrect predictions | {outcome_counts["incorrect"]} |

## Confusion Matrix

Labels:

```text
{training_result.labels}
```

Matrix:

```text
{training_result.confusion_matrix}
```

## Error Analysis

{error_summary}

## Top Feature Importances

| Feature | Importance |
|---|---:|
{chr(10).join(top_feature_lines)}

## Interpretation

This report helps identify whether the baseline model is making reliable predictions and which extracted features influence the model most.

For synthetic data, the model may perform very well. On real industrial images, errors are expected and should be reviewed carefully.

Important questions:

- Are false positives caused by shadows, reflections, or edges?
- Are false negatives caused by weak defects or low contrast?
- Are the most important features physically meaningful?
- Do we need more images, better labels, or better features?
"""

    return report


def save_evaluation_report_markdown(
    report_markdown: str,
    output_path: str | Path,
) -> Path:
    """Save a Markdown evaluation report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_markdown, encoding="utf-8")

    return path


def run_model_evaluation_workflow(
    feature_dataframe: pd.DataFrame,
    output_dir: str | Path,
    random_seed: int = 42,
) -> tuple[MLTrainingResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train baseline model and save evaluation/error-analysis artifacts.

    Returns:
        training_result: model metrics
        prediction_dataframe: test-set predictions
        error_dataframe: misclassified rows
        feature_importance_dataframe: sorted feature importances
    """

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    model, training_result, prediction_dataframe = train_classical_ml_baseline(
        feature_dataframe,
        random_seed=random_seed,
    )

    error_dataframe = get_error_rows(prediction_dataframe)
    outcome_counts = count_prediction_outcomes(prediction_dataframe)
    feature_importance_dataframe = get_feature_importance_dataframe(
        model,
        training_result.feature_columns,
    )

    save_confusion_matrix_csv(
        training_result,
        output_directory / "step11_confusion_matrix.csv",
    )
    save_dataframe_csv(
        error_dataframe,
        output_directory / "step11_misclassified_samples.csv",
    )
    save_dataframe_csv(
        feature_importance_dataframe,
        output_directory / "step11_feature_importance.csv",
    )

    report_markdown = create_evaluation_report_markdown(
        training_result=training_result,
        outcome_counts=outcome_counts,
        error_dataframe=error_dataframe,
        feature_importance_dataframe=feature_importance_dataframe,
    )
    save_evaluation_report_markdown(
        report_markdown,
        output_directory / "step11_evaluation_report.md",
    )

    return training_result, prediction_dataframe, error_dataframe, feature_importance_dataframe
