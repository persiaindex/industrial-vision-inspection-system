"""Step 11 command script: model evaluation and error analysis."""

from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.feature_extraction import extract_feature_dataset
from industrial_vision.ml_baseline import create_synthetic_ml_dataset
from industrial_vision.model_evaluation import (
    count_prediction_outcomes,
    run_model_evaluation_workflow,
)


def main() -> None:
    """Create data, extract features, train model, and save evaluation outputs."""

    ensure_project_directories()

    dataset_dir = SAMPLE_IMAGES_DIR / "step10_ml_dataset"
    output_dir = PREDICTIONS_OUTPUT_DIR / "step11_model_evaluation"
    feature_csv_path = output_dir / "step11_features.csv"

    create_synthetic_ml_dataset(
        dataset_dir,
        clean_count=30,
        defective_count=30,
        random_seed=42,
    )

    feature_dataframe = extract_feature_dataset(dataset_dir, feature_csv_path)

    (
        training_result,
        prediction_dataframe,
        error_dataframe,
        feature_importance_dataframe,
    ) = run_model_evaluation_workflow(
        feature_dataframe,
        output_dir,
        random_seed=42,
    )

    outcome_counts = count_prediction_outcomes(prediction_dataframe)

    print("Step 11 — Model Evaluation and Error Analysis")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Feature rows: {len(feature_dataframe)}")
    print(f"Train rows: {training_result.train_rows}")
    print(f"Test rows: {training_result.test_rows}")
    print(f"Accuracy: {training_result.accuracy:.3f}")
    print(f"Correct predictions: {outcome_counts['correct']}")
    print(f"Incorrect predictions: {outcome_counts['incorrect']}")
    print(f"Misclassified samples: {len(error_dataframe)}")

    print("\nTop 5 feature importances:")
    print(feature_importance_dataframe.head(5).to_string(index=False))

    print(f"\nSaved features to: {feature_csv_path}")
    print(f"Saved confusion matrix to: {output_dir / 'step11_confusion_matrix.csv'}")
    print(f"Saved misclassified samples to: {output_dir / 'step11_misclassified_samples.csv'}")
    print(f"Saved feature importance to: {output_dir / 'step11_feature_importance.csv'}")
    print(f"Saved evaluation report to: {output_dir / 'step11_evaluation_report.md'}")


if __name__ == "__main__":
    main()
