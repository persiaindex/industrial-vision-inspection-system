"""Step 10 command script: classical ML baseline."""

from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.ml_baseline import (
    create_synthetic_ml_dataset,
    run_classical_ml_baseline_workflow,
)


def main() -> None:
    """Create a synthetic ML dataset, extract features, train, and evaluate a baseline model."""

    ensure_project_directories()

    dataset_dir = SAMPLE_IMAGES_DIR / "step10_ml_dataset"
    output_dir = PREDICTIONS_OUTPUT_DIR / "step10_ml_baseline"

    created_images = create_synthetic_ml_dataset(
        dataset_dir,
        clean_count=30,
        defective_count=30,
        random_seed=42,
    )

    training_result, feature_dataframe, prediction_dataframe = run_classical_ml_baseline_workflow(
        dataset_dir,
        output_dir,
        random_seed=42,
    )

    print("Step 10 — Classical ML Baseline")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images created: {len(created_images)}")
    print(f"Feature rows: {len(feature_dataframe)}")
    print(f"Train rows: {training_result.train_rows}")
    print(f"Test rows: {training_result.test_rows}")
    print(f"Accuracy: {training_result.accuracy:.3f}")
    print(f"Predictions saved: {len(prediction_dataframe)}")
    print(f"Saved features to: {output_dir / 'step10_ml_features.csv'}")
    print(f"Saved metrics to: {output_dir / 'step10_ml_metrics.json'}")
    print(f"Saved predictions to: {output_dir / 'step10_ml_predictions.csv'}")
    print("\nConfusion matrix rows=true labels, columns=predicted labels")
    print(f"Labels: {training_result.labels}")
    print(training_result.confusion_matrix)


if __name__ == "__main__":
    main()
