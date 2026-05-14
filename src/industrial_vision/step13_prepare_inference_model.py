"""Step 13 command script: prepare model for FastAPI inference."""

from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.feature_extraction import extract_feature_dataset
from industrial_vision.ml_baseline import create_synthetic_ml_dataset
from industrial_vision.model_persistence import train_save_load_predict_workflow


def main() -> None:
    """Create dataset, train model, and save model artifact for the API."""

    ensure_project_directories()

    dataset_dir = SAMPLE_IMAGES_DIR / "step10_ml_dataset"
    output_dir = PREDICTIONS_OUTPUT_DIR / "step12_model_persistence"
    feature_csv_path = output_dir / "step13_api_training_features.csv"

    create_synthetic_ml_dataset(
        dataset_dir,
        clean_count=30,
        defective_count=30,
        random_seed=42,
    )

    feature_dataframe = extract_feature_dataset(dataset_dir, feature_csv_path)

    (
        model_path,
        metadata_path,
        prediction_dataframe,
        metadata,
    ) = train_save_load_predict_workflow(
        feature_dataframe,
        output_dir,
        random_seed=42,
    )

    print("Step 13 — Prepare Model for FastAPI Inference")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Feature rows: {len(feature_dataframe)}")
    print(f"Model type: {metadata.model_type}")
    print(f"Accuracy before saving: {metadata.accuracy:.3f}")
    print(f"Prediction rows from loaded model: {len(prediction_dataframe)}")
    print(f"Saved features to: {feature_csv_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
