"""Step 09 command script: feature extraction from images."""

from industrial_vision.batch_processing import create_demo_batch_images
from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.feature_extraction import extract_feature_dataset


def main() -> None:
    """Create demo batch images and extract a feature dataset."""

    ensure_project_directories()

    batch_input_dir = SAMPLE_IMAGES_DIR / "step08_batch_images"
    feature_output_path = PREDICTIONS_OUTPUT_DIR / "step09_features.csv"

    create_demo_batch_images(batch_input_dir)
    dataframe = extract_feature_dataset(batch_input_dir, feature_output_path)

    print("Step 09 — Feature Extraction from Images")
    print(f"Input directory: {batch_input_dir}")
    print(f"Feature CSV: {feature_output_path}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print("Columns:")
    for column in dataframe.columns:
        print(f"  - {column}")

    print("\nPreview:")
    print(dataframe.head().to_string(index=False))


if __name__ == "__main__":
    main()
