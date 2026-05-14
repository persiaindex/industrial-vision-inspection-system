"""Model persistence helpers for industrial vision ML."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from industrial_vision.ml_baseline import (
    FEATURE_COLUMNS,
    MLTrainingResult,
    train_classical_ml_baseline,
)


@dataclass(frozen=True)
class SavedModelMetadata:
    """Metadata stored together with a trained model."""

    model_type: str
    feature_columns: list[str]
    labels: list[str]
    accuracy: float
    train_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, object]:
        """Convert metadata into a dictionary."""

        return asdict(self)


def create_model_metadata(training_result: MLTrainingResult) -> SavedModelMetadata:
    """Create model metadata from a training result."""

    return SavedModelMetadata(
        model_type="RandomForestClassifier",
        feature_columns=training_result.feature_columns,
        labels=training_result.labels,
        accuracy=training_result.accuracy,
        train_rows=training_result.train_rows,
        test_rows=training_result.test_rows,
    )


def create_model_bundle(
    model: RandomForestClassifier,
    training_result: MLTrainingResult,
) -> dict[str, Any]:
    """Create a serializable model bundle."""

    metadata = create_model_metadata(training_result)

    return {
        "model": model,
        "metadata": metadata.to_dict(),
    }


def save_model_bundle(
    model: RandomForestClassifier,
    training_result: MLTrainingResult,
    output_path: str | Path,
) -> Path:
    """Save a trained model and metadata as a joblib file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = create_model_bundle(model, training_result)
    joblib.dump(bundle, path)

    return path


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load a trained model bundle from disk."""

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError("Loaded model bundle must be a dictionary.")

    if "model" not in bundle or "metadata" not in bundle:
        raise ValueError("Model bundle must contain 'model' and 'metadata'.")

    return bundle


def save_model_metadata_json(
    training_result: MLTrainingResult,
    output_path: str | Path,
) -> Path:
    """Save model metadata as a readable JSON file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = create_model_metadata(training_result)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata.to_dict(), file, indent=2)

    return path


def validate_feature_dataframe(
    feature_dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Validate that a dataframe contains the required model feature columns."""

    missing_columns = [
        column for column in feature_columns if column not in feature_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns for prediction: {missing_columns}")


def predict_with_model_bundle(
    model_bundle: dict[str, Any],
    feature_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Run predictions using a loaded model bundle."""

    if "model" not in model_bundle or "metadata" not in model_bundle:
        raise ValueError("Model bundle must contain 'model' and 'metadata'.")

    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)

    validate_feature_dataframe(feature_dataframe, feature_columns)

    x = feature_dataframe[feature_columns]
    predictions = model.predict(x)

    output = pd.DataFrame()

    if "filename" in feature_dataframe.columns:
        output["filename"] = feature_dataframe["filename"].values

    if "label" in feature_dataframe.columns:
        output["true_label"] = feature_dataframe["label"].values

    output["predicted_label"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(model.classes_)

        for class_index, class_name in enumerate(classes):
            output[f"probability_{class_name}"] = probabilities[:, class_index]

    if "true_label" in output.columns:
        output["correct"] = output["true_label"] == output["predicted_label"]

    return output


def save_loaded_model_predictions_csv(
    prediction_dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save predictions from a loaded model as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_dataframe.to_csv(path, index=False)

    return path


def train_save_load_predict_workflow(
    feature_dataframe: pd.DataFrame,
    output_dir: str | Path,
    random_seed: int = 42,
) -> tuple[Path, Path, pd.DataFrame, SavedModelMetadata]:
    """Train, save, load, and predict with a classical ML model.

    Returns:
        model_path: saved joblib model path
        metadata_path: saved readable metadata JSON path
        prediction_dataframe: predictions created by the loaded model
        metadata: saved model metadata object
    """

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    model, training_result, _ = train_classical_ml_baseline(
        feature_dataframe,
        random_seed=random_seed,
    )

    model_path = output_directory / "step12_random_forest_model.joblib"
    metadata_path = output_directory / "step12_model_metadata.json"
    predictions_path = output_directory / "step12_loaded_model_predictions.csv"

    save_model_bundle(model, training_result, model_path)
    save_model_metadata_json(training_result, metadata_path)

    loaded_bundle = load_model_bundle(model_path)
    prediction_dataframe = predict_with_model_bundle(
        loaded_bundle,
        feature_dataframe,
    )
    save_loaded_model_predictions_csv(prediction_dataframe, predictions_path)

    metadata = create_model_metadata(training_result)

    return model_path, metadata_path, prediction_dataframe, metadata
