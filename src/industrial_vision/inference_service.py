"""Inference helpers for the FastAPI industrial vision service."""

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from industrial_vision.config import PREDICTIONS_OUTPUT_DIR
from industrial_vision.feature_extraction import extract_image_features
from industrial_vision.ml_baseline import FEATURE_COLUMNS
from industrial_vision.model_persistence import load_model_bundle, validate_feature_dataframe


DEFAULT_MODEL_PATH = (
    PREDICTIONS_OUTPUT_DIR
    / "step12_model_persistence"
    / "step12_random_forest_model.joblib"
)


def get_model_path_from_environment() -> Path:
    """Return the model path from environment variable or the default project path."""

    configured_path = os.getenv("INDUSTRIAL_VISION_MODEL_PATH")

    if configured_path:
        return Path(configured_path)

    return DEFAULT_MODEL_PATH


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR OpenCV image."""

    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Uploaded file could not be decoded as an image.")

    return image


def image_to_feature_dataframe(image: np.ndarray, filename: str) -> pd.DataFrame:
    """Convert one image into a one-row feature dataframe."""

    feature_row = extract_image_features(
        image,
        filename=filename,
        label=None,
    )

    return pd.DataFrame([feature_row.to_dict()])


def get_probability_dictionary(model: Any, probabilities: np.ndarray) -> dict[str, float]:
    """Convert model probabilities into a readable dictionary."""

    classes = list(model.classes_)

    return {
        str(class_name): float(probabilities[0][class_index])
        for class_index, class_name in enumerate(classes)
    }


def predict_image_with_model_bundle(
    model_bundle: dict[str, Any],
    image: np.ndarray,
    filename: str,
) -> dict[str, Any]:
    """Extract features from one image and predict with a loaded model bundle."""

    if "model" not in model_bundle or "metadata" not in model_bundle:
        raise ValueError("Model bundle must contain 'model' and 'metadata'.")

    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)

    feature_dataframe = image_to_feature_dataframe(image, filename)
    validate_feature_dataframe(feature_dataframe, feature_columns)

    x = feature_dataframe[feature_columns]
    predicted_label = str(model.predict(x)[0])

    probabilities: dict[str, float] = {}

    if hasattr(model, "predict_proba"):
        probability_array = model.predict_proba(x)
        probabilities = get_probability_dictionary(model, probability_array)

    feature_row = feature_dataframe.iloc[0].to_dict()

    feature_summary = {
        "defect_count": int(feature_row["defect_count"]),
        "total_defect_area": float(feature_row["total_defect_area"]),
        "max_defect_area": float(feature_row["max_defect_area"]),
        "defect_area_ratio": float(feature_row["defect_area_ratio"]),
        "mean_intensity": float(feature_row["mean_intensity"]),
        "edge_density": float(feature_row["edge_density"]),
    }

    return {
        "filename": filename,
        "predicted_label": predicted_label,
        "probabilities": probabilities,
        "feature_summary": feature_summary,
    }


def load_inference_model(model_path: str | Path | None = None) -> dict[str, Any]:
    """Load the inference model bundle from disk."""

    path = Path(model_path) if model_path is not None else get_model_path_from_environment()

    return load_model_bundle(path)
