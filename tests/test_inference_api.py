"""Tests for Step 13 FastAPI inference service."""

import cv2
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from industrial_vision.inference_api import create_app
from industrial_vision.inference_service import (
    decode_image_bytes,
    image_to_feature_dataframe,
    predict_image_with_model_bundle,
)
from industrial_vision.ml_baseline import train_classical_ml_baseline
from industrial_vision.model_persistence import create_model_bundle, save_model_bundle


def create_feature_dataframe() -> pd.DataFrame:
    """Create a small separable feature dataframe for inference API tests."""

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


def create_test_image(with_defect: bool = True) -> np.ndarray:
    """Create a simple BGR test image."""

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (140, 80), (500, 280), color=(210, 210, 210), thickness=-1)
    cv2.rectangle(image, (140, 80), (500, 280), color=(255, 255, 255), thickness=3)

    if with_defect:
        cv2.circle(image, center=(330, 175), radius=28, color=(35, 35, 35), thickness=-1)

    return image


def encode_png(image: np.ndarray) -> bytes:
    """Encode a BGR image as PNG bytes."""

    success, encoded = cv2.imencode(".png", image)

    assert success

    return encoded.tobytes()


def create_saved_model(tmp_path):
    """Train and save a model bundle for API tests."""

    dataframe = create_feature_dataframe()
    model, training_result, _ = train_classical_ml_baseline(dataframe)
    model_path = tmp_path / "model.joblib"
    save_model_bundle(model, training_result, model_path)

    return model_path, create_model_bundle(model, training_result)


def test_decode_image_bytes_decodes_png() -> None:
    """Image bytes should decode into a BGR image."""

    image = create_test_image(with_defect=True)
    image_bytes = encode_png(image)

    decoded = decode_image_bytes(image_bytes)

    assert decoded.shape == image.shape


def test_image_to_feature_dataframe_returns_one_row() -> None:
    """A single image should become a one-row feature dataframe."""

    image = create_test_image(with_defect=True)

    dataframe = image_to_feature_dataframe(image, "part_defective_001.png")

    assert len(dataframe) == 1
    assert "defect_count" in dataframe.columns
    assert "total_defect_area" in dataframe.columns


def test_predict_image_with_model_bundle_returns_prediction(tmp_path) -> None:
    """Prediction helper should return label, probabilities, and feature summary."""

    _, bundle = create_saved_model(tmp_path)
    image = create_test_image(with_defect=True)

    prediction = predict_image_with_model_bundle(
        bundle,
        image,
        "part_defective_001.png",
    )

    assert "predicted_label" in prediction
    assert "probabilities" in prediction
    assert "feature_summary" in prediction
    assert "defect_count" in prediction["feature_summary"]


def test_health_endpoint_reports_model_available(tmp_path) -> None:
    """Health endpoint should report when the model file exists."""

    model_path, _ = create_saved_model(tmp_path)
    app = create_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_available"] is True


def test_health_endpoint_reports_model_missing(tmp_path) -> None:
    """Health endpoint should report when the model file is missing."""

    model_path = tmp_path / "missing.joblib"
    app = create_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_available"] is False


def test_predict_endpoint_returns_prediction(tmp_path) -> None:
    """Predict endpoint should classify an uploaded image."""

    model_path, _ = create_saved_model(tmp_path)
    app = create_app(model_path=model_path)
    client = TestClient(app)

    image = create_test_image(with_defect=True)
    image_bytes = encode_png(image)

    response = client.post(
        "/predict",
        files={"file": ("part_defective_001.png", image_bytes, "image/png")},
    )

    data = response.json()

    assert response.status_code == 200
    assert data["filename"] == "part_defective_001.png"
    assert "predicted_label" in data
    assert "probabilities" in data
    assert "feature_summary" in data


def test_predict_endpoint_rejects_invalid_image(tmp_path) -> None:
    """Predict endpoint should reject invalid image bytes."""

    model_path, _ = create_saved_model(tmp_path)
    app = create_app(model_path=model_path)
    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("not_image.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_predict_endpoint_returns_503_when_model_missing(tmp_path) -> None:
    """Predict endpoint should return service-unavailable when model file is missing."""

    model_path = tmp_path / "missing.joblib"
    app = create_app(model_path=model_path)
    client = TestClient(app)

    image = create_test_image(with_defect=True)
    image_bytes = encode_png(image)

    response = client.post(
        "/predict",
        files={"file": ("part_defective_001.png", image_bytes, "image/png")},
    )

    assert response.status_code == 503
