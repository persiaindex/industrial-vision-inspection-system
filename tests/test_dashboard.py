"""Tests for Step 14 simple inspection dashboard."""

import cv2
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from industrial_vision.dashboard import create_dashboard_app
from industrial_vision.ml_baseline import train_classical_ml_baseline
from industrial_vision.model_persistence import save_model_bundle


def create_feature_dataframe() -> pd.DataFrame:
    """Create a small separable feature dataframe for dashboard tests."""

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
    """Train and save a model bundle for dashboard tests."""

    dataframe = create_feature_dataframe()
    model, training_result, _ = train_classical_ml_baseline(dataframe)
    model_path = tmp_path / "model.joblib"
    save_model_bundle(model, training_result, model_path)

    return model_path


def test_dashboard_home_page_returns_html(tmp_path) -> None:
    """Dashboard home page should return the HTML interface."""

    model_path = create_saved_model(tmp_path)
    app = create_dashboard_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Industrial Vision Dashboard" in response.text
    assert "predictionForm" in response.text


def test_dashboard_css_is_available(tmp_path) -> None:
    """Dashboard CSS route should be available."""

    model_path = create_saved_model(tmp_path)
    app = create_dashboard_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/dashboard.css")

    assert response.status_code == 200
    assert ".hero" in response.text
    assert ".card" in response.text


def test_dashboard_javascript_is_available(tmp_path) -> None:
    """Dashboard JavaScript route should be available."""

    model_path = create_saved_model(tmp_path)
    app = create_dashboard_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/dashboard.js")

    assert response.status_code == 200
    assert "/predict" in response.text
    assert "/health" in response.text


def test_dashboard_health_endpoint_still_works(tmp_path) -> None:
    """The dashboard app should still expose the API health endpoint."""

    model_path = create_saved_model(tmp_path)
    app = create_dashboard_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_available"] is True


def test_dashboard_predict_endpoint_still_works(tmp_path) -> None:
    """The dashboard app should still expose the prediction endpoint."""

    model_path = create_saved_model(tmp_path)
    app = create_dashboard_app(model_path=model_path)
    client = TestClient(app)

    image = create_test_image(with_defect=True)
    image_bytes = encode_png(image)

    response = client.post(
        "/predict",
        files={"file": ("part_defective_001.png", image_bytes, "image/png")},
    )

    data = response.json()

    assert response.status_code == 200
    assert "predicted_label" in data
    assert "probabilities" in data
    assert "feature_summary" in data
