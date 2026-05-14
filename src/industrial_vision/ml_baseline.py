"""Classical machine learning baseline for industrial vision inspection."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from industrial_vision.feature_extraction import extract_feature_dataset
from industrial_vision.image_io import save_image


FEATURE_COLUMNS = [
    "width",
    "height",
    "image_area",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "edge_pixel_count",
    "edge_density",
    "contour_count",
    "product_area_pixels",
    "defect_count",
    "total_defect_area",
    "max_defect_area",
    "defect_area_ratio",
    "largest_defect_width",
    "largest_defect_height",
    "largest_defect_mean_intensity",
]


@dataclass(frozen=True)
class MLTrainingResult:
    """Artifacts and metrics produced by classical ML training."""

    accuracy: float
    train_rows: int
    test_rows: int
    feature_columns: list[str]
    labels: list[str]
    confusion_matrix: list[list[int]]
    classification_report: dict

    def to_dict(self) -> dict:
        """Convert result to a JSON-friendly dictionary."""

        return asdict(self)


def create_synthetic_ml_image(
    output_path: str | Path,
    label: str,
    rng: np.random.Generator,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Create one synthetic image for classical ML baseline training.

    The image simulates a bright product on a dark background.
    Defective samples contain one or two dark defect regions inside the product.
    Clean samples contain no dark defect region.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.zeros((height, width, 3), dtype=np.uint8)

    background_level = int(rng.integers(0, 15))
    image[:] = (background_level, background_level, background_level)

    product_intensity = int(rng.integers(185, 230))
    x1 = int(rng.integers(120, 155))
    y1 = int(rng.integers(65, 95))
    x2 = int(rng.integers(485, 525))
    y2 = int(rng.integers(260, 295))

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color=(product_intensity, product_intensity, product_intensity),
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color=(255, 255, 255),
        thickness=3,
    )

    if label == "defective":
        defect_count = int(rng.integers(1, 3))

        for _ in range(defect_count):
            center_x = int(rng.integers(x1 + 50, x2 - 50))
            center_y = int(rng.integers(y1 + 40, y2 - 40))
            radius = int(rng.integers(12, 34))
            defect_intensity = int(rng.integers(20, 70))

            cv2.circle(
                image,
                center=(center_x, center_y),
                radius=radius,
                color=(defect_intensity, defect_intensity, defect_intensity),
                thickness=-1,
            )

    label_text = "DEFECTIVE" if label == "defective" else "CLEAN"

    cv2.putText(
        image,
        label_text,
        (230, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (160, 160, 160),
        2,
        cv2.LINE_AA,
    )

    return save_image(image, path)


def create_synthetic_ml_dataset(
    output_dir: str | Path,
    clean_count: int = 30,
    defective_count: int = 30,
    random_seed: int = 42,
) -> list[Path]:
    """Create a deterministic synthetic image dataset for ML baseline training."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    created_paths: list[Path] = []

    for index in range(clean_count):
        image_path = directory / f"part_clean_{index:03d}.png"
        created_paths.append(
            create_synthetic_ml_image(
                image_path,
                label="clean",
                rng=rng,
            )
        )

    for index in range(defective_count):
        image_path = directory / f"part_defective_{index:03d}.png"
        created_paths.append(
            create_synthetic_ml_image(
                image_path,
                label="defective",
                rng=rng,
            )
        )

    return created_paths


def prepare_ml_dataframe(feature_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target vector y."""

    missing_columns = [column for column in FEATURE_COLUMNS if column not in feature_dataframe.columns]

    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    if "label" not in feature_dataframe.columns:
        raise ValueError("Feature dataframe must include a 'label' column.")

    filtered_dataframe = feature_dataframe[feature_dataframe["label"].isin(["clean", "defective"])].copy()

    if filtered_dataframe.empty:
        raise ValueError("No clean/defective rows available for training.")

    x = filtered_dataframe[FEATURE_COLUMNS]
    y = filtered_dataframe["label"]

    return x, y


def train_classical_ml_baseline(
    feature_dataframe: pd.DataFrame,
    random_seed: int = 42,
    test_size: float = 0.25,
) -> tuple[RandomForestClassifier, MLTrainingResult, pd.DataFrame]:
    """Train and evaluate a Random Forest baseline classifier."""

    x, y = prepare_ml_dataframe(feature_dataframe)

    if y.nunique() < 2:
        raise ValueError("Training requires at least two labels: clean and defective.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_seed,
        max_depth=5,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    labels = ["clean", "defective"]
    accuracy = float(accuracy_score(y_test, predictions))
    matrix = confusion_matrix(y_test, predictions, labels=labels).tolist()
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    prediction_dataframe = x_test.copy()
    prediction_dataframe["true_label"] = y_test.values
    prediction_dataframe["predicted_label"] = predictions
    prediction_dataframe["correct"] = prediction_dataframe["true_label"] == prediction_dataframe["predicted_label"]

    result = MLTrainingResult(
        accuracy=accuracy,
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_columns=FEATURE_COLUMNS,
        labels=labels,
        confusion_matrix=matrix,
        classification_report=report,
    )

    return model, result, prediction_dataframe


def save_ml_metrics(
    training_result: MLTrainingResult,
    output_path: str | Path,
) -> Path:
    """Save ML training metrics as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(training_result.to_dict(), file, indent=2)

    return path


def save_predictions_csv(
    prediction_dataframe: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Save ML test-set predictions as CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_dataframe.to_csv(path, index=False)

    return path


def run_classical_ml_baseline_workflow(
    image_dir: str | Path,
    output_dir: str | Path,
    random_seed: int = 42,
) -> tuple[MLTrainingResult, pd.DataFrame, pd.DataFrame]:
    """Run the full Step 10 workflow.

    Workflow:
        images → feature CSV → train classifier → metrics JSON → predictions CSV
    """

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    feature_csv_path = output_directory / "step10_ml_features.csv"
    metrics_json_path = output_directory / "step10_ml_metrics.json"
    predictions_csv_path = output_directory / "step10_ml_predictions.csv"

    feature_dataframe = extract_feature_dataset(image_dir, feature_csv_path)
    _, training_result, prediction_dataframe = train_classical_ml_baseline(
        feature_dataframe,
        random_seed=random_seed,
    )

    save_ml_metrics(training_result, metrics_json_path)
    save_predictions_csv(prediction_dataframe, predictions_csv_path)

    return training_result, feature_dataframe, prediction_dataframe
