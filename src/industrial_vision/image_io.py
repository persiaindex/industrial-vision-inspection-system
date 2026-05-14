"""Image input/output helpers for the industrial vision project."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageInfo:
    """Basic metadata about an image."""

    filename: str
    path: str
    width: int
    height: int
    channels: int
    dtype: str
    min_value: int
    max_value: int
    mean_intensity: float
    file_size_bytes: int


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk using OpenCV.

    OpenCV loads color images in BGR channel order by default.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not read the image file: {path}")

    return image


def inspect_image(image_path: str | Path) -> ImageInfo:
    """Load an image and return useful metadata."""

    path = Path(image_path)
    image = load_image(path)

    height, width, channels = image.shape

    return ImageInfo(
        filename=path.name,
        path=str(path),
        width=width,
        height=height,
        channels=channels,
        dtype=str(image.dtype),
        min_value=int(image.min()),
        max_value=int(image.max()),
        mean_intensity=float(image.mean()),
        file_size_bytes=path.stat().st_size,
    )


def save_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an image to disk and return the output path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise ValueError(f"OpenCV could not save image to: {path}")

    return path


def create_synthetic_sample_image(output_path: str | Path, width: int = 640, height: int = 360) -> Path:
    """Create a simple synthetic industrial sample image.

    The image simulates a bright rectangular product on a dark background,
    with one dark defect-like spot.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Product body
    cv2.rectangle(image, (140, 80), (500, 280), color=(210, 210, 210), thickness=-1)

    # Product border
    cv2.rectangle(image, (140, 80), (500, 280), color=(255, 255, 255), thickness=3)

    # Defect-like dark spot
    cv2.circle(image, center=(330, 175), radius=28, color=(35, 35, 35), thickness=-1)

    # Small label text
    cv2.putText(
        image,
        "SAMPLE PART",
        (235, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )

    return save_image(image, path)
