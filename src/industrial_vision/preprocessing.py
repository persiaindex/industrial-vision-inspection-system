"""Image preprocessing helpers for the industrial vision project."""

from typing import Tuple

import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR color image to grayscale.

    If the input image is already grayscale, a copy is returned.
    """

    if image.ndim == 2:
        return image.copy()

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Expected a grayscale image or BGR image with 3 channels, got shape: {image.shape}")


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma_x: float = 0,
) -> np.ndarray:
    """Apply Gaussian blur to reduce image noise."""

    kernel_width, kernel_height = kernel_size

    if kernel_width <= 0 or kernel_height <= 0:
        raise ValueError("Gaussian blur kernel values must be positive.")

    if kernel_width % 2 == 0 or kernel_height % 2 == 0:
        raise ValueError("Gaussian blur kernel values must be odd numbers.")

    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def apply_binary_threshold(
    grayscale_image: np.ndarray,
    threshold_value: int = 100,
    max_value: int = 255,
    invert: bool = False,
) -> np.ndarray:
    """Apply binary thresholding to a grayscale image.

    When invert=False:
        pixels above threshold become white, others become black.

    When invert=True:
        pixels below threshold become white, others become black.
    """

    if grayscale_image.ndim != 2:
        raise ValueError("Binary thresholding expects a single-channel grayscale image.")

    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    _, thresholded = cv2.threshold(
        grayscale_image,
        threshold_value,
        max_value,
        threshold_type,
    )

    return thresholded


def preprocess_basic(
    image: np.ndarray,
    blur_kernel_size: Tuple[int, int] = (5, 5),
    threshold_value: int = 100,
    invert_threshold: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the basic Step 03 preprocessing pipeline.

    Pipeline:
        BGR image → grayscale → Gaussian blur → binary threshold
    """

    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale, kernel_size=blur_kernel_size)
    thresholded = apply_binary_threshold(
        blurred,
        threshold_value=threshold_value,
        invert=invert_threshold,
    )

    return grayscale, blurred, thresholded
