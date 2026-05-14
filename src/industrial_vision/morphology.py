"""Morphological image operations for industrial vision inspection."""

from typing import Tuple

import cv2
import numpy as np


def validate_single_channel_image(image: np.ndarray) -> None:
    """Validate that an image is single-channel.

    Morphological operations in this project are applied to grayscale
    or binary images, not directly to BGR color images.
    """

    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel image, got shape: {image.shape}")


def create_morphology_kernel(kernel_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """Create a rectangular morphology kernel."""

    kernel_width, kernel_height = kernel_size

    if kernel_width <= 0 or kernel_height <= 0:
        raise ValueError("Morphology kernel values must be positive.")

    return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)


def erode_image(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Apply erosion.

    Erosion shrinks white regions and can remove small white noise.
    """

    validate_single_channel_image(image)

    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    kernel = create_morphology_kernel(kernel_size)

    return cv2.erode(image, kernel, iterations=iterations)


def dilate_image(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    """Apply dilation.

    Dilation expands white regions and can connect nearby white areas.
    """

    validate_single_channel_image(image)

    if iterations <= 0:
        raise ValueError("iterations must be positive.")

    kernel = create_morphology_kernel(kernel_size)

    return cv2.dilate(image, kernel, iterations=iterations)


def apply_opening(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (3, 3),
) -> np.ndarray:
    """Apply opening.

    Opening is erosion followed by dilation.
    It is useful for removing small white noise.
    """

    validate_single_channel_image(image)

    kernel = create_morphology_kernel(kernel_size)

    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def apply_closing(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Apply closing.

    Closing is dilation followed by erosion.
    It is useful for closing small black holes inside white regions.
    """

    validate_single_channel_image(image)

    kernel = create_morphology_kernel(kernel_size)

    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def clean_binary_mask(
    binary_image: np.ndarray,
    opening_kernel_size: Tuple[int, int] = (3, 3),
    closing_kernel_size: Tuple[int, int] = (5, 5),
) -> np.ndarray:
    """Clean a binary mask using opening followed by closing."""

    opened = apply_opening(binary_image, kernel_size=opening_kernel_size)
    closed = apply_closing(opened, kernel_size=closing_kernel_size)

    return closed
