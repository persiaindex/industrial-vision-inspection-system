"""Tests for Step 03 preprocessing helpers."""

import numpy as np
import pytest

from industrial_vision.preprocessing import (
    apply_binary_threshold,
    apply_gaussian_blur,
    convert_to_grayscale,
    preprocess_basic,
)


def test_convert_to_grayscale_from_bgr_image() -> None:
    """A BGR image should be converted into a 2D grayscale image."""

    image = np.zeros((120, 200, 3), dtype=np.uint8)

    grayscale = convert_to_grayscale(image)

    assert grayscale.shape == (120, 200)
    assert grayscale.dtype == np.uint8


def test_convert_to_grayscale_keeps_grayscale_shape() -> None:
    """A grayscale input should remain a 2D grayscale image."""

    image = np.zeros((80, 160), dtype=np.uint8)

    grayscale = convert_to_grayscale(image)

    assert grayscale.shape == (80, 160)
    assert grayscale.dtype == np.uint8


def test_apply_gaussian_blur_preserves_shape() -> None:
    """Gaussian blur should preserve the image shape."""

    image = np.zeros((100, 150), dtype=np.uint8)

    blurred = apply_gaussian_blur(image, kernel_size=(5, 5))

    assert blurred.shape == image.shape
    assert blurred.dtype == image.dtype


def test_apply_gaussian_blur_rejects_even_kernel_size() -> None:
    """Gaussian blur kernel size must use odd numbers."""

    image = np.zeros((100, 150), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_gaussian_blur(image, kernel_size=(4, 4))


def test_apply_binary_threshold_returns_binary_image() -> None:
    """Binary thresholding should return only 0 and 255 values."""

    image = np.array(
        [
            [0, 50, 120],
            [180, 220, 255],
        ],
        dtype=np.uint8,
    )

    thresholded = apply_binary_threshold(image, threshold_value=100)

    unique_values = set(np.unique(thresholded).tolist())

    assert unique_values == {0, 255}


def test_preprocess_basic_returns_three_images() -> None:
    """The basic preprocessing pipeline should return grayscale, blurred, and thresholded images."""

    image = np.zeros((120, 200, 3), dtype=np.uint8)

    grayscale, blurred, thresholded = preprocess_basic(image)

    assert grayscale.shape == (120, 200)
    assert blurred.shape == (120, 200)
    assert thresholded.shape == (120, 200)
