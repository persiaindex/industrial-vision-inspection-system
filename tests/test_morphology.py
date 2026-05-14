"""Tests for Step 05 morphological operations."""

import numpy as np
import pytest

from industrial_vision.morphology import (
    apply_closing,
    apply_opening,
    clean_binary_mask,
    create_morphology_kernel,
    dilate_image,
    erode_image,
)


def test_create_morphology_kernel_has_expected_shape() -> None:
    """The morphology kernel should have the requested shape."""

    kernel = create_morphology_kernel((5, 3))

    assert kernel.shape == (3, 5)
    assert kernel.dtype == np.uint8


def test_erode_image_reduces_white_region() -> None:
    """Erosion should reduce the number of white pixels."""

    image = np.zeros((50, 50), dtype=np.uint8)
    image[15:35, 15:35] = 255

    eroded = erode_image(image, kernel_size=(3, 3))

    assert eroded.shape == image.shape
    assert eroded.sum() < image.sum()


def test_dilate_image_expands_white_region() -> None:
    """Dilation should increase the number of white pixels."""

    image = np.zeros((50, 50), dtype=np.uint8)
    image[20:30, 20:30] = 255

    dilated = dilate_image(image, kernel_size=(3, 3))

    assert dilated.shape == image.shape
    assert dilated.sum() > image.sum()


def test_apply_opening_removes_small_white_noise() -> None:
    """Opening should remove tiny white noise pixels."""

    image = np.zeros((50, 50), dtype=np.uint8)
    image[20:30, 20:30] = 255
    image[5, 5] = 255

    opened = apply_opening(image, kernel_size=(3, 3))

    assert opened[5, 5] == 0
    assert opened.shape == image.shape


def test_apply_closing_fills_small_black_hole() -> None:
    """Closing should fill a small black hole inside a white region."""

    image = np.ones((50, 50), dtype=np.uint8) * 255
    image[25, 25] = 0

    closed = apply_closing(image, kernel_size=(3, 3))

    assert closed[25, 25] == 255
    assert closed.shape == image.shape


def test_clean_binary_mask_preserves_shape() -> None:
    """The combined clean mask helper should preserve image shape and dtype."""

    image = np.zeros((60, 80), dtype=np.uint8)
    image[20:40, 25:55] = 255
    image[3, 3] = 255

    cleaned = clean_binary_mask(image)

    assert cleaned.shape == image.shape
    assert cleaned.dtype == image.dtype


def test_morphology_rejects_color_image() -> None:
    """Morphological helpers in this project should reject BGR color images."""

    image = np.zeros((50, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        erode_image(image)
