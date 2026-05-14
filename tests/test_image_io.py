"""Tests for image loading, inspection, saving, and sample generation."""

import numpy as np

from industrial_vision.image_io import create_synthetic_sample_image, inspect_image, load_image, save_image


def test_create_and_load_synthetic_sample_image(tmp_path) -> None:
    """A generated sample image should be readable by OpenCV."""

    image_path = tmp_path / "sample.png"

    create_synthetic_sample_image(image_path, width=320, height=180)
    image = load_image(image_path)

    assert image.shape == (180, 320, 3)
    assert image.dtype == np.uint8


def test_inspect_image_returns_expected_metadata(tmp_path) -> None:
    """Image inspection should return correct width, height, and channel count."""

    image_path = tmp_path / "sample.png"

    create_synthetic_sample_image(image_path, width=400, height=250)
    info = inspect_image(image_path)

    assert info.filename == "sample.png"
    assert info.width == 400
    assert info.height == 250
    assert info.channels == 3
    assert info.dtype == "uint8"
    assert info.file_size_bytes > 0


def test_save_image_writes_file(tmp_path) -> None:
    """Saving an image should create an output image file."""

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    output_path = tmp_path / "output.png"

    saved_path = save_image(image, output_path)

    assert saved_path.exists()
    assert saved_path.name == "output.png"
