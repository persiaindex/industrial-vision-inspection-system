"""Tests for Step 04 edge detection and contour helpers."""

import numpy as np
import pytest

from industrial_vision.edge_detection import (
    BoundingBox,
    apply_canny_edges,
    contours_to_bounding_boxes,
    detect_candidate_regions,
    draw_bounding_boxes,
    find_external_contours,
)


def test_apply_canny_edges_returns_2d_edge_image() -> None:
    """Canny edge detection should return a 2D image with the same height and width."""

    image = np.zeros((100, 150), dtype=np.uint8)
    image[25:75, 40:110] = 255

    edges = apply_canny_edges(image)

    assert edges.shape == image.shape
    assert edges.dtype == np.uint8


def test_apply_canny_edges_rejects_color_image() -> None:
    """Canny edge detection should reject 3-channel images."""

    image = np.zeros((100, 150, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_canny_edges(image)


def test_find_external_contours_detects_rectangle() -> None:
    """Contour detection should find a simple white rectangle."""

    image = np.zeros((100, 150), dtype=np.uint8)
    image[25:75, 40:110] = 255

    contours = find_external_contours(image)

    assert len(contours) >= 1


def test_contours_to_bounding_boxes_returns_filtered_boxes() -> None:
    """Contours should be converted into bounding boxes."""

    image = np.zeros((100, 150), dtype=np.uint8)
    image[25:75, 40:110] = 255
    contours = find_external_contours(image)

    boxes = contours_to_bounding_boxes(contours, min_area=20.0)

    assert len(boxes) >= 1
    assert isinstance(boxes[0], BoundingBox)
    assert boxes[0].width > 0
    assert boxes[0].height > 0
    assert boxes[0].area > 0


def test_draw_bounding_boxes_preserves_color_shape() -> None:
    """Drawing boxes on a BGR image should preserve the image shape."""

    image = np.zeros((100, 150, 3), dtype=np.uint8)
    boxes = [BoundingBox(x=10, y=20, width=40, height=30, area=1200.0)]

    output = draw_bounding_boxes(image, boxes)

    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_detect_candidate_regions_returns_edges_contours_and_boxes() -> None:
    """The candidate detection helper should return edges, contours, and bounding boxes."""

    image = np.zeros((100, 150), dtype=np.uint8)
    image[25:75, 40:110] = 255

    edges, contours, boxes = detect_candidate_regions(image, min_area=20.0)

    assert edges.shape == image.shape
    assert len(contours) >= 1
    assert len(boxes) >= 1
