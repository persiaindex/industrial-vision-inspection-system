"""Tests for Step 06 rule-based inspection pipeline."""

import cv2
import numpy as np

from industrial_vision.inspection import (
    create_dark_defect_mask,
    create_product_mask,
    draw_inspection_result,
    extract_defect_candidates,
    inspect_image_rule_based,
)


def create_test_part_image(with_defect: bool = True) -> np.ndarray:
    """Create a simple synthetic part image for tests."""

    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 40), (170, 120), color=(210, 210, 210), thickness=-1)

    if with_defect:
        cv2.circle(image, center=(110, 80), radius=12, color=(30, 30, 30), thickness=-1)

    return image


def test_create_product_mask_fills_product_area() -> None:
    """The product mask should include the product body area."""

    image = create_test_part_image(with_defect=True)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    product_mask = create_product_mask(grayscale, product_threshold=100, min_product_area=1000)

    assert product_mask.shape == grayscale.shape
    assert product_mask[80, 110] == 255


def test_create_dark_defect_mask_ignores_dark_background() -> None:
    """Dark background outside the product should not be treated as a defect."""

    image = create_test_part_image(with_defect=True)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    product_mask = create_product_mask(grayscale, product_threshold=100, min_product_area=1000)

    defect_mask = create_dark_defect_mask(grayscale, product_mask, dark_threshold=80)

    assert defect_mask[80, 110] == 255
    assert defect_mask[10, 10] == 0


def test_extract_defect_candidates_detects_dark_spot() -> None:
    """A dark spot inside the product should be extracted as a defect candidate."""

    image = create_test_part_image(with_defect=True)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    product_mask = create_product_mask(grayscale, product_threshold=100, min_product_area=1000)
    defect_mask = create_dark_defect_mask(grayscale, product_mask, dark_threshold=80)

    candidates = extract_defect_candidates(defect_mask, grayscale, min_defect_area=20.0)

    assert len(candidates) == 1
    assert candidates[0].area > 20.0
    assert candidates[0].mean_intensity < 80


def test_rule_based_inspection_fails_defective_part() -> None:
    """The inspection should fail when a dark defect is present."""

    image = create_test_part_image(with_defect=True)

    result, product_mask, defect_mask = inspect_image_rule_based(
        image,
        filename="defective.png",
        min_product_area=1000.0,
        min_defect_area=20.0,
    )

    assert result.status == "FAIL"
    assert result.passed is False
    assert result.defect_count >= 1
    assert product_mask.shape[:2] == image.shape[:2]
    assert defect_mask.shape[:2] == image.shape[:2]


def test_rule_based_inspection_passes_clean_part() -> None:
    """The inspection should pass when no dark defect is present."""

    image = create_test_part_image(with_defect=False)

    result, _, _ = inspect_image_rule_based(
        image,
        filename="clean.png",
        min_product_area=1000.0,
        min_defect_area=20.0,
    )

    assert result.status == "PASS"
    assert result.passed is True
    assert result.defect_count == 0


def test_draw_inspection_result_preserves_image_shape() -> None:
    """Drawing the inspection result should preserve the image shape."""

    image = create_test_part_image(with_defect=True)

    result, _, _ = inspect_image_rule_based(
        image,
        filename="defective.png",
        min_product_area=1000.0,
        min_defect_area=20.0,
    )
    visual = draw_inspection_result(image, result)

    assert visual.shape == image.shape
    assert visual.dtype == image.dtype
