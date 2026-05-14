"""Edge detection and contour helpers for industrial vision inspection."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """A rectangular region around a detected contour."""

    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def x2(self) -> int:
        """Right-side x coordinate."""

        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom-side y coordinate."""

        return self.y + self.height


def apply_canny_edges(
    grayscale_image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """Apply Canny edge detection to a grayscale image."""

    if grayscale_image.ndim != 2:
        raise ValueError("Canny edge detection expects a single-channel grayscale image.")

    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Canny thresholds must be non-negative.")

    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be smaller than high_threshold.")

    return cv2.Canny(grayscale_image, low_threshold, high_threshold)


def find_external_contours(binary_or_edge_image: np.ndarray) -> list[np.ndarray]:
    """Find external contours from a binary or edge image."""

    if binary_or_edge_image.ndim != 2:
        raise ValueError("Contour detection expects a single-channel image.")

    contours, _ = cv2.findContours(
        binary_or_edge_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return list(contours)


def contour_to_bounding_box(contour: np.ndarray) -> BoundingBox:
    """Convert one contour to a bounding box."""

    x, y, width, height = cv2.boundingRect(contour)
    area = float(cv2.contourArea(contour))

    return BoundingBox(
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
        area=area,
    )


def contours_to_bounding_boxes(
    contours: list[np.ndarray],
    min_area: float = 20.0,
) -> list[BoundingBox]:
    """Convert contours into bounding boxes and remove very small regions."""

    bounding_boxes: list[BoundingBox] = []

    for contour in contours:
        box = contour_to_bounding_box(contour)

        if box.area >= min_area:
            bounding_boxes.append(box)

    return bounding_boxes


def draw_bounding_boxes(
    image: np.ndarray,
    bounding_boxes: list[BoundingBox],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a copy of an image."""

    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        output = image.copy()
    else:
        raise ValueError(f"Expected grayscale or BGR image, got shape: {image.shape}")

    for box in bounding_boxes:
        cv2.rectangle(
            output,
            (box.x, box.y),
            (box.x2, box.y2),
            color=color,
            thickness=thickness,
        )

    return output


def detect_candidate_regions(
    grayscale_image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    min_area: float = 20.0,
) -> tuple[np.ndarray, list[np.ndarray], list[BoundingBox]]:
    """Detect edge-based candidate regions from a grayscale image.

    Returns:
        edges: Canny edge image
        contours: detected external contours
        bounding_boxes: filtered bounding boxes
    """

    edges = apply_canny_edges(
        grayscale_image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    contours = find_external_contours(edges)
    bounding_boxes = contours_to_bounding_boxes(contours, min_area=min_area)

    return edges, contours, bounding_boxes
