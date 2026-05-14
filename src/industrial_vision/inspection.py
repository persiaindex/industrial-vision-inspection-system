"""Rule-based defect inspection pipeline for industrial vision."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from industrial_vision.preprocessing import apply_gaussian_blur, convert_to_grayscale


@dataclass(frozen=True)
class DefectCandidate:
    """Measured information about one detected defect candidate."""

    x: int
    y: int
    width: int
    height: int
    area: float
    mean_intensity: float

    @property
    def x2(self) -> int:
        """Right-side x coordinate."""

        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom-side y coordinate."""

        return self.y + self.height


@dataclass(frozen=True)
class InspectionResult:
    """Final result of a rule-based inspection."""

    filename: str
    status: str
    passed: bool
    defect_count: int
    total_defect_area: float
    max_defect_area: float
    candidates: list[DefectCandidate]

    def to_dict(self) -> dict:
        """Convert the result into a JSON-friendly dictionary."""

        result = asdict(self)
        return result


def validate_grayscale_image(image: np.ndarray) -> None:
    """Validate that an image is grayscale."""

    if image.ndim != 2:
        raise ValueError(f"Expected a single-channel grayscale image, got shape: {image.shape}")


def create_product_mask(
    grayscale_image: np.ndarray,
    product_threshold: int = 100,
    min_product_area: float = 1000.0,
) -> np.ndarray:
    """Create a filled product mask from a grayscale image.

    The function thresholds bright product regions, finds the largest external
    contour, and fills it. Filling the largest contour also fills dark holes
    inside the product, which allows dark defects to be detected later.
    """

    validate_grayscale_image(grayscale_image)

    _, raw_mask = cv2.threshold(
        grayscale_image,
        product_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        raw_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_product_area
    ]

    product_mask = np.zeros_like(grayscale_image, dtype=np.uint8)

    if not valid_contours:
        return product_mask

    largest_contour = max(valid_contours, key=cv2.contourArea)

    cv2.drawContours(
        product_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    return product_mask


def create_dark_defect_mask(
    grayscale_image: np.ndarray,
    product_mask: np.ndarray,
    dark_threshold: int = 80,
) -> np.ndarray:
    """Create a mask of dark defect candidates inside the product area."""

    validate_grayscale_image(grayscale_image)
    validate_grayscale_image(product_mask)

    if grayscale_image.shape != product_mask.shape:
        raise ValueError("grayscale_image and product_mask must have the same shape.")

    dark_mask = cv2.inRange(grayscale_image, 0, dark_threshold)
    defect_mask = cv2.bitwise_and(dark_mask, product_mask)

    return defect_mask


def extract_defect_candidates(
    defect_mask: np.ndarray,
    grayscale_image: np.ndarray,
    min_defect_area: float = 50.0,
) -> list[DefectCandidate]:
    """Extract measured defect candidates from a binary defect mask."""

    validate_grayscale_image(defect_mask)
    validate_grayscale_image(grayscale_image)

    if defect_mask.shape != grayscale_image.shape:
        raise ValueError("defect_mask and grayscale_image must have the same shape.")

    contours, _ = cv2.findContours(
        defect_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[DefectCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_defect_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        contour_mask = np.zeros_like(grayscale_image, dtype=np.uint8)
        cv2.drawContours(
            contour_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        mean_intensity = float(cv2.mean(grayscale_image, mask=contour_mask)[0])

        candidates.append(
            DefectCandidate(
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                area=area,
                mean_intensity=mean_intensity,
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.area, reverse=True)


def inspect_image_rule_based(
    image: np.ndarray,
    filename: str = "image",
    product_threshold: int = 100,
    dark_defect_threshold: int = 80,
    min_product_area: float = 1000.0,
    min_defect_area: float = 50.0,
) -> tuple[InspectionResult, np.ndarray, np.ndarray]:
    """Run a simple rule-based industrial defect inspection.

    Returns:
        result: measured pass/fail inspection result
        product_mask: filled product area mask
        defect_mask: binary mask of dark defect candidates
    """

    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale, kernel_size=(5, 5))

    product_mask = create_product_mask(
        blurred,
        product_threshold=product_threshold,
        min_product_area=min_product_area,
    )

    defect_mask = create_dark_defect_mask(
        blurred,
        product_mask=product_mask,
        dark_threshold=dark_defect_threshold,
    )

    candidates = extract_defect_candidates(
        defect_mask,
        blurred,
        min_defect_area=min_defect_area,
    )

    defect_count = len(candidates)
    total_defect_area = float(sum(candidate.area for candidate in candidates))
    max_defect_area = float(max((candidate.area for candidate in candidates), default=0.0))
    passed = defect_count == 0
    status = "PASS" if passed else "FAIL"

    result = InspectionResult(
        filename=filename,
        status=status,
        passed=passed,
        defect_count=defect_count,
        total_defect_area=total_defect_area,
        max_defect_area=max_defect_area,
        candidates=candidates,
    )

    return result, product_mask, defect_mask


def draw_inspection_result(
    image: np.ndarray,
    result: InspectionResult,
) -> np.ndarray:
    """Draw inspection result and defect boxes on an image."""

    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        output = image.copy()
    else:
        raise ValueError(f"Expected grayscale or BGR image, got shape: {image.shape}")

    box_color = (0, 0, 255) if not result.passed else (0, 180, 0)

    for candidate in result.candidates:
        cv2.rectangle(
            output,
            (candidate.x, candidate.y),
            (candidate.x2, candidate.y2),
            color=box_color,
            thickness=2,
        )

    label = f"{result.status} | defects: {result.defect_count}"

    cv2.putText(
        output,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        box_color,
        2,
        cv2.LINE_AA,
    )

    return output
