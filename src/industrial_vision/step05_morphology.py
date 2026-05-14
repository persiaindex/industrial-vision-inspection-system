"""Step 05 command script: morphological operations."""

from industrial_vision.config import PREPROCESSING_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.image_io import create_synthetic_sample_image, load_image, save_image
from industrial_vision.morphology import (
    apply_closing,
    apply_opening,
    clean_binary_mask,
    dilate_image,
    erode_image,
)
from industrial_vision.preprocessing import preprocess_basic


def main() -> None:
    """Run morphological operations on a thresholded image and save outputs."""

    ensure_project_directories()

    sample_path = SAMPLE_IMAGES_DIR / "step02_sample_part.png"

    if not sample_path.exists():
        create_synthetic_sample_image(sample_path)

    image = load_image(sample_path)

    _, _, thresholded = preprocess_basic(
        image,
        blur_kernel_size=(5, 5),
        threshold_value=100,
        invert_threshold=False,
    )

    eroded = erode_image(thresholded, kernel_size=(5, 5), iterations=1)
    dilated = dilate_image(thresholded, kernel_size=(5, 5), iterations=1)
    opened = apply_opening(thresholded, kernel_size=(5, 5))
    closed = apply_closing(thresholded, kernel_size=(7, 7))
    cleaned = clean_binary_mask(
        thresholded,
        opening_kernel_size=(5, 5),
        closing_kernel_size=(7, 7),
    )

    threshold_path = PREPROCESSING_OUTPUT_DIR / "step05_threshold_input.png"
    eroded_path = PREPROCESSING_OUTPUT_DIR / "step05_eroded.png"
    dilated_path = PREPROCESSING_OUTPUT_DIR / "step05_dilated.png"
    opened_path = PREPROCESSING_OUTPUT_DIR / "step05_opened.png"
    closed_path = PREPROCESSING_OUTPUT_DIR / "step05_closed.png"
    cleaned_path = PREPROCESSING_OUTPUT_DIR / "step05_cleaned_mask.png"

    save_image(thresholded, threshold_path)
    save_image(eroded, eroded_path)
    save_image(dilated, dilated_path)
    save_image(opened, opened_path)
    save_image(closed, closed_path)
    save_image(cleaned, cleaned_path)

    print("Step 05 — Morphological Operations")
    print(f"Input image: {sample_path}")
    print(f"Threshold image shape: {thresholded.shape}")
    print(f"Eroded shape: {eroded.shape}")
    print(f"Dilated shape: {dilated.shape}")
    print(f"Opened shape: {opened.shape}")
    print(f"Closed shape: {closed.shape}")
    print(f"Cleaned mask shape: {cleaned.shape}")
    print(f"Saved threshold input to: {threshold_path}")
    print(f"Saved eroded image to: {eroded_path}")
    print(f"Saved dilated image to: {dilated_path}")
    print(f"Saved opened image to: {opened_path}")
    print(f"Saved closed image to: {closed_path}")
    print(f"Saved cleaned mask to: {cleaned_path}")


if __name__ == "__main__":
    main()
