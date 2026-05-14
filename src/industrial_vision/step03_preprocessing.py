"""Step 03 command script: grayscale, blur, and thresholding."""

from industrial_vision.config import PREPROCESSING_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.image_io import create_synthetic_sample_image, load_image, save_image
from industrial_vision.preprocessing import preprocess_basic


def main() -> None:
    """Run the Step 03 preprocessing pipeline and save output images."""

    ensure_project_directories()

    sample_path = SAMPLE_IMAGES_DIR / "step02_sample_part.png"

    if not sample_path.exists():
        create_synthetic_sample_image(sample_path)

    image = load_image(sample_path)

    grayscale, blurred, thresholded = preprocess_basic(
        image,
        blur_kernel_size=(5, 5),
        threshold_value=100,
        invert_threshold=False,
    )

    grayscale_path = PREPROCESSING_OUTPUT_DIR / "step03_grayscale.png"
    blurred_path = PREPROCESSING_OUTPUT_DIR / "step03_blurred.png"
    threshold_path = PREPROCESSING_OUTPUT_DIR / "step03_threshold.png"

    save_image(grayscale, grayscale_path)
    save_image(blurred, blurred_path)
    save_image(thresholded, threshold_path)

    print("Step 03 — Grayscale, Blur, and Thresholding")
    print(f"Input image: {sample_path}")
    print(f"Original shape: {image.shape}")
    print(f"Grayscale shape: {grayscale.shape}")
    print(f"Blurred shape: {blurred.shape}")
    print(f"Thresholded shape: {thresholded.shape}")
    print(f"Saved grayscale image to: {grayscale_path}")
    print(f"Saved blurred image to: {blurred_path}")
    print(f"Saved threshold image to: {threshold_path}")


if __name__ == "__main__":
    main()
