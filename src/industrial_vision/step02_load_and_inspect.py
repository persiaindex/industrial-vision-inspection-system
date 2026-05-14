"""Step 02 command script: load and inspect an image with OpenCV."""

from industrial_vision.config import PREPROCESSING_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.image_io import create_synthetic_sample_image, inspect_image, load_image, save_image


def main() -> None:
    """Create a sample image, load it, inspect it, and save a copy."""

    ensure_project_directories()

    sample_path = SAMPLE_IMAGES_DIR / "step02_sample_part.png"
    output_path = PREPROCESSING_OUTPUT_DIR / "step02_loaded_copy.png"

    if not sample_path.exists():
        create_synthetic_sample_image(sample_path)

    image = load_image(sample_path)
    info = inspect_image(sample_path)
    save_image(image, output_path)

    print("Step 02 — Load and Inspect Images with OpenCV")
    print(f"Filename: {info.filename}")
    print(f"Path: {info.path}")
    print(f"Width: {info.width}")
    print(f"Height: {info.height}")
    print(f"Channels: {info.channels}")
    print(f"Data type: {info.dtype}")
    print(f"Min pixel value: {info.min_value}")
    print(f"Max pixel value: {info.max_value}")
    print(f"Mean intensity: {info.mean_intensity:.2f}")
    print(f"File size bytes: {info.file_size_bytes}")
    print(f"Saved copy to: {output_path}")


if __name__ == "__main__":
    main()
