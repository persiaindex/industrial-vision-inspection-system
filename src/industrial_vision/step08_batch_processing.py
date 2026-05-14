"""Step 08 command script: batch image processing."""

from industrial_vision.batch_processing import (
    count_statuses,
    create_demo_batch_images,
    run_batch_inspection,
)
from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories


def main() -> None:
    """Create demo batch images and run batch inspection."""

    ensure_project_directories()

    batch_input_dir = SAMPLE_IMAGES_DIR / "step08_batch_images"
    batch_output_dir = PREDICTIONS_OUTPUT_DIR / "step08_batch"

    created_images = create_demo_batch_images(batch_input_dir)
    results = run_batch_inspection(batch_input_dir, batch_output_dir)
    counts = count_statuses(results)

    print("Step 08 — Batch Image Processing")
    print(f"Batch input directory: {batch_input_dir}")
    print(f"Batch output directory: {batch_output_dir}")
    print(f"Demo images created: {len(created_images)}")
    print(f"Images inspected: {len(results)}")
    print(f"PASS count: {counts.get('PASS', 0)}")
    print(f"FAIL count: {counts.get('FAIL', 0)}")
    print(f"Saved batch summary to: {batch_output_dir / 'batch_inspection_summary.csv'}")
    print(f"Saved JSON results to: {batch_output_dir / 'json'}")
    print(f"Saved candidate CSV files to: {batch_output_dir / 'candidates'}")
    print(f"Saved masks to: {batch_output_dir / 'masks'}")
    print(f"Saved visual inspection images to: {batch_output_dir / 'visuals'}")


if __name__ == "__main__":
    main()
