"""Step 07 command script: save inspection results as JSON and CSV."""

from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.image_io import create_synthetic_sample_image, load_image, save_image
from industrial_vision.inspection import draw_inspection_result, inspect_image_rule_based
from industrial_vision.result_export import (
    save_defect_candidates_csv,
    save_inspection_result_json,
    save_inspection_summary_csv,
)


def main() -> None:
    """Run inspection and save structured result files."""

    ensure_project_directories()

    sample_path = SAMPLE_IMAGES_DIR / "step02_sample_part.png"

    if not sample_path.exists():
        create_synthetic_sample_image(sample_path)

    image = load_image(sample_path)

    result, product_mask, defect_mask = inspect_image_rule_based(
        image,
        filename=sample_path.name,
        product_threshold=100,
        dark_defect_threshold=80,
        min_product_area=1000.0,
        min_defect_area=50.0,
    )

    visual_result = draw_inspection_result(image, result)

    json_path = PREDICTIONS_OUTPUT_DIR / "step07_inspection_result.json"
    summary_csv_path = PREDICTIONS_OUTPUT_DIR / "step07_inspection_summary.csv"
    candidates_csv_path = PREDICTIONS_OUTPUT_DIR / "step07_defect_candidates.csv"
    visual_result_path = PREDICTIONS_OUTPUT_DIR / "step07_inspection_result.png"
    product_mask_path = PREDICTIONS_OUTPUT_DIR / "step07_product_mask.png"
    defect_mask_path = PREDICTIONS_OUTPUT_DIR / "step07_defect_mask.png"

    save_inspection_result_json(result, json_path)
    save_inspection_summary_csv([result], summary_csv_path)
    save_defect_candidates_csv(result, candidates_csv_path)
    save_image(visual_result, visual_result_path)
    save_image(product_mask, product_mask_path)
    save_image(defect_mask, defect_mask_path)

    print("Step 07 — Save Inspection Results as JSON and CSV")
    print(f"Input image: {sample_path}")
    print(f"Status: {result.status}")
    print(f"Defect count: {result.defect_count}")
    print(f"Saved JSON result to: {json_path}")
    print(f"Saved summary CSV to: {summary_csv_path}")
    print(f"Saved candidates CSV to: {candidates_csv_path}")
    print(f"Saved visual result to: {visual_result_path}")
    print(f"Saved product mask to: {product_mask_path}")
    print(f"Saved defect mask to: {defect_mask_path}")


if __name__ == "__main__":
    main()
