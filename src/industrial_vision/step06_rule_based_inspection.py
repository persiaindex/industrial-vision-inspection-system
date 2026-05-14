"""Step 06 command script: rule-based defect detection pipeline."""

from industrial_vision.config import PREDICTIONS_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.image_io import create_synthetic_sample_image, load_image, save_image
from industrial_vision.inspection import draw_inspection_result, inspect_image_rule_based


def main() -> None:
    """Run the rule-based inspection pipeline and save visual outputs."""

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

    product_mask_path = PREDICTIONS_OUTPUT_DIR / "step06_product_mask.png"
    defect_mask_path = PREDICTIONS_OUTPUT_DIR / "step06_defect_mask.png"
    visual_result_path = PREDICTIONS_OUTPUT_DIR / "step06_inspection_result.png"

    save_image(product_mask, product_mask_path)
    save_image(defect_mask, defect_mask_path)
    save_image(visual_result, visual_result_path)

    print("Step 06 — Rule-Based Defect Detection Pipeline")
    print(f"Input image: {sample_path}")
    print(f"Status: {result.status}")
    print(f"Passed: {result.passed}")
    print(f"Defect count: {result.defect_count}")
    print(f"Total defect area: {result.total_defect_area:.2f}")
    print(f"Max defect area: {result.max_defect_area:.2f}")

    for index, candidate in enumerate(result.candidates, start=1):
        print(
            f"Defect {index}: "
            f"x={candidate.x}, y={candidate.y}, "
            f"width={candidate.width}, height={candidate.height}, "
            f"area={candidate.area:.2f}, "
            f"mean_intensity={candidate.mean_intensity:.2f}"
        )

    print(f"Saved product mask to: {product_mask_path}")
    print(f"Saved defect mask to: {defect_mask_path}")
    print(f"Saved inspection result to: {visual_result_path}")


if __name__ == "__main__":
    main()
