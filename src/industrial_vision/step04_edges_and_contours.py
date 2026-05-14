"""Step 04 command script: edge detection and contours."""

from industrial_vision.config import PREPROCESSING_OUTPUT_DIR, SAMPLE_IMAGES_DIR, ensure_project_directories
from industrial_vision.edge_detection import detect_candidate_regions, draw_bounding_boxes
from industrial_vision.image_io import create_synthetic_sample_image, load_image, save_image
from industrial_vision.preprocessing import preprocess_basic


def main() -> None:
    """Run Canny edge detection, find contours, and draw bounding boxes."""

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

    edges, contours, bounding_boxes = detect_candidate_regions(
        blurred,
        low_threshold=50,
        high_threshold=150,
        min_area=20.0,
    )

    boxes_on_image = draw_bounding_boxes(image, bounding_boxes)

    edges_path = PREPROCESSING_OUTPUT_DIR / "step04_edges.png"
    threshold_path = PREPROCESSING_OUTPUT_DIR / "step04_threshold_reference.png"
    boxes_path = PREPROCESSING_OUTPUT_DIR / "step04_bounding_boxes.png"

    save_image(edges, edges_path)
    save_image(thresholded, threshold_path)
    save_image(boxes_on_image, boxes_path)

    print("Step 04 — Edge Detection and Contours")
    print(f"Input image: {sample_path}")
    print(f"Original shape: {image.shape}")
    print(f"Edge image shape: {edges.shape}")
    print(f"Number of contours: {len(contours)}")
    print(f"Number of bounding boxes after filtering: {len(bounding_boxes)}")

    for index, box in enumerate(bounding_boxes, start=1):
        print(
            f"Box {index}: "
            f"x={box.x}, y={box.y}, width={box.width}, height={box.height}, area={box.area:.2f}"
        )

    print(f"Saved edge image to: {edges_path}")
    print(f"Saved threshold reference image to: {threshold_path}")
    print(f"Saved bounding box image to: {boxes_path}")


if __name__ == "__main__":
    main()
