"""Small helper for checking the containerized inference service."""

from industrial_vision.inference_service import get_model_path_from_environment


def main() -> None:
    """Print model path information used by the container."""

    model_path = get_model_path_from_environment()

    print("Industrial Vision Container Health Helper")
    print(f"Model path: {model_path}")
    print(f"Model exists: {model_path.exists()}")


if __name__ == "__main__":
    main()
