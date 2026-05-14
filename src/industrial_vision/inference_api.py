"""FastAPI inference service for industrial vision defect classification."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from industrial_vision.inference_service import (
    decode_image_bytes,
    get_model_path_from_environment,
    load_inference_model,
    predict_image_with_model_bundle,
)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    selected_model_path = (
        Path(model_path) if model_path is not None else get_model_path_from_environment()
    )

    app = FastAPI(
        title="Industrial Vision Inference API",
        description="FastAPI service for clean/defective image classification.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        """Return service and model availability status."""

        return {
            "status": "ok",
            "model_path": str(selected_model_path),
            "model_available": selected_model_path.exists(),
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        """Predict whether an uploaded inspection image is clean or defective."""

        if not selected_model_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model file is not available: {selected_model_path}",
            )

        image_bytes = await file.read()

        try:
            image = decode_image_bytes(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            model_bundle = load_inference_model(selected_model_path)
            prediction = predict_image_with_model_bundle(
                model_bundle=model_bundle,
                image=image,
                filename=file.filename or "uploaded_image",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {exc}",
            ) from exc

        return prediction

    return app


app = create_app()
