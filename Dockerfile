# Docker image for the Industrial Vision FastAPI dashboard/inference service.
#
# Build:
#   docker build -t industrial-vision-inspection:step15 .
#
# Run:
#   docker run --rm -p 8001:8001 industrial-vision-inspection:step15

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV INDUSTRIAL_VISION_MODEL_PATH=/app/outputs/predictions/step12_model_persistence/step12_random_forest_model.joblib

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY docs ./docs
COPY README.md ./README.md

RUN mkdir -p /app/data /app/outputs

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:8001/health || exit 1

CMD ["uvicorn", "industrial_vision.dashboard:app", "--host", "0.0.0.0", "--port", "8001"]
