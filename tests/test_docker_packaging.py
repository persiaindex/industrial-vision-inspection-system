"""Tests for Step 15 Docker packaging files."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exists_and_uses_python_311() -> None:
    """Dockerfile should exist and use a Python 3.11 base image."""

    dockerfile = PROJECT_ROOT / "Dockerfile"

    content = dockerfile.read_text(encoding="utf-8")

    assert dockerfile.exists()
    assert "FROM python:3.11-slim" in content
    assert "PYTHONPATH=/app/src" in content
    assert "uvicorn" in content


def test_dockerfile_exposes_api_port() -> None:
    """Dockerfile should expose the FastAPI service port."""

    content = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "EXPOSE 8001" in content
    assert "HEALTHCHECK" in content
    assert "/health" in content


def test_docker_compose_file_exists_and_maps_port() -> None:
    """Compose file should define the API service and port mapping."""

    compose_file = PROJECT_ROOT / "docker-compose.yml"

    content = compose_file.read_text(encoding="utf-8")

    assert compose_file.exists()
    assert "industrial-vision-api" in content
    assert "${INDUSTRIAL_VISION_API_PORT:-8001}:8001" in content
    assert "industrial_vision.dashboard:app" in content


def test_docker_compose_mounts_outputs_and_data() -> None:
    """Compose should mount data and outputs folders for model and images."""

    content = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "./outputs:/app/outputs" in content
    assert "./data:/app/data" in content


def test_dockerignore_excludes_local_artifacts() -> None:
    """Docker ignore file should exclude local cache and artifact folders."""

    dockerignore = PROJECT_ROOT / ".dockerignore"

    content = dockerignore.read_text(encoding="utf-8")

    assert dockerignore.exists()
    assert ".venv/" in content
    assert "__pycache__/" in content
    assert "outputs/" in content
    assert "data/" in content


def test_docker_env_example_exists() -> None:
    """Docker environment example file should document the API port."""

    env_file = PROJECT_ROOT / ".env.docker.example"

    content = env_file.read_text(encoding="utf-8")

    assert env_file.exists()
    assert "INDUSTRIAL_VISION_API_PORT=8001" in content
