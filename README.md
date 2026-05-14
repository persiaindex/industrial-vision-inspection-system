# Industrial Vision Inspection System

A practical computer-vision and AI application engineering project for industrial inspection.

This project demonstrates an end-to-end workflow:

```text
image preprocessing
→ defect detection
→ feature extraction
→ classical ML baseline
→ model evaluation
→ saved model artifact
→ FastAPI inference API
→ browser dashboard
→ Docker packaging
```

## Why This Project Exists

Industrial companies often need practical AI systems that connect image processing, backend services, dashboards, and deployment workflows.

This project focuses on a realistic inspection scenario:

```text
product image → clean/defective prediction → dashboard result
```

## Main Features

- OpenCV image loading and preprocessing
- thresholding, edges, contours, and morphology
- rule-based defect detection
- batch processing
- CSV/JSON result export
- numerical feature extraction
- Random Forest defect classifier
- model evaluation and error analysis
- saved `.joblib` model artifact
- FastAPI `/health` and `/predict` endpoints
- browser-based inspection dashboard
- Docker and Docker Compose support
- automated tests

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Computer Vision | OpenCV, NumPy |
| Data | Pandas |
| Machine Learning | scikit-learn, joblib |
| API | FastAPI, Uvicorn |
| Testing | pytest |
| Deployment | Docker, Docker Compose |

## Quick Start

### 1. Create virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Prepare the model

```powershell
python -m industrial_vision.step14_prepare_dashboard_model
```

### 4. Start the dashboard

```powershell
uvicorn industrial_vision.dashboard:app --reload --host 127.0.0.1 --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Browser dashboard |
| GET | `/health` | Service/model health check |
| POST | `/predict` | Upload image and receive prediction |

## Docker Usage

Prepare the model locally first:

```powershell
python -m industrial_vision.step14_prepare_dashboard_model
```

Run with Docker Compose:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:8001/
```

## Testing

```powershell
python -m pytest
```

Expected after Step 16:

```text
99 passed
```

## Portfolio Summary

This project shows how to build a practical industrial computer-vision system that connects image processing, ML inference, API design, dashboard integration, testing, and Docker packaging.

It is suitable for roles related to Computer Vision Engineering, AI Application Engineering, Industrial AI, Quality Inspection Automation, Python Backend Development, and Manufacturing Software Tools.
