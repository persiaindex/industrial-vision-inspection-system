# Industrial Vision Inspection System

A practical **industrial computer-vision portfolio project** built with Python and OpenCV.

The project demonstrates how an inspection workflow can connect image acquisition/preprocessing, defect analysis, feature extraction, a classical ML baseline, API-based inference, testing, and deployment-oriented packaging.

> This is a portfolio/learning project. It is separate from my professional employer projects and contains no proprietary production data.

## Why This Project Exists

Industrial inspection software often needs more than a single image-processing algorithm. A usable system must connect vision logic with repeatable preprocessing, measurable results, APIs, testing, and deployment.

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

The example inspection flow is:

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

## What This Project Demonstrates

- structured OpenCV-based inspection pipelines
- practical defect-analysis workflows
- connection between computer vision and backend services
- measurable testing and error analysis
- deployment-oriented packaging with Docker

It is relevant to roles in **Machine Vision, Computer Vision Engineering, Industrial AI, Quality Inspection Automation, Python Backend Development, and Manufacturing Software**.
