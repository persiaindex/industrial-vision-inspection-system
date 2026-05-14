"""Simple browser dashboard for the industrial vision inference API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse

from industrial_vision.inference_api import create_app


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Industrial Vision Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="/dashboard.css" rel="stylesheet">
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <p class="eyebrow">Industrial Computer Vision</p>
        <h1>Inspection Dashboard</h1>
        <p class="subtitle">
          Upload a product image and receive a clean/defective prediction from the FastAPI inference service.
        </p>
      </div>
      <button id="healthButton" class="secondary-button">Check API Health</button>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Upload Image</h2>
        <p class="muted">Choose a PNG or JPG inspection image.</p>

        <form id="predictionForm">
          <input id="imageInput" name="file" type="file" accept="image/*" required>
          <button type="submit">Run Inspection</button>
        </form>

        <div class="preview-box">
          <img id="imagePreview" alt="Selected inspection preview">
          <p id="emptyPreviewText">No image selected yet.</p>
        </div>
      </article>

      <article class="card">
        <h2>Prediction Result</h2>
        <div id="statusBox" class="status-box">Waiting for an image...</div>

        <dl class="result-list">
          <div>
            <dt>Filename</dt>
            <dd id="filenameValue">-</dd>
          </div>
          <div>
            <dt>Predicted Label</dt>
            <dd id="labelValue">-</dd>
          </div>
          <div>
            <dt>Clean Probability</dt>
            <dd id="cleanProbabilityValue">-</dd>
          </div>
          <div>
            <dt>Defective Probability</dt>
            <dd id="defectiveProbabilityValue">-</dd>
          </div>
        </dl>
      </article>

      <article class="card wide">
        <h2>Feature Summary</h2>
        <div class="feature-grid">
          <div><span>Defect Count</span><strong id="defectCountValue">-</strong></div>
          <div><span>Total Defect Area</span><strong id="totalDefectAreaValue">-</strong></div>
          <div><span>Max Defect Area</span><strong id="maxDefectAreaValue">-</strong></div>
          <div><span>Defect Area Ratio</span><strong id="defectAreaRatioValue">-</strong></div>
          <div><span>Mean Intensity</span><strong id="meanIntensityValue">-</strong></div>
          <div><span>Edge Density</span><strong id="edgeDensityValue">-</strong></div>
        </div>
      </article>
    </section>
  </main>

  <script src="/dashboard.js"></script>
</body>
</html>
"""

DASHBOARD_CSS = """
:root {
  font-family: Arial, Helvetica, sans-serif;
  color: #172033;
  background: #f4f6fb;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
}

.page {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 32px 0;
}

.hero {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: center;
  margin-bottom: 24px;
  padding: 28px;
  border-radius: 24px;
  background: #ffffff;
  box-shadow: 0 16px 40px rgba(23, 32, 51, 0.08);
}

.eyebrow {
  margin: 0 0 8px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #526070;
}

h1,
h2 {
  margin: 0;
}

h1 {
  font-size: clamp(2rem, 4vw, 3.2rem);
}

h2 {
  font-size: 1.2rem;
  margin-bottom: 8px;
}

.subtitle,
.muted {
  color: #526070;
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 20px;
}

.card {
  padding: 24px;
  border-radius: 24px;
  background: #ffffff;
  box-shadow: 0 16px 40px rgba(23, 32, 51, 0.08);
}

.wide {
  grid-column: 1 / -1;
}

form {
  display: grid;
  gap: 14px;
  margin-top: 20px;
}

input[type="file"] {
  padding: 14px;
  border: 1px dashed #a7b0bd;
  border-radius: 16px;
  background: #f8fafc;
}

button {
  border: 0;
  border-radius: 16px;
  padding: 14px 18px;
  font-weight: 700;
  color: #ffffff;
  background: #172033;
  cursor: pointer;
}

.secondary-button {
  background: #eef2f7;
  color: #172033;
  white-space: nowrap;
}

.preview-box {
  display: grid;
  place-items: center;
  min-height: 260px;
  margin-top: 20px;
  border-radius: 18px;
  background: #f8fafc;
  overflow: hidden;
}

.preview-box img {
  display: none;
  max-width: 100%;
  max-height: 320px;
}

.status-box {
  margin: 18px 0;
  padding: 14px;
  border-radius: 16px;
  background: #eef2f7;
  color: #172033;
  font-weight: 700;
}

.status-box.pass {
  background: #e8f7ed;
}

.status-box.fail {
  background: #fff1f0;
}

.status-box.warning {
  background: #fff8df;
}

.result-list {
  display: grid;
  gap: 14px;
}

.result-list div,
.feature-grid div {
  padding: 14px;
  border-radius: 16px;
  background: #f8fafc;
}

dt,
.feature-grid span {
  display: block;
  margin-bottom: 6px;
  color: #526070;
  font-size: 0.85rem;
}

dd {
  margin: 0;
  font-weight: 700;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}

.feature-grid strong {
  font-size: 1.1rem;
}

@media (max-width: 760px) {
  .hero {
    align-items: flex-start;
    flex-direction: column;
  }

  .grid,
  .feature-grid {
    grid-template-columns: 1fr;
  }
}
"""

DASHBOARD_JS = """
const form = document.querySelector("#predictionForm");
const imageInput = document.querySelector("#imageInput");
const imagePreview = document.querySelector("#imagePreview");
const emptyPreviewText = document.querySelector("#emptyPreviewText");
const statusBox = document.querySelector("#statusBox");
const healthButton = document.querySelector("#healthButton");

const fields = {
  filename: document.querySelector("#filenameValue"),
  label: document.querySelector("#labelValue"),
  cleanProbability: document.querySelector("#cleanProbabilityValue"),
  defectiveProbability: document.querySelector("#defectiveProbabilityValue"),
  defectCount: document.querySelector("#defectCountValue"),
  totalDefectArea: document.querySelector("#totalDefectAreaValue"),
  maxDefectArea: document.querySelector("#maxDefectAreaValue"),
  defectAreaRatio: document.querySelector("#defectAreaRatioValue"),
  meanIntensity: document.querySelector("#meanIntensityValue"),
  edgeDensity: document.querySelector("#edgeDensityValue"),
};

function formatNumber(value) {
  if (value === undefined || value === null) {
    return "-";
  }

  return Number(value).toFixed(4);
}

function formatProbability(value) {
  if (value === undefined || value === null) {
    return "-";
  }

  return `${(Number(value) * 100).toFixed(1)}%`;
}

function setStatus(message, mode = "") {
  statusBox.textContent = message;
  statusBox.className = `status-box ${mode}`.trim();
}

function renderPrediction(data) {
  const probabilities = data.probabilities || {};
  const summary = data.feature_summary || {};

  fields.filename.textContent = data.filename || "-";
  fields.label.textContent = data.predicted_label || "-";
  fields.cleanProbability.textContent = formatProbability(probabilities.clean);
  fields.defectiveProbability.textContent = formatProbability(probabilities.defective);

  fields.defectCount.textContent = summary.defect_count ?? "-";
  fields.totalDefectArea.textContent = formatNumber(summary.total_defect_area);
  fields.maxDefectArea.textContent = formatNumber(summary.max_defect_area);
  fields.defectAreaRatio.textContent = formatNumber(summary.defect_area_ratio);
  fields.meanIntensity.textContent = formatNumber(summary.mean_intensity);
  fields.edgeDensity.textContent = formatNumber(summary.edge_density);

  if (data.predicted_label === "defective") {
    setStatus("Inspection result: DEFECTIVE", "fail");
  } else if (data.predicted_label === "clean") {
    setStatus("Inspection result: CLEAN", "pass");
  } else {
    setStatus("Inspection completed.", "warning");
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (!file) {
    imagePreview.style.display = "none";
    emptyPreviewText.style.display = "block";
    return;
  }

  const reader = new FileReader();

  reader.onload = () => {
    imagePreview.src = reader.result;
    imagePreview.style.display = "block";
    emptyPreviewText.style.display = "none";
  };

  reader.readAsDataURL(file);
});

healthButton.addEventListener("click", async () => {
  setStatus("Checking API health...", "warning");

  try {
    const response = await fetch("/health");
    const data = await response.json();

    if (data.model_available) {
      setStatus("API is healthy and the model is available.", "pass");
    } else {
      setStatus("API is running, but the model file is missing.", "warning");
    }
  } catch (error) {
    setStatus(`Health check failed: ${error}`, "fail");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];

  if (!file) {
    setStatus("Please choose an image first.", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setStatus("Running inspection...", "warning");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderPrediction(data);
  } catch (error) {
    setStatus(`Prediction failed: ${error.message}`, "fail");
  }
});
"""


def create_dashboard_app(model_path: str | Path | None = None) -> FastAPI:
    """Create a FastAPI app with API endpoints and a simple dashboard."""

    app = create_app(model_path=model_path)

    @app.get("/", response_class=HTMLResponse)
    def dashboard_home() -> str:
        """Serve the dashboard HTML page."""

        return DASHBOARD_HTML

    @app.get("/dashboard.css", response_class=PlainTextResponse)
    def dashboard_css() -> str:
        """Serve dashboard CSS."""

        return DASHBOARD_CSS

    @app.get("/dashboard.js", response_class=PlainTextResponse)
    def dashboard_js() -> str:
        """Serve dashboard JavaScript."""

        return DASHBOARD_JS

    return app


app = create_dashboard_app()
