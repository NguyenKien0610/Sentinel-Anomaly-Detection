# 🛡️ Sentinel: Server Anomaly Detection API

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

A lightweight, production-ready Machine Learning API service designed to analyze real-time system metrics (CPU, RAM, Network) and predict server anomalies to safeguard against potential infrastructure overloads.

This project was built using the **Pooled Server Metrics (PSM)** dataset (eBay telemetry data) and implements an Unsupervised Machine Learning model (**Isolation Forest**).

## 🏗️ System Architecture
1. **Model Training:** An offline script processes raw telemetry data, scales features, and trains an Isolation Forest model to isolate anomalous data points (saved as `.pkl` artifacts).
2. **Inference API:** A high-performance FastAPI server loads the artifacts into memory and exposes a RESTful endpoint for real-time, low-latency predictions.

## 🚀 Features
* **Unsupervised Learning:** Utilizes `scikit-learn`'s Isolation Forest.
* **High Performance:** Built with FastAPI and Uvicorn for asynchronous request handling.
* **Data Validation:** Strict payload validation using Pydantic schemas.
* **Test Coverage:** Automated unit testing suite implemented with `Pytest`.
* **Containerized:** Fully deployable via Docker with an optimized, lightweight image layer caching strategy.

## ⚙️ Local Installation & Setup

### Option 1: Running with Docker (Recommended)
Ensure you have Docker installed. No Python environment setup is required.

```bash
# 1. Clone the repository
git clone [https://github.com/NguyenKien0610/Sentinel-Anomaly-Detection.git](https://github.com/NguyenKien0610/Sentinel-Anomaly-Detection.git)
cd Sentinel-Anomaly-Detection

# 2. Build the Docker image
docker build -t sentinel-api .

# 3. Run the container
docker run -d -p 8000:8000 sentinel-api

```

### Option 2: Running with Python Venv

```bash
# 1. Clone and navigate to the directory
git clone [https://github.com/NguyenKien0610/Sentinel-Anomaly-Detection.git](https://github.com/NguyenKien0610/Sentinel-Anomaly-Detection.git)
cd Sentinel-Anomaly-Detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

```

## 📡 API Usage

Once the server is running, the interactive API documentation (Swagger UI) is available at:
👉 **http://localhost:8000/docs**

### Endpoint: `POST /api/v1/analyze`

**Request Payload:**

```json
{
  "feature_15": 0.3187968986906816,
  "feature_16": 0.4518560659934575,
  "feature_19": 0.6098826098826098,
  "feature_20": 0.0084317032040472,
  "feature_9": 0.4470765464645514
}

```

**Response Payload:**

```json
{
  "status": "success",
  "prediction": "Anomaly",
  "timestamp": "2026-05-26T12:00:00Z"
}

```

## 🧪 Running Tests

To execute the automated test suite (requires local Python environment):

```bash
python -m pytest tests/test_api.py -v

```

## 📄 License

This project is licensed under the MIT License.
