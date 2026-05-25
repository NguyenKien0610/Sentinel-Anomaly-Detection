from fastapi.testclient import TestClient

from src.main import app


def test_analyze_with_valid_payload_returns_prediction() -> None:
    payload = {
        "feature_15": 0.3187968986906816,
        "feature_16": 0.4518560659934575,
        "feature_19": 0.6098826098826098,
        "feature_20": 0.0084317032040472,
        "feature_9": 0.4470765464645514,
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body


def test_analyze_with_missing_required_fields_returns_422() -> None:
    payload = {
        "feature_15": 0.3187968986906816,
        "feature_16": 0.4518560659934575,
        "feature_19": 0.6098826098826098,
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json=payload)

    assert response.status_code == 422


def test_analyze_with_invalid_field_type_returns_422() -> None:
    payload = {
        "feature_15": "invalid_string",
        "feature_16": 0.4518560659934575,
        "feature_19": 0.6098826098826098,
        "feature_20": 0.0084317032040472,
        "feature_9": 0.4470765464645514,
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", json=payload)

    assert response.status_code == 422
