from fastapi.testclient import TestClient

from src.main import app


def test_healthz_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_component_checks() -> None:
    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code in {200, 503}
    body = response.json()
    assert "status" in body
    assert "checks" in body
    assert "model" in body["checks"]
    assert "drift_monitoring" in body["checks"]
    assert "database" in body["checks"]
    assert "redis" in body["checks"]


def test_model_info_endpoint_returns_active_model_state() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    body = response.json()
    assert "loaded" in body
    assert "version" in body or body["loaded"] is False


def test_drift_status_endpoint_returns_monitoring_state() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/monitoring/drift")

    assert response.status_code == 200
    body = response.json()
    assert "baseline_ready" in body
    assert "threshold" in body


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
    assert response.headers["X-Request-ID"]
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
