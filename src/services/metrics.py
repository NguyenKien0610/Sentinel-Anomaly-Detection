from fastapi import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


TOTAL_REQUESTS = Counter(
    "total_requests",
    "Total number of POST /api/v1/analyze requests.",
)
ANOMALY_DETECTED_TOTAL = Counter(
    "anomaly_detected_total",
    "Total number of anomaly predictions.",
)
DRIFT_ALERT_TOTAL = Counter(
    "drift_alert_total",
    "Total number of drift alerts triggered.",
)
PREDICTION_LABEL_TOTAL = Counter(
    "prediction_label_total",
    "Total number of predictions by label.",
    ["prediction"],
)
REQUEST_LATENCY_SECONDS = Histogram(
    "request_latency_seconds",
    "Inference request latency in seconds.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INPUT_DRIFT_SCORE = Gauge(
    "input_drift_score",
    "Mean absolute z-score of the latest inference payload against the training baseline.",
)
INPUT_DRIFT_THRESHOLD = Gauge(
    "input_drift_threshold",
    "Configured drift alert threshold.",
)
FEATURE_ABS_ZSCORE = Gauge(
    "feature_abs_zscore",
    "Absolute z-score of the latest request by feature.",
    ["feature"],
)
FEATURE_INPUT_VALUE = Gauge(
    "feature_input_value",
    "Latest request value by feature.",
    ["feature"],
)


def record_request() -> None:
    TOTAL_REQUESTS.inc()


def record_anomaly() -> None:
    ANOMALY_DETECTED_TOTAL.inc()


def record_prediction_label(prediction_label: str) -> None:
    PREDICTION_LABEL_TOTAL.labels(prediction=prediction_label).inc()


def record_request_latency(duration_seconds: float) -> None:
    REQUEST_LATENCY_SECONDS.observe(duration_seconds)


def record_drift_snapshot(
    score: float,
    threshold: float,
    per_feature_abs_zscore: dict[str, float],
    values: dict[str, float],
    alert: bool,
) -> None:
    INPUT_DRIFT_SCORE.set(score)
    INPUT_DRIFT_THRESHOLD.set(threshold)
    for feature_name, zscore in per_feature_abs_zscore.items():
        FEATURE_ABS_ZSCORE.labels(feature=feature_name).set(zscore)
    for feature_name, value in values.items():
        FEATURE_INPUT_VALUE.labels(feature=feature_name).set(value)
    if alert:
        DRIFT_ALERT_TOTAL.inc()


def build_metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
