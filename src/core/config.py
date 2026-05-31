import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    project_root: Path
    database_url: str | None
    redis_url: str | None
    prediction_log_stream: str
    prediction_log_group: str
    prediction_log_consumer: str
    prediction_log_batch_size: int
    feature_columns: tuple[str, ...]
    drift_alert_threshold: float

    @property
    def model_path(self) -> Path:
        return self.project_root / "models" / "isolation_forest.pkl"

    @property
    def scaler_path(self) -> Path:
        return self.project_root / "models" / "scaler.pkl"

    @property
    def monitoring_baseline_path(self) -> Path:
        return self.project_root / "models" / "monitoring_baseline.json"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def model_registry_dir(self) -> Path:
        return self.models_dir / "registry"

    @property
    def current_model_manifest_path(self) -> Path:
        return self.models_dir / "current_model.json"

    @property
    def training_data_path(self) -> Path:
        return self.project_root / "data" / "server_metrics.csv"

    @property
    def static_index_path(self) -> Path:
        return self.project_root / "src" / "static" / "index.html"


@lru_cache
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    return Settings(
        app_name="Sentinel - Server Anomaly Detection API",
        project_root=project_root,
        database_url=os.getenv("DATABASE_URL"),
        redis_url=os.getenv("REDIS_URL"),
        prediction_log_stream=os.getenv("PREDICTION_LOG_STREAM", "sentinel:prediction_logs"),
        prediction_log_group=os.getenv("PREDICTION_LOG_GROUP", "prediction-log-writers"),
        prediction_log_consumer=os.getenv("PREDICTION_LOG_CONSUMER", "worker-1"),
        prediction_log_batch_size=int(os.getenv("PREDICTION_LOG_BATCH_SIZE", "100")),
        drift_alert_threshold=float(os.getenv("DRIFT_ALERT_THRESHOLD", "2.5")),
        feature_columns=(
            "feature_15",
            "feature_16",
            "feature_19",
            "feature_20",
            "feature_9",
        ),
    )
