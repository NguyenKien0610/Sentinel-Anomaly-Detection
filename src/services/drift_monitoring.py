import json
from dataclasses import dataclass

import pandas as pd

from src.core.config import Settings
from src.services.model_registry import ModelRegistryService


@dataclass(frozen=True)
class DriftFeatureStat:
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float


@dataclass(frozen=True)
class DriftSnapshot:
    score: float
    threshold: float
    alert: bool
    per_feature_abs_zscore: dict[str, float]
    values: dict[str, float]


class DriftMonitoringService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_registry = ModelRegistryService(settings)
        self._baseline: dict[str, DriftFeatureStat] = {}
        self._baseline_source: str | None = None
        self._load_error: str | None = None
        self._latest_snapshot: DriftSnapshot | None = None

    @property
    def baseline_ready(self) -> bool:
        return bool(self._baseline)

    @property
    def baseline_source(self) -> str | None:
        return self._baseline_source

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def latest_snapshot(self) -> DriftSnapshot | None:
        return self._latest_snapshot

    def load_baseline(self) -> None:
        self._baseline = {}
        self._baseline_source = None
        self._load_error = None
        self._latest_snapshot = None

        try:
            active_bundle = self._model_registry.load_active_bundle()
            if active_bundle.monitoring_baseline_path.exists():
                self._load_from_baseline_file(active_bundle.monitoring_baseline_path)
                return

            self._load_from_training_data()
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)

    def evaluate(self, values: dict[str, float]) -> DriftSnapshot | None:
        if not self.baseline_ready:
            return None

        per_feature_abs_zscore: dict[str, float] = {}
        for feature_name, feature_value in values.items():
            feature_stat = self._baseline[feature_name]
            std = feature_stat.std if feature_stat.std > 1e-12 else 1.0
            per_feature_abs_zscore[feature_name] = abs((feature_value - feature_stat.mean) / std)

        drift_score = sum(per_feature_abs_zscore.values()) / len(per_feature_abs_zscore)
        snapshot = DriftSnapshot(
            score=float(drift_score),
            threshold=self._settings.drift_alert_threshold,
            alert=drift_score >= self._settings.drift_alert_threshold,
            per_feature_abs_zscore=per_feature_abs_zscore,
            values=values,
        )
        self._latest_snapshot = snapshot
        return snapshot

    def current_status(self) -> dict[str, object]:
        return {
            "baseline_ready": self.baseline_ready,
            "baseline_source": self._baseline_source,
            "load_error": self._load_error,
            "threshold": self._settings.drift_alert_threshold,
            "latest_snapshot": None
            if self._latest_snapshot is None
            else {
                "score": self._latest_snapshot.score,
                "threshold": self._latest_snapshot.threshold,
                "alert": self._latest_snapshot.alert,
                "per_feature_abs_zscore": self._latest_snapshot.per_feature_abs_zscore,
                "values": self._latest_snapshot.values,
            },
        }

    def _load_from_baseline_file(self, baseline_path) -> None:
        with baseline_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", [])
        summary = payload.get("summary", {})
        if list(features) != list(self._settings.feature_columns):
            raise ValueError(
                "Monitoring baseline features do not match configured inference features."
            )

        self._baseline = {
            feature_name: DriftFeatureStat(
                mean=float(summary[feature_name]["mean"]),
                std=float(summary[feature_name]["std"]),
                min=float(summary[feature_name]["min"]),
                max=float(summary[feature_name]["max"]),
                p50=float(summary[feature_name]["p50"]),
                p95=float(summary[feature_name]["p95"]),
            )
            for feature_name in self._settings.feature_columns
        }
        self._baseline_source = baseline_path.relative_to(self._settings.project_root).as_posix()

    def _load_from_training_data(self) -> None:
        if not self._settings.training_data_path.exists():
            raise FileNotFoundError(
                f"Training data not found: {self._settings.training_data_path}"
            )

        df = pd.read_csv(self._settings.training_data_path)
        feature_df = df[list(self._settings.feature_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        feature_df = feature_df.dropna()
        if feature_df.empty:
            raise ValueError("Cannot build monitoring baseline from empty feature set.")

        self._baseline = {
            feature_name: DriftFeatureStat(
                mean=float(feature_df[feature_name].mean()),
                std=float(feature_df[feature_name].std(ddof=0)),
                min=float(feature_df[feature_name].min()),
                max=float(feature_df[feature_name].max()),
                p50=float(feature_df[feature_name].median()),
                p95=float(feature_df[feature_name].quantile(0.95)),
            )
            for feature_name in self._settings.feature_columns
        }
        self._baseline_source = "data/server_metrics.csv"
