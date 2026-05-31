import pickle

import pandas as pd

from src.core.config import Settings
from src.schemas import ServerMetrics
from src.services.model_registry import ActiveModelBundle, ModelRegistryService


class InferenceService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_registry = ModelRegistryService(settings)
        self.model = None
        self.scaler = None
        self.active_bundle: ActiveModelBundle | None = None
        self.load_error: str | None = None

    def load_artifacts(self) -> None:
        self.model = None
        self.scaler = None
        self.active_bundle = None
        self.load_error = None

        try:
            self.active_bundle = self._model_registry.load_active_bundle()
            if not self.active_bundle.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.active_bundle.model_path}")
            if not self.active_bundle.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self.active_bundle.scaler_path}")

            with self.active_bundle.model_path.open("rb") as model_file:
                self.model = pickle.load(model_file)

            with self.active_bundle.scaler_path.open("rb") as scaler_file:
                self.scaler = pickle.load(scaler_file)
        except Exception as exc:  # noqa: BLE001
            self.load_error = str(exc)

    def predict(self, payload: ServerMetrics) -> str:
        if self.load_error is not None:
            raise RuntimeError(f"Model artifacts unavailable: {self.load_error}")

        if self.model is None or self.scaler is None:
            raise RuntimeError("Model artifacts are not loaded.")

        try:
            feature_payload = payload.to_feature_payload()
            input_frame = pd.DataFrame([feature_payload], columns=self._settings.feature_columns)
            scaled_input = self.scaler.transform(input_frame)
            prediction = self.model.predict(scaled_input)[0]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Inference error: {exc}") from exc

        return "Normal" if prediction == 1 else "Anomaly"

    def model_info(self) -> dict[str, object]:
        if self.active_bundle is None:
            return {
                "loaded": False,
                "load_error": self.load_error,
            }

        return {
            "loaded": self.load_error is None and self.model is not None and self.scaler is not None,
            "load_error": self.load_error,
            "version": self.active_bundle.version,
            "source": self.active_bundle.source,
            "model_path": self.active_bundle.model_path.relative_to(
                self._settings.project_root
            ).as_posix(),
            "scaler_path": self.active_bundle.scaler_path.relative_to(
                self._settings.project_root
            ).as_posix(),
            "monitoring_baseline_path": self.active_bundle.monitoring_baseline_path.relative_to(
                self._settings.project_root
            ).as_posix(),
            "metadata": self.active_bundle.metadata,
        }
