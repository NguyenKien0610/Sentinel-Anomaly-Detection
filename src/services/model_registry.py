import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import Settings


@dataclass(frozen=True)
class ActiveModelBundle:
    version: str
    model_path: Path
    scaler_path: Path
    monitoring_baseline_path: Path
    metadata_path: Path | None
    metadata: dict[str, Any]
    source: str


class ModelRegistryService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load_active_bundle(self) -> ActiveModelBundle:
        manifest_path = self._settings.current_model_manifest_path
        if manifest_path.exists():
            return self._load_from_manifest(manifest_path)
        return self._load_legacy_bundle()

    def _load_from_manifest(self, manifest_path: Path) -> ActiveModelBundle:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts", {})

        model_path = self._resolve_project_path(artifacts["model"])
        scaler_path = self._resolve_project_path(artifacts["scaler"])
        baseline_path = self._resolve_project_path(artifacts["monitoring_baseline"])

        metadata_path: Path | None = None
        metadata: dict[str, Any] = {}
        metadata_rel = manifest.get("metadata_path")
        if metadata_rel:
            metadata_path = self._resolve_project_path(metadata_rel)
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        version = str(manifest.get("active_version", "unknown"))
        if metadata:
            version = str(metadata.get("model_version", version))

        return ActiveModelBundle(
            version=version,
            model_path=model_path,
            scaler_path=scaler_path,
            monitoring_baseline_path=baseline_path,
            metadata_path=metadata_path,
            metadata=metadata,
            source="manifest",
        )

    def _load_legacy_bundle(self) -> ActiveModelBundle:
        metadata = {
            "model_version": "legacy",
            "artifacts": {
                "model": self._relative_to_project(self._settings.model_path),
                "scaler": self._relative_to_project(self._settings.scaler_path),
                "monitoring_baseline": self._relative_to_project(
                    self._settings.monitoring_baseline_path
                ),
            },
            "features": list(self._settings.feature_columns),
            "source": "legacy-artifacts",
        }
        return ActiveModelBundle(
            version="legacy",
            model_path=self._settings.model_path,
            scaler_path=self._settings.scaler_path,
            monitoring_baseline_path=self._settings.monitoring_baseline_path,
            metadata_path=None,
            metadata=metadata,
            source="legacy",
        )

    def _resolve_project_path(self, relative_path: str) -> Path:
        return self._settings.project_root / relative_path

    def _relative_to_project(self, path: Path) -> str:
        return path.relative_to(self._settings.project_root).as_posix()
