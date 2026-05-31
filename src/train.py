import hashlib
import json
import pickle
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_monitoring_baseline(feature_df: pd.DataFrame, selected_features: list[str]) -> dict[str, object]:
    return {
        "features": selected_features,
        "row_count": int(feature_df.shape[0]),
        "summary": {
            feature: {
                "mean": float(feature_df[feature].mean()),
                "std": float(feature_df[feature].std(ddof=0)),
                "min": float(feature_df[feature].min()),
                "max": float(feature_df[feature].max()),
                "p50": float(feature_df[feature].median()),
                "p95": float(feature_df[feature].quantile(0.95)),
            }
            for feature in selected_features
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "server_metrics.csv"
    models_dir = project_root / "models"
    registry_dir = models_dir / "registry"
    models_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset_sha256 = compute_sha256(data_path)
    trained_at = datetime.now(timezone.utc)
    model_version = f"{trained_at.strftime('%Y%m%dT%H%M%SZ')}-{dataset_sha256[:8]}"
    version_dir = registry_dir / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    original_shape = df.shape

    timestamp_cols = [col for col in df.columns if "timestamp" in col.lower()]
    if timestamp_cols:
        df = df.drop(columns=timestamp_cols)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] < 5:
        raise ValueError("Need at least 5 numeric columns to build feature set.")

    random_seed = 42
    rng = np.random.default_rng(random_seed)
    shuffled_cols = numeric_df.columns.to_numpy().copy()
    rng.shuffle(shuffled_cols)
    selected_features = list(shuffled_cols[:5])

    feature_df = numeric_df[selected_features].apply(pd.to_numeric, errors="coerce")
    dropped_na_rows = int(feature_df.isna().any(axis=1).sum())
    feature_df = feature_df.dropna()
    if feature_df.empty:
        raise ValueError("No valid rows remain after numeric conversion and NaN removal.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    model = IsolationForest(random_state=random_seed, contamination="auto")
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    anomaly_ratio = float((predictions == -1).mean())
    decision_scores = model.decision_function(X_scaled)

    monitoring_baseline = build_monitoring_baseline(feature_df, selected_features)
    sample_row = feature_df.iloc[0].to_dict()
    sample_payload = {k: float(v) for k, v in sample_row.items()}

    version_model_path = version_dir / "model.pkl"
    version_scaler_path = version_dir / "scaler.pkl"
    version_baseline_path = version_dir / "monitoring_baseline.json"
    version_metadata_path = version_dir / "metadata.json"

    with version_model_path.open("wb") as f:
        pickle.dump(model, f)

    with version_scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    with version_baseline_path.open("w", encoding="utf-8") as f:
        json.dump(monitoring_baseline, f, indent=2)

    metadata = {
        "model_version": model_version,
        "created_at": trained_at.isoformat().replace("+00:00", "Z"),
        "dataset": {
            "path": data_path.relative_to(project_root).as_posix(),
            "sha256": dataset_sha256,
            "row_count": int(original_shape[0]),
            "column_count": int(original_shape[1]),
            "numeric_column_count": int(numeric_df.shape[1]),
            "rows_used_for_training": int(feature_df.shape[0]),
            "dropped_na_rows": dropped_na_rows,
            "removed_timestamp_columns": timestamp_cols,
        },
        "features": selected_features,
        "training": {
            "random_seed": random_seed,
            "feature_selection_strategy": "seeded_shuffle_take_first_5_numeric_columns",
            "model_type": "IsolationForest",
            "model_params": model.get_params(),
            "scaler_type": "StandardScaler",
            "library_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": __import__("sklearn").__version__,
            },
            "anomaly_ratio_on_training_data": anomaly_ratio,
            "decision_function_summary": {
                "min": float(decision_scores.min()),
                "max": float(decision_scores.max()),
                "mean": float(decision_scores.mean()),
                "p05": float(np.quantile(decision_scores, 0.05)),
                "p95": float(np.quantile(decision_scores, 0.95)),
            },
        },
        "artifacts": {
            "model": version_model_path.relative_to(project_root).as_posix(),
            "scaler": version_scaler_path.relative_to(project_root).as_posix(),
            "monitoring_baseline": version_baseline_path.relative_to(project_root).as_posix(),
        },
        "sample_payload": sample_payload,
    }

    with version_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    manifest = {
        "active_version": model_version,
        "updated_at": trained_at.isoformat().replace("+00:00", "Z"),
        "metadata_path": version_metadata_path.relative_to(project_root).as_posix(),
        "artifacts": metadata["artifacts"],
    }

    current_manifest_path = models_dir / "current_model.json"
    with current_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    shutil.copy2(version_model_path, models_dir / "isolation_forest.pkl")
    shutil.copy2(version_scaler_path, models_dir / "scaler.pkl")
    shutil.copy2(version_baseline_path, models_dir / "monitoring_baseline.json")

    print(
        json.dumps(
            {
                "model_version": model_version,
                "sample_payload": sample_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
