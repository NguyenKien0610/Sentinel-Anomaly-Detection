import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "server_metrics.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Drop timestamp-like columns if present.
    timestamp_cols = [col for col in df.columns if "timestamp" in col.lower()]
    if timestamp_cols:
        df = df.drop(columns=timestamp_cols)

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] < 5:
        raise ValueError("Need at least 5 numeric columns to build feature set.")

    rng = np.random.default_rng(42)
    shuffled_cols = numeric_df.columns.to_numpy().copy()
    rng.shuffle(shuffled_cols)
    selected_features = list(shuffled_cols[:5])

    feature_df = numeric_df[selected_features].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.dropna()

    if feature_df.empty:
        raise ValueError("No valid rows remain after numeric conversion and NaN removal.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    model = IsolationForest(random_state=42, contamination="auto")
    model.fit(X_scaled)

    model_path = models_dir / "isolation_forest.pkl"
    scaler_path = models_dir / "scaler.pkl"

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)

    sample_row = feature_df.iloc[0].to_dict()
    sample_payload = {k: float(v) for k, v in sample_row.items()}

    print(json.dumps(sample_payload, indent=2))


if __name__ == "__main__":
    main()
