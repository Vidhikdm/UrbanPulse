#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except Exception:
    xgb = None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_geo_features(df: pd.DataFrame) -> np.ndarray:
    cols = ["lat", "lon", "road_density", "poi_density", "landuse_entropy"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing geo columns: {missing}")
    return df[cols].to_numpy(dtype=float)


def train_ridge(X_train, y_train) -> Ridge:
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    if xgb is None:
        raise RuntimeError("xgboost not available. Use --model ridge.")
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train income prediction model (v1.0)")
    parser.add_argument("--data", required=True, help="Parquet dataset path")
    parser.add_argument("--model", choices=["ridge", "xgboost"], default="ridge")
    parser.add_argument("--geo-only", action="store_true", help="Use geo features only")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    out_root = Path(args.output)
    _ensure_dir(out_root / "models")
    _ensure_dir(out_root / "results")

    df = pd.read_parquet(args.data).dropna(subset=["median_income"]).reset_index(drop=True)

    X = get_geo_features(df)
    y = df["median_income"].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if args.model == "ridge":
        model = train_ridge(X_train, y_train)
        import pickle
        model_path = out_root / "models" / "income_ridge.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
    else:
        model = train_xgboost(X_train, y_train)
        model_path = out_root / "models" / "income_xgboost.json"
        model.save_model(model_path.as_posix())

    y_pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float((mean_squared_error(y_test, y_pred)) ** 0.5),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
        "model": args.model,
    }

    metrics_path = out_root / "results" / "income_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("✅ Saved:", metrics_path)
    print(f"MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | R2: {metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
