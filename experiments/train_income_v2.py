#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


GEO_COLS = ["road_density", "poi_density", "landuse_entropy"]


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": rmse, "r2": float(r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["ridge", "xgboost"], default="xgboost")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--log-target", action="store_true", help="Train on log1p(income) then invert for metrics")
    args = ap.parse_args()

    out_dir = Path(args.output)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)

    # Optional: include NYC 311 features
    include_311_cols = []
    if getattr(args, "include_311", False):
        include_311_cols = [c for c in df.columns if c.startswith("complaint_")]
        for c in ("complaints_total_density", "complaint_entropy"):
            if c in df.columns:
                include_311_cols.append(c)
        # Fill missing 311 features with 0
        if include_311_cols:
            df[include_311_cols] = df[include_311_cols].fillna(0.0)
            print(f"✅ Using {len(include_311_cols)} NYC 311 feature columns")
        else:
            print("⚠️  --include-311 set, but no 311 columns found in dataset")

    # Optional: include NYC 311 tract features if present
    extra_311_cols = []
    if getattr(args, "include_311", False):
        for c in df.columns:
            if c == "complaints_total_density" or c == "complaint_entropy" or c.startswith("complaint_"):
                extra_311_cols.append(c)
        # Keep only numeric 311 cols (defensive)
        extra_311_cols = [c for c in extra_311_cols if c != "tract_id"]
        if extra_311_cols:
            print(f"✅ Using {len(extra_311_cols)} NYC 311 feature columns")
        else:
            print("ℹ️  --include-311 set, but no 311 feature columns found in dataset")
    missing = [c for c in GEO_COLS + ["median_income"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # Clean & stabilize: clip extreme densities and add log features (helps tree + linear)
    # Build feature list (geo + optional 311)
    base_features = [c for c in df.columns if c in ("road_density","poi_density","landuse_entropy")]
    features = features + include_311_cols
    features = base_features + extra_311_cols
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing feature columns: {missing}")

    X = df[features].values
    y = df["median_income"].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    if args.model == "ridge":
        base = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=10.0, random_state=args.seed))
        ])
    else:
        if XGBRegressor is None:
            raise RuntimeError("xgboost not installed. pip install xgboost")
        base = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=3,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=4,
        )

    model = base
    if args.log_target:
        # Train on log1p(y), predict back on original scale for metrics
        model = TransformedTargetRegressor(
            regressor=base,
            func=np.log1p,
            inverse_func=np.expm1
        )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    m = metrics(y_test, preds)

    # Save
    out = {
        "model": args.model,
        "log_target": bool(args.log_target),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        **m,
    }
    (out_dir / "results" / "income_metrics.json").write_text(json.dumps(out, indent=2))
    print("✅ Saved:", out_dir / "results" / "income_metrics.json")
    print(f"MAE: {m['mae']:.2f} | RMSE: {m['rmse']:.2f} | R2: {m['r2']:.3f}")


if __name__ == "__main__":
    main()
