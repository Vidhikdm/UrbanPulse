#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def build_features(df: pd.DataFrame, include_311: bool) -> pd.DataFrame:
    # Core geo features (must exist)
    base = ["road_density", "poi_density", "landuse_entropy"]
    missing = [c for c in base if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing geo columns: {missing}")

    X = df[base].copy()

    # Simple, stable feature engineering
    eps = 1e-12
    X["log_road_density"] = np.log1p(np.maximum(X["road_density"].values, 0.0))
    X["log_poi_density"] = np.log1p(np.maximum(X["poi_density"].values, 0.0))
    X["interaction_road_poi"] = X["road_density"].values * X["poi_density"].values
    X["entropy_sq"] = X["landuse_entropy"].values ** 2

    # Optional 311 features (only if flag enabled AND columns exist)
    if include_311:
        cols_311 = [c for c in df.columns if c.startswith("complaint_") and c.endswith("_density")]
        extra = []
        if "complaints_total_density" in df.columns:
            extra.append("complaints_total_density")
        if "complaint_entropy" in df.columns:
            extra.append("complaint_entropy")

        use = cols_311 + extra
        if use:
            X = pd.concat([X, df[use].fillna(0.0)], axis=1)

    # Replace any remaining NaNs/Infs defensively
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def train_model(X_train: np.ndarray, y_train: np.ndarray, model_name: str, seed: int):
    if model_name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)

    if model_name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
        return XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
        )

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["ridge", "xgboost"], default="ridge")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--log-target", action="store_true", help="Train on log1p(income) then invert for metrics")
    ap.add_argument("--include-311", action="store_true", help="Include NYC 311 tract features if present")
    args = ap.parse_args()

    out_dir = Path(args.output)
    _ensure_dir(out_dir / "results")

    df = pd.read_parquet(args.data)
    if "median_income" not in df.columns:
        raise ValueError("Dataset must contain 'median_income'")

    # Build X/y
    Xdf = build_features(df, include_311=bool(args.include_311))
    y = df["median_income"].astype(float).values

    # Log-transform target if requested (common for income)
    if args.log_target:
        y_train_target = np.log1p(y)
    else:
        y_train_target = y

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        Xdf.values,
        y_train_target,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = train_model(X_train, y_train, args.model, seed=args.seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    if args.log_target:
        preds = np.expm1(preds)
        y_eval = y_test_raw
    else:
        y_eval = y_test_raw

    m = metrics(y_eval, preds)

    out = {
        "model": args.model,
        "include_311": bool(args.include_311),
        "log_target": bool(args.log_target),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        **m,
    }
    (out_dir / "results" / "income_metrics.json").write_text(json.dumps(out, indent=2))
    print("✅ Saved:", out_dir / "results" / "income_metrics.json")
    print(f"MAE: {m['mae']:.2f} | RMSE: {m['rmse']:.2f} | R2: {m['r2']:.3f}")
    print(f"Features used: {out['n_features']} (include_311={out['include_311']})")


if __name__ == "__main__":
    main()
