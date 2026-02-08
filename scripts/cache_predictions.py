from __future__ import annotations


import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.model_zoo import get_model

TARGETS = [
    "median_income",
    "median_rent",
    "median_home_value",
    "poverty_rate",
    "unemployment_rate",
    "bachelors_plus_rate",
]

BASE_COLS = {"city", "tract_id", "lat", "lon", "image_paths"}

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = pd.Series(a).rank().to_numpy()
    b = pd.Series(b).rank().to_numpy()
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def load_dataset(city: str) -> pd.DataFrame:
    p = Path(f"data/processed/{city}_dataset.parquet")
    if not p.exists():
        raise SystemExit(f"❌ Missing dataset: {p}")
    return pd.read_parquet(p)

def select_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    targets = set(TARGETS)
    feat_cols = [c for c in df.columns if c not in BASE_COLS.union(targets)]

    if feature_set == "osm_only":
        return feat_cols

    if feature_set == "osm_311":
        keep = []
        for c in feat_cols:
            lc = c.lower()
            if ("311" in lc) or ("complaint" in lc) or ("osm" in lc) or ("poi" in lc) or ("road" in lc) or ("landuse" in lc):
                keep.append(c)
        return keep if len(keep) >= 5 else feat_cols

    raise SystemExit(f"❌ Unknown feature_set: {feature_set}")

def clean_xy(X: pd.DataFrame, y: pd.Series):
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pd.to_numeric(y, errors="coerce")
    ok = y.notna()
    return X.loc[ok].astype(float), y.loc[ok].astype(float)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["nyc", "chicago", "san_francisco"])
    ap.add_argument("--models", default="xgboost", help="Comma list: ridge,xgboost,extra_trees,random_forest,hist_gb,mlp,elasticnet")
    ap.add_argument("--feature_sets", default="osm_only", help="Comma list: osm_only,osm_311")
    ap.add_argument("--targets", default="all", help="Comma list of targets or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    feature_sets = [f.strip() for f in args.feature_sets.split(",") if f.strip()]

    if args.targets.strip().lower() == "all":
        targets = TARGETS
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    df = load_dataset(args.city)

    out_dir = Path("outputs/preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for feature_set in feature_sets:
        if feature_set == "osm_311" and args.city != "nyc":
            continue

        feat_cols = select_features(df, feature_set)
        X = df[feat_cols].copy()

        for target in targets:
            if target not in df.columns:
                continue

            y = df[target].copy()
            Xc, yc = clean_xy(X, y)

            X_tr, X_te, y_tr, y_te = train_test_split(
                Xc, yc, test_size=args.test_size, random_state=args.seed
            )

            for model_name in models:
                model = get_model(model_name, seed=args.seed)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                mae = float(mean_absolute_error(y_te, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                r2 = float(r2_score(y_te, y_pred))
                sp = spearman(y_te.to_numpy(), np.asarray(y_pred))

                idx = y_te.index
                out = df.loc[idx, ["tract_id", "lat", "lon"]].copy()
                out["y_true"] = y_te.to_numpy()
                out["y_pred"] = np.asarray(y_pred)
                out["error"] = out["y_pred"] - out["y_true"]
                out["city"] = args.city
                out["model"] = model_name
                out["feature_set"] = feature_set
                out["target"] = target

                fname = f"{args.city}__{model_name}__{feature_set}__{target}.parquet"
                path = out_dir / fname
                out.to_parquet(path, index=False)

                row = {
                    "city": args.city,
                    "model": model_name,
                    "feature_set": feature_set,
                    "target": target,
                    "path": str(path),
                    "n": int(len(out)),
                    "n_features": int(len(feat_cols)),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "spearman": sp,
                }
                manifest_rows.append(row)

                (out_dir / f"{args.city}__{model_name}__{feature_set}__{target}.metrics.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )

                print(f"✅ cached: {fname} | R2={r2:.3f} MAE={mae:.0f} Spearman={sp:.3f}")

    manifest = pd.DataFrame(manifest_rows)
    mfile = out_dir / "manifest.csv"
    # If a manifest already exists, append + dedupe (so multi-city caching works)
    if mfile.exists():
        try:
            old = pd.read_csv(mfile)
            manifest = pd.concat([old, manifest], ignore_index=True)
        except Exception:
            pass
    # Deduplicate by (city, model, feature_set, target)
    if len(manifest) > 0:
        manifest = manifest.drop_duplicates(subset=["city","model","feature_set","target"], keep="last")
    manifest.to_csv(mfile, index=False)

    manifest.to_csv(mfile, index=False)
    print(f"\n✅ Wrote manifest: {mfile} (rows={len(manifest)})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
