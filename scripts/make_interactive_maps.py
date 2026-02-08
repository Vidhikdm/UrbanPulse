#!/usr/bin/env python3
from __future__ import annotations


import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import MarkerCluster
except Exception as e:
    raise SystemExit("❌ Missing folium. Install with: python -m pip install folium branca") from e


def _safe_json_list(s: str):
    try:
        x = json.loads(s)
        return x if isinstance(x, list) else []
    except Exception:
        return []


def load_dataset(city: str) -> pd.DataFrame:
    p = Path(f"data/processed/{city}_dataset.parquet")
    if not p.exists():
        raise SystemExit(f"❌ Missing dataset: {p}. Build it with scripts/build_dataset_multi_city.py")
    df = pd.read_parquet(p)
    needed = {"tract_id", "lat", "lon", "median_income"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"❌ Dataset missing columns: {missing}")
    return df


def predict_income(df: pd.DataFrame, model: str, feature_set: str) -> np.ndarray:
    # Reuse your project's train_predict helper via a local import that works when run from repo root
    from experiments.cross_city_eval import train_predict  # type: ignore

    # Feature selection aligned with your experiment logic:
    # - osm_only: all non-target numeric features (excluding coords/id/city)
    # - osm_311: if present, keep only a subset that includes 311-derived cols (heuristic)
    drop_cols = {"city", "tract_id", "image_paths"}
    target_cols = {
        "median_income", "median_rent", "median_home_value",
        "poverty_rate", "unemployment_rate", "bachelors_plus_rate"
    }

    feat_cols = [c for c in df.columns if c not in drop_cols and c not in target_cols]
    # remove obvious non-features
    feat_cols = [c for c in feat_cols if c not in {"lat", "lon"}]

    if feature_set == "osm_311":
        # Keep columns that look like 311 + a few core OSM/geo ones if they exist
        keep = []
        for c in feat_cols:
            lc = c.lower()
            if "311" in lc or "complaint" in lc or "service" in lc:
                keep.append(c)
        # Fallback: if no 311-like columns exist, just use all features
        if len(keep) >= 3:
            feat_cols = keep

    if not feat_cols:
        raise SystemExit("❌ No feature columns found. Check your dataset columns.")

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["median_income"].to_numpy(dtype=float)

    # Map is for NYC only; do simple split inside city for demo visualization
    rng = np.random.default_rng(42)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_train = int(0.8 * len(df))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te = X  # predict for all rows so we can map everything

    # Match your best practice: log1p target, then expm1 back, with clipping in log-space
    y_tr_fit = np.log1p(y_tr)
    y_pred_fit = train_predict(X_tr, y_tr_fit, X_te, model_name=model, seed=42)

    # clip to training log-range before inverse
    lo = float(np.min(y_tr_fit))
    hi = float(np.max(y_tr_fit))
    y_pred_fit = np.clip(y_pred_fit, lo, hi)
    y_pred = np.expm1(y_pred_fit)
    return y_pred


def make_map(df: pd.DataFrame, y_pred: np.ndarray, out_html: Path, title: str):
    df = df.copy()
    df["pred_income"] = y_pred
    df["abs_err"] = np.abs(df["pred_income"] - df["median_income"])

    center = [float(df["lat"].median()), float(df["lon"].median())]
    m = folium.Map(location=center, zoom_start=10, tiles="cartodbpositron")

    folium.map.CustomPane("labels").add_to(m)

    # Choropleth-like effect via circle markers (fast, dependency-light)
    # radius scaled by error; color scaled by predicted income quantiles
    q = np.quantile(df["pred_income"], [0.1, 0.3, 0.5, 0.7, 0.9])

    def color(v: float) -> str:
        if v <= q[0]: return "#2c7bb6"
        if v <= q[1]: return "#00a6ca"
        if v <= q[2]: return "#00ccbc"
        if v <= q[3]: return "#90eb9d"
        if v <= q[4]: return "#f9d057"
        return "#d7191c"

    cluster = MarkerCluster(name="Tracts").add_to(m)

    for _, r in df.iterrows():
        popup = folium.Popup(
            f"<b>tract_id</b>: {r['tract_id']}<br>"
            f"<b>true income</b>: ${r['median_income']:.0f}<br>"
            f"<b>pred income</b>: ${r['pred_income']:.0f}<br>"
            f"<b>abs err</b>: ${r['abs_err']:.0f}",
            max_width=320,
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=float(np.clip(3 + (r["abs_err"] / 20000.0), 3, 12)),
            color=color(float(r["pred_income"])),
            fill=True,
            fill_opacity=0.65,
            weight=1,
            popup=popup,
        ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)
    folium.map.Marker(
        center,
        icon=folium.DivIcon(
            html=f"<div style='font-size:16px;font-weight:600'>{title}</div>"
        ),
    ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"✅ Wrote interactive map: {out_html}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["nyc", "chicago", "san_francisco"])
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "ridge"])
    ap.add_argument("--feature_set", default="osm_only", choices=["osm_only", "osm_311"])
    args = ap.parse_args()

    df = load_dataset(args.city)
    y_pred = predict_income(df, model=args.model, feature_set=args.feature_set)

    out = Path(f"docs/maps/{args.city}_{args.model}_{args.feature_set}_income_map.html")
    title = f"{args.city.upper()} income map | {args.model} | {args.feature_set}"
    make_map(df, y_pred, out, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
