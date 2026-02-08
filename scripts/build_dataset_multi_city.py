#!/usr/bin/env python3
"""
Build tract-level datasets for multiple cities using:
- TIGER tract geometries (.gpkg)
- ACS median income (.parquet)
- OSM-derived tract features (.parquet)

Usage:
  python3 scripts/build_dataset_multi_city.py --city chicago --output data/processed/chicago_dataset.parquet
  python3 scripts/build_dataset_multi_city.py --city san_francisco --output data/processed/san_francisco_dataset.parquet
"""

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd


def load_tracts(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    # Ensure tract_id exists
    if "tract_id" not in gdf.columns:
        if "GEOID" in gdf.columns:
            gdf["tract_id"] = gdf["GEOID"].astype(str)
        else:
            raise SystemExit(f"❌ tract file missing GEOID/tract_id: {path}")
    gdf["tract_id"] = gdf["tract_id"].astype(str)
    return gdf


def load_income(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "tract_id" not in df.columns or "median_income" not in df.columns:
        raise SystemExit(f"❌ income parquet must have tract_id + median_income: {path}")
    df = df.copy()
    df["tract_id"] = df["tract_id"].astype(str)
    df["median_income"] = pd.to_numeric(df["median_income"], errors="coerce")
    df = df.dropna(subset=["median_income"])
    return df[["tract_id", "median_income"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=["nyc", "chicago", "san_francisco"])
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap after merge (0 = no cap)")
    args = ap.parse_args()

    city = args.city
    year = args.year

    tracts_path = Path(f"data/raw/census/{city}_tracts_{year}.gpkg")
    income_path = Path(f"data/raw/census/{city}_income_{year}.parquet")
    geo_path = Path(f"data/features/{city}_geo_features.parquet")

    for p in [tracts_path, income_path, geo_path]:
        if not p.exists():
            raise SystemExit(f"❌ Missing required input: {p}")

    tracts = load_tracts(tracts_path)
    income = load_income(income_path)
    geo = pd.read_parquet(geo_path).copy()
    if "tract_id" not in geo.columns:
        raise SystemExit(f"❌ geo features missing tract_id: {geo_path}")
    geo["tract_id"] = geo["tract_id"].astype(str)

    # Merge labels into tracts
    gdf = tracts.merge(income, on="tract_id", how="inner")
    # Centroids for lat/lon (safe in projected CRS? we'll do in WGS84)
    if gdf.crs is None:
        # TIGER usually ships with EPSG:4269; but if missing, still set to WGS84-ish
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    # Compute centroids in a projected CRS (avoids centroid-in-geographic warning)
    # Web Mercator is fine here for centroids; then convert centroid points back to WGS84.
    gdf_proj = gdf.to_crs("EPSG:3857")
    cent_proj = gdf_proj.geometry.centroid
    cent_ll = gpd.GeoSeries(cent_proj, crs="EPSG:3857").to_crs("EPSG:4326")
    gdf["lat"] = cent_ll.y
    gdf["lon"] = cent_ll.x

    # Merge geo features
    df = pd.DataFrame({
        "city": city,
        "tract_id": gdf_ll["tract_id"].astype(str),
        "median_income": gdf_ll["median_income"].astype(float),
        "lat": gdf_ll["lat"].astype(float),
        "lon": gdf_ll["lon"].astype(float),
        "image_paths": ["[]"] * len(gdf_ll),  # placeholder for later vision phase
    })
    df = df.merge(geo, on="tract_id", how="left").fillna(0)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"✅ Wrote dataset: {out_path} (rows={len(df)}, cols={len(df.columns)})")
    print(f"   income min/max: {df['median_income'].min():.0f} / {df['median_income'].max():.0f}")
    print(f"   unique tracts: {df['tract_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
