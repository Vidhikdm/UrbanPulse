#!/usr/bin/env python3
"""
Fetch census tract geometries for configured cities (TIGER/Line 2021, county-based).

Usage:
  python3 scripts/fetch_census_tracts_multi.py --cities nyc chicago sf
"""
import argparse
from pathlib import Path
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import yaml

TIGER_YEAR_DEFAULT = 2021

def load_cfg():
    with open("configs/cities.yaml", "r") as f:
        return yaml.safe_load(f)["cities"]

def download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--year", type=int, default=TIGER_YEAR_DEFAULT)
    args = ap.parse_args()

    cities = load_cfg()
    cache_dir = Path("data/raw/census/tiger_cache")
    out_dir = Path("data/raw/census")
    out_dir.mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        if city not in cities:
            raise SystemExit(f"Unknown city '{city}'. Check configs/cities.yaml")

        cfg = cities[city]
        st = cfg["state_fips"]
        counties = cfg["counties"]
        year = args.year

        print(f"\n=== {city.upper()} tracts (TIGER {year}) ===")
        gdfs = []

        for cty in counties:
            fname = f"tl_{year}_{st}{cty}_tract.zip"
            url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/{fname}"
            zpath = cache_dir / fname
            try:
                download(url, zpath)
                gdf = gpd.read_file(f"zip://{zpath}")
                gdfs.append(gdf)
                print(f"✓ {st}{cty}: {len(gdf)} tracts")
            except Exception as e:
                print(f"✗ Failed {fname}: {e}")

        if not gdfs:
            raise SystemExit(f"No tract files downloaded for {city}")

        merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
        # Standardize id column
        if "GEOID" not in merged.columns:
            raise SystemExit(f"{city}: GEOID column missing in TIGER file")
        merged["tract_id"] = merged["GEOID"].astype(str)

        out_path = out_dir / f"{city}_tracts_{year}.gpkg"
        merged.to_file(out_path, driver="GPKG")
        print(f"✅ Saved: {out_path} (rows={len(merged)})")

if __name__ == "__main__":
    main()
