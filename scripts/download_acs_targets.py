#!/usr/bin/env python3
"""
Download tract-level ACS 5-year targets for a city:
- median_income
- poverty_rate
- unemployment_rate
- bachelors_plus_rate
- median_rent
- median_home_value

Uses Census API (ACS 5-year). Optional API key via env var CENSUS_API_KEY.
Writes: data/raw/census/{city}_targets_{year}.parquet

Cities supported: nyc, chicago, san_francisco
Year supported: 2021 (ACS 5-year) (works for other years if variables remain consistent)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import requests
import pandas as pd

CITY_FIPS = {
    # Tracts are statewide; we filter by county list.
    "nyc": {"state": "36", "counties": ["005", "047", "061", "081", "085"]},  # Bronx, Kings, New York, Queens, Richmond
    "chicago": {"state": "17", "counties": ["031"]},  # Cook
    "san_francisco": {"state": "06", "counties": ["075"]},  # San Francisco
}

# ACS variables (5-year)
VARS = {
    "median_income": "B19013_001E",

    # Poverty rate: below poverty / total (from B17001)
    "poverty_total": "B17001_001E",
    "poverty_below": "B17001_002E",

    # Unemployment rate: unemployed / labor force (from B23025)
    "labor_force": "B23025_003E",   # in labor force
    "unemployed": "B23025_005E",    # unemployed

    # Bachelor's+ rate: (B15003_022..025) / total (B15003_001)
    "edu_total": "B15003_001E",
    "edu_ba": "B15003_022E",
    "edu_ma": "B15003_023E",
    "edu_prof": "B15003_024E",
    "edu_phd": "B15003_025E",

    "median_rent": "B25064_001E",
    "median_home_value": "B25077_001E",
}

def fetch_county_tracts(year: int, state: str, county: str, api_key: str | None) -> pd.DataFrame:
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    get_vars = ["NAME"] + list(VARS.values())
    params = {
        "get": ",".join(get_vars),
        "for": "tract:*",
        "in": f"state:{state} county:{county}",
    }
    if api_key:
        params["key"] = api_key

    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    cols = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=cols)

    # tract_id = state+county+tract
    df["tract_id"] = df["state"].astype(str) + df["county"].astype(str) + df["tract"].astype(str)
    return df

def to_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(CITY_FIPS.keys()))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--output", default="", help="Override output parquet path")
    args = ap.parse_args()

    city = args.city
    year = args.year
    conf = CITY_FIPS[city]
    api_key = os.getenv("CENSUS_API_KEY")

    parts = []
    for county in conf["counties"]:
        parts.append(fetch_county_tracts(year, conf["state"], county, api_key))
    raw = pd.concat(parts, ignore_index=True)

    # Numeric conversion
    df = pd.DataFrame({
        "tract_id": raw["tract_id"].astype(str),
        "median_income": to_num(raw, VARS["median_income"]),
        "median_rent": to_num(raw, VARS["median_rent"]),
        "median_home_value": to_num(raw, VARS["median_home_value"]),
    })

    poverty_total = to_num(raw, VARS["poverty_total"])
    poverty_below = to_num(raw, VARS["poverty_below"])
    labor_force = to_num(raw, VARS["labor_force"])
    unemployed = to_num(raw, VARS["unemployed"])
    edu_total = to_num(raw, VARS["edu_total"])
    edu_plus = (
        to_num(raw, VARS["edu_ba"]) +
        to_num(raw, VARS["edu_ma"]) +
        to_num(raw, VARS["edu_prof"]) +
        to_num(raw, VARS["edu_phd"])
    )

    # Rates (safe divide)
    df["poverty_rate"] = (poverty_below / poverty_total).where(poverty_total > 0)
    df["unemployment_rate"] = (unemployed / labor_force).where(labor_force > 0)
    df["bachelors_plus_rate"] = (edu_plus / edu_total).where(edu_total > 0)

    # Clean
    for c in ["median_income", "median_rent", "median_home_value", "poverty_rate", "unemployment_rate", "bachelors_plus_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without a core label
    df = df.dropna(subset=["median_income"]).copy()

    out = Path(args.output) if args.output else Path(f"data/raw/census/{city}_targets_{year}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"✅ Wrote ACS targets: {out} (rows={len(df)}, cols={len(df.columns)})")
    print("   columns:", list(df.columns))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
