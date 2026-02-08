#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

def run(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    # Cross-city OSM baseline (already implemented in cross_city_eval.py)
    run(["python3", "experiments/cross_city_eval.py", "--full-matrix", "--model", "ridge"])
    run(["python3", "experiments/cross_city_eval.py", "--full-matrix", "--model", "xgboost"])

    # NYC ablation (OSM vs OSM+311, ridge vs xgb, raw vs log, fairness reports)
    run(["python3", "experiments/ablation_nyc.py"])

    print("\n✅ All experiments finished. Check outputs/results/")

if __name__ == "__main__":
    main()
