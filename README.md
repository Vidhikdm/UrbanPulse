# UrbanPulse — Real-Time Socioeconomic Sensing via Multimodal Urban Data

UrbanPulse is a reproducible research + engineering project that builds tract-level datasets from **open urban data** and evaluates how well different urban signals predict **median household income**. It supports multi-city pipelines and emphasizes **cross-city robustness** and **fairness diagnostics**.

> Status: runnable end-to-end (data → features → datasets → experiments). Active development.

---

## What this repo does

### Inputs (open data)
- **US Census TIGER/Line** tract boundaries (GeoPackage)
- **ACS 5-year** tract median income (variable `B19013_001E`)
- **OpenStreetMap (Geofabrik extracts)** for urban form / POIs / roads
- **NYC 311** complaints (optional NYC-only civic signal)

### Outputs
- Clean tract-level datasets in `data/processed/`
- Feature tables in `data/features/`
- Evaluation results in `outputs/results/`

---

## Repository structure (key paths)

- `configs/cities.yaml` — city definitions (FIPS, counties, bbox, geofabrik path)
- `scripts/` — data + feature + dataset builders
  - `fetch_census_tracts_multi.py`
  - `fetch_census_income_multi.py`
  - `compute_osm_features_from_pbf.py` (works for smaller PBFS; can be memory heavy for NYC)
  - `compute_osm_features_streaming.py` (NYC-safe, low-memory)
  - `build_dataset_multi_city.py` (tracts + ACS + OSM features → dataset)
- `experiments/`
  - `cross_city_eval.py` — cross-city evaluation matrix
  - `ablation_nyc.py` — NYC ablation (OSM vs OSM+311), log vs raw, ridge vs xgboost + fairness
  - `run_all.py` — orchestrates all experiments
  - `summarize_results.py` — prints README-friendly summaries
- `urbanpulse/evaluation/`
  - `metrics.py` — MAE/RMSE/R²/Spearman
  - `fairness.py` — error-by-income-quartile report

---

## Quickstart (run experiments now)

Assuming you already have the datasets in `data/processed/`:

```bash
python3 experiments/run_all.py
python3 experiments/summarize_results.py

