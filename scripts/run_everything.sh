#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source venv/bin/activate

YEAR="${1:-2021}"

python scripts/fetch_acs_targets.py --city nyc --year "$YEAR"
python scripts/fetch_acs_targets.py --city chicago --year "$YEAR"
python scripts/fetch_acs_targets.py --city san_francisco --year "$YEAR"

python scripts/build_dataset_multi_city.py --city nyc --year "$YEAR" --output data/processed/nyc_dataset.parquet
python scripts/build_dataset_multi_city.py --city chicago --year "$YEAR" --output data/processed/chicago_dataset.parquet
python scripts/build_dataset_multi_city.py --city san_francisco --year "$YEAR" --output data/processed/san_francisco_dataset.parquet

bash scripts/refresh_results.sh

python scripts/make_interactive_maps.py --city nyc --model xgboost --feature_set osm_only
python scripts/make_interactive_maps.py --city nyc --model xgboost --feature_set osm_311 || true

mkdir -p docs/figures
cp -f outputs/figures/*.png docs/figures/ 2>/dev/null || true

echo "✅ Done. Open:"
echo " - docs/maps/*.html"
echo " - docs/figures/*.png"
