#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
if [ ! -f "README.md" ]; then
  echo "❌ Run from repo root (README.md not found)."
  exit 1
fi

source venv/bin/activate

echo "==> Running experiments..."
python experiments/run_all.py
python experiments/summarize_results.py

echo "==> Generating markdown + figures..."
python scripts/generate_results_md.py
python scripts/make_figures.py

echo "✅ Done. See:"
echo " - outputs/results/RESULTS.md"
echo " - outputs/figures/"
