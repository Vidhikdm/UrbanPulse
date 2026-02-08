.PHONY: help install test format lint clean data-check smoke

help:
	@echo "UrbanPulse - Available Commands"
	@echo "================================"
	@echo ""
	@echo "Development:"
	@echo "  make format       - Auto-format code (black + isort)"
	@echo "  make lint         - Lint code (flake8)"
	@echo "  make test         - Run tests"
	@echo ""
	@echo "Data:"
	@echo "  make data-check   - Verify data availability for NYC"
	@echo "  make smoke        - Quick end-to-end test (geo-only, no downloads)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove caches and artifacts"

install:
	pip install -e .

test:
	pytest -q

format:
	black urbanpulse/ scripts/ experiments/ tests/ --line-length=100
	isort urbanpulse/ scripts/ experiments/ tests/ --profile black

lint:
	flake8 urbanpulse/ scripts/ experiments/ --max-line-length=100 --ignore=E203,W503

clean:
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete"

data-check:
	python scripts/check_data_ready.py --city nyc

smoke:
	python scripts/build_dataset.py --city nyc --limit 50 --geo-only --output data/processed/smoke_test.parquet
	python experiments/train_income.py --data data/processed/smoke_test.parquet --model ridge --geo-only --output outputs/smoke/
	@echo "✅ Smoke test complete"
