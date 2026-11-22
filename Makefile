.PHONY: all test lint typecheck train-smoke eval api web clean

PYTHON ?= python

all: lint test train-smoke eval

lint:
	$(PYTHON) -m ruff check src tests api

typecheck:
	$(PYTHON) -m mypy src tests

test: lint typecheck
	$(PYTHON) -m pytest tests/

train-smoke:
	$(PYTHON) -m eatsched.train --mode paper --budget smoke --cluster edge32 --seed 42

eval:
	$(PYTHON) -m eatsched.evaluate --mode paper --cluster edge32 --episodes 5 --seed 42

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm run dev

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
