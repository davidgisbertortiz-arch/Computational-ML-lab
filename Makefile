.PHONY: help setup install install-dev test lint format clean docs serve-docs run-module

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Initial setup: install dependencies and pre-commit hooks
	pip install -e ".[dev,docs]"
	pre-commit install
	@echo "✅ Setup complete! Run 'make test' to verify."

install:  ## Install core dependencies only
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev,docs]"

test:  ## Run all tests with coverage
	pytest --cov=modules --cov-report=term-missing --cov-report=html

test-fast:  ## Run tests without coverage (faster)
	pytest -x

test-module:  ## Run tests for a specific module (usage: make test-module MODULE=01_numerical_toolbox)
	pytest modules/$(MODULE)/tests/ -v

lint:  ## Run all linters (ruff, black check, mypy)
	ruff check .
	black --check .
	mypy modules/ --ignore-missing-imports || true

format:  ## Auto-format code with black and ruff
	ruff check --fix .
	black .

clean:  ## Remove cache, build artifacts, and coverage reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage

docs:  ## Build documentation
	mkdocs build

serve-docs:  ## Serve documentation locally
	mkdocs serve

run-module:  ## Run a module's main entry point (usage: make run-module MODULE=01_numerical_toolbox ARGS="--help")
	python -m modules.$(MODULE).src.main $(ARGS)

mlflow-ui:  ## Start MLflow UI to view experiment tracking
	mlflow ui --backend-store-uri file:./mlruns

docker-build:  ## Build Docker image for module 10 (MLOps)
	docker build -t computational-ml-lab:latest -f modules/10_mlops_production/Dockerfile .

docker-run:  ## Run Docker container
	docker run -p 8000:8000 computational-ml-lab:latest

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

update-deps:  ## Update all dependencies
	pip install --upgrade pip
	pip install --upgrade -e ".[dev,docs]"
