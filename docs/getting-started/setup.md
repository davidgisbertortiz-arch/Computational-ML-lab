# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Docker for containerized experiments

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/davidgisbertortiz-arch/Computational-ML-lab.git
cd Computational-ML-lab
```

### 2. Create Virtual Environment

=== "venv"
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

=== "conda"
    ```bash
    conda create -n ml-lab python=3.11
    conda activate ml-lab
    ```

### 3. Install Dependencies

```bash
# Quick setup with Make
make setup

# Or manually
pip install -e ".[dev,docs]"
pre-commit install
```

### 4. Verify Installation

```bash
# Run tests
make test

# Check linting
make lint
```

## Development Workflow

### Running Experiments

Each module has a CLI entry point:

```bash
# General pattern
python -m modules.<module_name>.src.main --config configs/experiment.yaml --seed 42

# Example
python -m modules.01_numerical_toolbox.src.main --help
```

### Running Tests

```bash
# All tests
make test

# Specific module
make test-module MODULE=01_numerical_toolbox

# Fast tests (no coverage)
make test-fast
```

### Code Quality

```bash
# Auto-format
make format

# Check linting
make lint

# Run pre-commit hooks
make pre-commit
```

### Documentation

```bash
# Serve docs locally
make serve-docs
# Open http://localhost:8000

# Build docs
make docs
```

## MLflow Tracking

MLflow is **optional** for experiment tracking. The local database (`mlflow.db`) and run artifacts (`mlruns/`) are generated at runtime and not committed to version control.

View experiment results:

```bash
make mlflow-ui
# Open http://localhost:5000
```

**Note**: 
- If MLflow is not configured, experiments will still run normally without tracking
- Enable tracking per experiment via `config.mlflow_tracking = True` in YAML configs
- Local artifacts can be cleaned with `make clean`
- The system uses a local SQLite backend by default (no server setup required)

## Troubleshooting

### Import Errors

Ensure you installed the package in editable mode:
```bash
pip install -e .
```

### Pre-commit Hook Failures

Update hooks:
```bash
pre-commit autoupdate
pre-commit run --all-files
```

### Test Failures

Check Python version (must be 3.11+):
```bash
python --version
```

## Next Steps

- Read the [Module Structure](module-structure.md) guide
- Check out [Module 00: Repo Standards](../../modules/00_repo_standards/README.md)
- Explore the [Contributing Guide](../contributing.md)
