# Module 00: Repository Standards

**Status**: ✅ Complete

Establishes the "how we work" conventions for the entire repository. This module provides:
- Core utility package (`mlphys_core`) for all modules
- Repo-wide patterns for experiments, configs, testing
- Module template generator for consistent structure
- Working demo showing all conventions in action

---

## 📚 What You'll Learn

- **Reproducibility**: Seeding, config management, experiment tracking
- **Code Quality**: Linting, formatting, testing, type hints
- **Experiment Structure**: YAML configs, artifact management, deterministic outputs
- **Module Organization**: Consistent structure for all learning modules
- **Workflow**: How to run experiments, tests, and add new modules

---

## 🏗️ Repository Conventions

### Module Structure

Every module follows this structure:

```
modules/XX_module_name/
├── README.md              # Learning objectives, theory, experiments
├── src/                   # Importable Python package
│   ├── __init__.py
│   └── ...                # Module-specific code
├── tests/                 # Unit + integration tests
│   ├── test_*.py
│   └── ...
├── notebooks/             # Jupyter notebooks for exploration
│   └── *.ipynb
├── configs/               # YAML experiment configurations
│   └── *.yaml
└── reports/               # Generated outputs (gitignored)
    ├── figures/
    ├── metrics.json
    └── *.log
```

### Experiment Workflow

1. **Configuration**: Define experiments in YAML files
2. **Seeding**: Use `set_seed(seed)` for reproducibility
3. **Logging**: Use structured logging with `mlphys_core.setup_logger`
4. **Artifacts**: Save outputs to `reports/` directory
5. **Tracking**: Log metrics with git commit hash + config

### Output Management

- **Source code** (`src/`, `tests/`): Committed to git
- **Configs** (`configs/*.yaml`): Committed to git
- **Outputs** (`reports/`): Gitignored (except small examples)
- **Data** (`data/`): Gitignored; provide download scripts

### Reproducibility Requirements

Every experiment must:
1. Accept a `--seed` parameter
2. Log the git commit hash
3. Save the config used
4. Produce identical outputs with same seed
5. Log all metrics to JSON/MLflow

---

## 🧰 Core Utilities: `mlphys_core`

Located in `modules/00_repo_standards/src/mlphys_core/`, this package provides foundational utilities used across all modules.

### Configuration (`config.py`)

```python
from modules.00_repo_standards.src.mlphys_core import ExperimentConfig, load_config

# Define config schema
class MyConfig(ExperimentConfig):
    learning_rate: float = 0.01
    n_epochs: int = 100

# Load from YAML
config = load_config(Path("configs/experiment.yaml"), MyConfig)
```

**Key features**:
- Pydantic validation
- Type safety
- Automatic path handling
- Easy serialization

### Seeding (`seeding.py`)

```python
from modules.00_repo_standards.src.mlphys_core import set_seed, get_rng

# Set global seed (Python, NumPy, PyTorch)
set_seed(42)

# Get isolated RNG (preferred)
rng = get_rng(42)
data = rng.standard_normal(100)
```

### Logging (`logging_utils.py`)

```python
from modules.00_repo_standards.src.mlphys_core import setup_logger, log_metrics

# Setup logger
logger = setup_logger("my_exp", level="INFO", log_file=Path("exp.log"))
logger.info("Starting...")

# Log metrics
metrics = {"accuracy": 0.95, "loss": 0.15}
log_metrics(metrics, output_path=Path("reports/metrics.json"))
```

### Experiment Runner (`experiment.py`)

```python
from modules.00_repo_standards.src.mlphys_core import BaseExperiment

class MyExperiment(BaseExperiment):
    def prepare_data(self):
        # Load/generate data
        return X, y
    
    def build_model(self):
        # Create model
        return LinearRegression()
    
    def train(self, model, data):
        # Train model
        model.fit(*data)
        return model
    
    def evaluate(self, model, data):
        # Return metrics
        return {"rmse": 0.15}

# Run experiment
config = ExperimentConfig(name="test", seed=42)
exp = MyExperiment(config)
metrics = exp.run()
```

---

## 🚀 Quick Start

### Run the Demo

```bash
# Basic demo
python -m modules.00_repo_standards.run_demo --seed 42

# Custom config
python -m modules.00_repo_standards.run_demo \
  --config modules/00_repo_standards/configs/demo.yaml \
  --seed 123

# Help
python -m modules.00_repo_standards.run_demo --help
```

The demo:
- Generates synthetic classification data
- Trains a sklearn LogisticRegression model
- Logs metrics (accuracy, f1_score)
- Saves a confusion matrix plot to `reports/`
- Is fully deterministic (same seed → same outputs)

### Run Tests

```bash
# All module 00 tests
pytest modules/00_repo_standards/tests/ -v

# Specific test file
pytest modules/00_repo_standards/tests/test_seeding.py -v

# With coverage
pytest modules/00_repo_standards/tests/ --cov=modules.00_repo_standards.src
```

### Lint and Format

```bash
# Check code quality
make lint

# Auto-fix issues
make format

# Run pre-commit hooks
pre-commit run --all-files
```

---

## 🆕 Adding a New Module

Use the module template generator:

```bash
python modules/00_repo_standards/create_module.py \
  --name "05_simulation_monte_carlo" \
  --description "Monte Carlo methods and particle filters"
```

This creates:
- Complete directory structure
- Template README with sections
- Placeholder `src/__init__.py`
- Example test file
- Sample config file

Then:
1. Fill in the README with learning objectives and theory
2. Implement algorithms in `src/`
3. Add tests in `tests/`
4. Create experiment configs
5. Run experiments and save to `reports/`

---

## 🧪 Demo Experiment Details

**Task**: Binary classification on synthetic data

**Model**: Scikit-learn LogisticRegression

**Metrics**:
- Accuracy
- F1 Score
- Confusion Matrix (saved as PNG)

**Artifacts** (in `reports/`):
- `metrics.json`: Numerical results with timestamp
- `confusion_matrix.png`: Visualization
- `demo_experiment.log`: Execution log
- `config.yaml`: Config used

**Reproducibility**: Running with `--seed 42` twice produces:
- Identical train/test splits
- Identical model parameters
- Identical metric values
- Identical confusion matrix

---

## 🧪 Testing Patterns

### Unit Tests

Test individual functions:

```python
def test_set_seed_reproducibility():
    """Test that set_seed produces identical results."""
    set_seed(42)
    x1 = np.random.randn(10)
    
    set_seed(42)
    x2 = np.random.randn(10)
    
    assert np.array_equal(x1, x2)
```

### Integration Tests

Test full pipelines:

```python
def test_experiment_reproducibility():
    """Test that experiment gives same metrics with same seed."""
    config = DemoConfig(name="test", seed=42)
    
    exp1 = DemoExperiment(config)
    metrics1 = exp1.run()
    
    exp2 = DemoExperiment(config)
    metrics2 = exp2.run()
    
    assert metrics1 == metrics2
```

### Determinism Tests

Explicitly test reproducibility:

```python
from modules.00_repo_standards.src.mlphys_core import check_determinism

def my_experiment():
    set_seed(42)
    return np.random.randn(10).tolist()

assert check_determinism(my_experiment, seed=42, n_runs=3)
```

---

## ⚙️ Configuration Format

Example `configs/demo.yaml`:

```yaml
name: "demo_experiment"
description: "Classification demo with sklearn"
seed: 42

# Output settings
output_dir: "modules/00_repo_standards/reports"
save_artifacts: true
log_level: "INFO"

# Data generation
n_samples: 500
n_features: 20
test_size: 0.2

# Model hyperparameters
C: 1.0
max_iter: 1000
```

Config is loaded and validated with Pydantic:

```python
class DemoConfig(ExperimentConfig):
    n_samples: int = 500
    n_features: int = 20
    test_size: float = 0.2
    C: float = 1.0
    max_iter: int = 1000

config = load_config(Path("configs/demo.yaml"), DemoConfig)
```

---

## 📊 Metrics and Artifacts

### Metrics Format

`reports/metrics.json`:

```json
{
  "timestamp": "2025-12-31T10:30:00",
  "metrics": {
    "accuracy": 0.95,
    "f1_score": 0.948,
    "n_train": 400,
    "n_test": 100
  }
}
```

### Plots

- Save to `reports/figures/` 
- Use descriptive names: `confusion_matrix.png`, `learning_curve.png`
- Include in `.gitignore` (except small examples)

### Logs

- Saved to `reports/*.log`
- Include timestamps, log level, messages
- Useful for debugging and audit trails

---

## 🔧 Workflow Commands

```bash
# Setup
make setup                    # Install deps + pre-commit

# Testing
make test                     # All tests with coverage
make test-module MODULE=00_repo_standards

# Code quality
make lint                     # Check style
make format                   # Auto-fix

# Experiments
python -m modules.XX_module.src.main --config configs/exp.yaml --seed 42

# Create new module
python modules/00_repo_standards/create_module.py --name XX_new_module
```

---

## ✅ Definition of Done

- [x] `mlphys_core` package with config, seeding, logging, experiment runner
- [x] Demo CLI with sklearn model
- [x] Comprehensive tests (unit + integration + determinism)
- [x] README documenting all conventions
- [x] Module template generator script
- [x] All lint checks pass
- [x] All tests pass
- [x] Demo produces deterministic outputs

---

## 🔗 Key Files

- [mlphys_core/__init__.py](src/mlphys_core/__init__.py) - Package exports
- [mlphys_core/config.py](src/mlphys_core/config.py) - Pydantic config management
- [mlphys_core/seeding.py](src/mlphys_core/seeding.py) - Reproducibility utilities
- [mlphys_core/logging_utils.py](src/mlphys_core/logging_utils.py) - Logging setup
- [mlphys_core/experiment.py](src/mlphys_core/experiment.py) - Base experiment class
- [run_demo.py](run_demo.py) - CLI demo experiment
- [create_module.py](create_module.py) - Template generator

---

## 🚀 Next Steps

1. Run the demo: `python -m modules.00_repo_standards.run_demo --seed 42`
2. Examine outputs in `reports/`
3. Read the source code in `src/mlphys_core/`
4. Run tests: `pytest modules/00_repo_standards/tests/ -v`
5. Create a new module using the template generator
6. Move to [Module 01: Numerical Toolbox](../01_numerical_toolbox/README.md)
