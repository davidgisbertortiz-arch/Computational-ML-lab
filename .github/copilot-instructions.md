# Copilot Instructions for Computational ML Lab

> **TL;DR**: Physics/maths-driven ML monorepo with numbered modules (00-11). Use `safe_import_from()` for all imports due to Python 3.12+ limitation. Run `make test` before commits. All experiments need `set_seed()` and config files.

## Quick Reference Card

### Essential Commands
```bash
make test                              # Run all tests with coverage (>80% target)
make test-module MODULE=07_physics_informed_ml  # Test specific module
make test-fast                         # Quick test (stop on first failure)
make lint && make format               # Check and fix code quality (Ruff + Black + MyPy)
make run-module MODULE=XX_name ARGS="..." # Run module CLI entry point (note: use module folder name)
make mlflow-ui                         # Launch MLflow tracking UI (port 5000)
make setup                             # First-time setup: install deps + pre-commit hooks
python modules/00_repo_standards/create_module.py --number XX --title "Name"  # Scaffold new module
```

**Command gotchas**:
- Module names for `make run-module` use folder names (e.g., `MODULE=06_deep_learning_systems`)
- Always run `make` commands from repo root
- MLflow UI runs on port 5000 (http://localhost:5000) - don't forget to open in browser

### Critical Import Pattern (Python 3.12+ Workaround)
```python
# ✅ ALWAYS use this - standard imports WILL fail
from modules._import_helper import safe_import_from
set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
MyClass = safe_import_from('XX_module.src.file', 'MyClass')

# ❌ NEVER use this - will cause SyntaxError in Python 3.12+
from modules.07_physics_informed_ml.src import PINN  # SyntaxError!
```

### Module Progression & Status
- ✅ **00** (repo standards) → **01** (numerical) → **02** (stats/UQ) → **03** (tabular ML) → **04** (time series) → **05** (Monte Carlo) → **06** (deep learning) → **07** (physics-informed)
- 🚧 **08** (NLP/RAG) → **09** (vision) → **10** (MLOps) → **11** (capstones) - Framework exists, implementations in progress

### File Locations Cheat Sheet
- **Core utilities**: `modules/00_repo_standards/src/mlphys_core/` (config, seeding, logging, BaseExperiment)
- **Module scaffold**: `python modules/00_repo_standards/create_module.py`
- **Test configs**: `modules/XX_name/tests/conftest.py` (module-specific fixtures)
- **Experiment configs**: `modules/XX_name/configs/*.yaml` (Pydantic-validated)
- **Notebooks**: `modules/XX_name/notebooks/NN_topic.ipynb` (numbered sequentially)
- **Reports** (gitignored): `modules/XX_name/reports/` (plots, logs, metrics)

---

## Repository Architecture

**Purpose**: Physics/maths-driven ML learning path from numerical foundations to production systems. This is a **learning-focused monorepo** organized as a progressive curriculum (modules 00-11) covering numerical methods → ML fundamentals → deep learning → MLOps.

**Educational philosophy**: 
- **Foundations-first**: Start with numerical methods (module 01) before ML
- **From-scratch implementations**: Core algorithms built without libraries for understanding
- **Library-grade quality**: Then refactor to production-quality code with tests
- **Theory → Practice**: Each module combines minimal theory, implementation, and failure mode analysis

**Key pattern**: Modular monorepo where each `modules/XX_*` is a self-contained learning module with standardized structure:
```
modules/XX_name/
├── src/          # Importable package code (library implementations)
├── tests/        # Unit + integration tests (>80% coverage target)
├── configs/      # YAML experiment configs (Pydantic-validated)
├── notebooks/    # Jupyter exploration (numbered, educational)
├── experiments/  # Standalone experiment scripts (optional)
└── reports/      # Outputs (gitignored: plots, logs, metrics)
```

**Core utilities**: `modules/00_repo_standards/src/mlphys_core/` provides shared infrastructure (config, seeding, logging, experiment base class) for all modules. This is the "stdlib" of the repo.

**Module scaffold**: Use `python modules/00_repo_standards/create_module.py --number XX --title "Name"` to generate standardized module structure with templates.

## Import Strategy

**Critical**: Python 3.12+ has a syntax limitation with module names starting with digits. Use this strategy:

### Within Same Package: Relative Imports
```python
# ✅ In modules/00_repo_standards/src/mlphys_core/__init__.py or nested packages
from .config import ExperimentConfig
from .seeding import set_seed
from .experiment import BaseExperiment
```

### Across Modules: Import Helper
```python
# ✅ In modules/02_stat_inference_uq/tests/test_something.py
from modules._import_helper import safe_import_from

# Import from another numeric module (can destructure multiple names)
get_rng, set_seed = safe_import_from(
    '00_repo_standards.src.mlphys_core.seeding',
    'get_rng', 'set_seed'
)
BayesianLinearRegression = safe_import_from(
    '02_stat_inference_uq.src.bayesian_regression',
    'BayesianLinearRegression'
)

# ❌ NEVER DO THIS - Will fail in Python 3.12+
from modules.00_repo_standards.src.mlphys_core import set_seed  # SyntaxError!
from modules.02_stat_inference_uq.src import BayesianLinearRegression  # SyntaxError!
```

**Why**: Python 3.12+ interprets `modules.02_` as an invalid octal literal (`02` looks like octal prefix).  
**Documentation**: See [docs/getting-started/python312-imports.md](../docs/getting-started/python312-imports.md) for complete details.

### Import Helper Details
- `safe_import(module_path)` - Returns module object
- `safe_import_from(module_path, *names)` - Returns tuple (or single value if one name)
- Module paths **exclude** the `modules.` prefix: use `'00_repo_standards.src.mlphys_core'` not `'modules.00_repo_standards.src.mlphys_core'`
- Returns single value when one name, tuple when multiple - destructure appropriately

## Reproducibility Requirements

**Every experiment MUST**:
1. Accept `--seed` parameter (default 42)
2. Call `set_seed(seed)` from `mlphys_core.seeding` before randomness
3. Use YAML configs extending `ExperimentConfig` (Pydantic validation)
4. Log git commit hash and config to `reports/`
5. Produce bit-identical outputs with same seed

**Example**:
```python
from modules._import_helper import safe_import_from

set_seed, ExperimentConfig, load_config = safe_import_from(
    '00_repo_standards.src.mlphys_core',
    'set_seed', 'ExperimentConfig', 'load_config'
)

config = load_config(Path("configs/exp.yaml"), MyConfig)
set_seed(config.seed)  # Sets Python, NumPy, PyTorch seeds
# ... run experiment ...
```

**RNG isolation**: For function-level isolation without affecting global state, use:
```python
rng = get_rng(42)  # Isolated NumPy Generator
samples = rng.standard_normal(100)
```

## Testing Conventions

- **Location**: `modules/XX_name/tests/test_*.py`
- **Run**: `make test` (all), `make test-module MODULE=01_numerical_toolbox` (specific)
- **Coverage**: Target >80%, checked via pytest-cov, reports in `htmlcov/`
- **Types**: 
  - **Unit tests** - Isolated function/class tests (pure logic)
  - **Integration tests** - Full pipeline tests (see [modules/00_repo_standards/tests/test_integration.py](../modules/00_repo_standards/tests/test_integration.py))
  - **Sanity checks** - Optimizer reduces loss, predictions have correct shape, etc.
- **Fixtures**: Use pytest fixtures for shared setup, especially for data/configs
- **Root conftest**: [conftest.py](../conftest.py) adds repo root to `sys.path` for imports

**Example integration test** (reproducibility check):
```python
def test_reproducibility_with_seed(self):
    set_seed(42)
    result1 = run_experiment()
    set_seed(42)
    result2 = run_experiment()
    assert np.array_equal(result1, result2)  # Bit-identical
```

**Example sanity test**:
```python
def test_optimizer_decreases_loss(self):
    x0 = np.random.randn(10) * 10
    x_opt, history = gradient_descent(x0, grad_fn, lr=0.1)
    assert history[-1] < history[0]  # Loss decreased
```

## Development Workflow

**Key commands** (see [Makefile](../Makefile)):
- `make setup` - Install deps + pre-commit hooks (first time)
- `make test` - Run tests with coverage (pytest + coverage report)
- `make test-module MODULE=XX_name` - Test specific module only
- `make test-fast` - Quick tests without coverage (`-x` stops on first failure)
- `make lint` - Ruff + Black + MyPy checks
- `make format` - Auto-fix formatting
- `make run-module MODULE=XX_name ARGS="--help"` - Run module CLI
- `make mlflow-ui` - View experiment tracking at http://localhost:5000
- `make clean` - Remove `__pycache__`, `.pytest_cache`, coverage reports

**Adding a new module**:
1. Use `python modules/00_repo_standards/create_module.py --number XX --title "Name"` for scaffolding
2. Extend `ExperimentConfig` for module-specific config schema
3. Inherit from `BaseExperiment` for standard experiment structure (see [modules/00_repo_standards/src/mlphys_core/experiment.py](../modules/00_repo_standards/src/mlphys_core/experiment.py))
4. Add tests before implementation (TDD encouraged)
5. Update module README with completion checklist

**Module CLI pattern**: Modules expose Typer CLIs at `src/main.py` - run via `python -m modules.XX_name.src.main train --config configs/exp.yaml --seed 123`

## Configuration Pattern

**Use Pydantic models** inheriting from `ExperimentConfig`:
```python
from modules._import_helper import safe_import_from

ExperimentConfig = safe_import_from(
    '00_repo_standards.src.mlphys_core',
    'ExperimentConfig'
)

class OptimizerConfig(ExperimentConfig):
    learning_rate: float = 0.01
    max_iter: int = 1000
    # Inherits: name, seed, output_dir, log_level, etc.
```

**YAML files** in `configs/` validated against schema. Paths are auto-converted to `pathlib.Path`.

**Loading configs**:
```python
config = load_config(Path("configs/experiment.yaml"), OptimizerConfig)
```

## Code Quality Standards

- **Type hints**: All public functions (MyPy enforced)
- **Docstrings**: Google-style for public APIs
- **Line length**: 100 chars (Black + Ruff)
- **Imports**: Auto-sorted with isort via Ruff
- **No hardcoded paths**: Use `Path` objects, configs, or repo-relative paths
- **Pre-commit hooks**: Auto-run formatting/linting checks on commit (trailing whitespace, YAML, large files, etc.)

**Math/physics comments**: Add inline comments for non-obvious numerical steps (e.g., "Condition number affects convergence rate"), avoid narrating obvious operations.

## Notebook Conventions

**Jupyter notebooks** in `notebooks/` are numbered for sequential learning (e.g., `01_topic.ipynb`, `02_next_topic.ipynb`):

- **Always include** this import pattern at the top:
  ```python
  from modules._import_helper import safe_import_from
  # Import what you need
  set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
  MyClass = safe_import_from('XX_module.src.my_file', 'MyClass')
  ```
- **Setup cell**: imports, matplotlib style, reports directory creation
- **Educational focus**: Show both "how it works" (theory) and "how to use it" (practical)
- **Visualizations**: Save plots to `../reports/` (gitignored)
- **Failure modes**: Demonstrate what breaks and why (e.g., bad initialization, poor hyperparams)

## Experiment CLI Pattern

Modules expose Typer CLIs at `src/main.py`:
```python
import typer
from pathlib import Path
from modules._import_helper import safe_import_from

ExperimentConfig = safe_import_from('00_repo_standards.src.mlphys_core', 'ExperimentConfig')

app = typer.Typer()

@app.command()
def train(
    config: Path = typer.Option("configs/default.yaml"),
    seed: int = typer.Option(42),
):
    """Train model with specified config."""
    # Implementation
    
if __name__ == "__main__":
    app()
```

Run via: `python -m modules.XX_name.src.main train --config configs/exp.yaml --seed 123`  
Or use Makefile: `make run-module MODULE=XX_name ARGS="train --config configs/exp.yaml"`

## MLflow Integration

Optional tracking enabled via `config.mlflow_tracking = True`:
- Logs to `mlruns/` directory
- UI: `make mlflow-ui` → http://localhost:5000
- Automatically logs: params, metrics, git commit, config file

## Pitfalls & Gotchas

1. **Import errors**: Always use `safe_import_from()` for cross-module imports with numeric prefixes
2. **RNG state**: Call `set_seed()` at experiment start, not inside functions (unless using isolated RNG via `get_rng()`)
3. **Path handling**: Config `output_dir` auto-resolves relative to module dir (see `BaseExperiment.__init__`)
4. **PyTorch seeding**: `set_seed()` handles torch.manual_seed if PyTorch available
5. **Test isolation**: Tests run from repo root, so use `Path("modules/XX/configs/...")` not `Path("configs/...")`
6. **safe_import_from destructuring**: Returns tuple if multiple names, single value if one name. For single values, use: `value = safe_import_from(..., 'name')` not `(value,) = ...`
7. **Module paths**: In `safe_import_from()`, always exclude `modules.` prefix - use `'00_repo_standards.src.mlphys_core'`
8. **Notebook imports**: In notebooks, import via `safe_import_from()` - never use standard imports for numeric-prefixed modules
9. **Running Python modules**: Use `python -m modules.XX_name.src.main` not `python modules/XX_name/src/main.py` to preserve import paths
10. **Git tracking**: The `reports/` directory is gitignored - outputs won't be committed (by design for reproducibility)

## Module-Specific Patterns

### Module README Structure
Every module README follows this pattern (enforced by `create_module.py` scaffold):
- **Status**: 📋 Planned / 🚧 In Progress / ✅ Complete
- **Learning Objectives**: What you'll learn
- **Theory**: Minimal background (not a textbook)
- **Implementation Plan**: Checklist of tasks
- **Experiments & Metrics**: What to measure and baselines
- **Failure Modes**: Common pitfalls with solutions
- **Resources**: Papers, tutorials, documentation

### Notebooks vs Tests vs Experiments
- **`notebooks/`**: Interactive exploration, visualization, educational demos (not tested)
- **`tests/`**: Pytest unit/integration tests for code correctness (>80% coverage)
- **`experiments/`**: Standalone scripts for benchmarks/ablations (optional, when too complex for notebooks)
- **`src/`**: Library-quality implementations imported everywhere

## Adding Dependencies

Edit `pyproject.toml` under `dependencies` or `[project.optional-dependencies]`, then:
```bash
pip install -e ".[dev,docs]"  # Reinstall to pick up new deps
```

Prefer pinning major versions (e.g., `numpy>=1.26.0`) for stability. Current stack:
- **Core ML**: numpy, pandas, scipy, scikit-learn, torch, lightgbm
- **Config/CLI**: pydantic, typer, pyyaml, gitpython
- **Tracking**: mlflow
- **Dev**: pytest, pytest-cov, ruff, black, mypy, pre-commit
- **Notebooks**: jupyter, ipykernel

## Pre-commit Hooks

Hooks auto-run on commit (see `.pre-commit-config.yaml`):
- **Ruff**: Linting with auto-fix
- **Black**: Code formatting (Python 3.11+)
- **MyPy**: Static type checking (skips tests/notebooks)
- **Pre-commit-hooks**: Trailing whitespace, YAML validation, large file checks, merge conflicts, debug statements

**Manual run**: `make pre-commit` or `pre-commit run --all-files`

## Testing Best Practices

### Parametrized Tests
Use `@pytest.mark.parametrize` for testing multiple cases efficiently:
```python
@pytest.mark.parametrize("seed", [0, 1, 42, 100, 12345])
def test_reproducibility_different_seeds(seed):
    set_seed(seed)
    result = run_experiment()
    # Check determinism...
```

### Error Validation
Test error messages with `pytest.raises`:
```python
def test_invalid_input():
    with pytest.raises(ValueError, match="X must be 2D"):
        model.fit(x_1d, y)
```

### Fixtures for Shared Setup
```python
@pytest.fixture
def sample_data():
    """Setup test fixtures."""
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    return X, y

def test_model_fit(sample_data):
    X, y = sample_data
    model.fit(X, y)
    assert model.is_fitted
```

## License & Attribution

MIT License - compatible with all major ML libraries. When adding code adapted from papers/tutorials, cite in docstring.

---

## Troubleshooting Guide

### Import Errors
**Problem**: `SyntaxError: invalid decimal literal` when importing  
**Solution**: Use `safe_import_from()` instead of standard imports for any numeric-prefixed module

**Problem**: `ModuleNotFoundError: No module named 'modules.XX_name'`  
**Solution**: Check that you're running from repo root, or that `conftest.py` added repo to `sys.path`

### Test Failures
**Problem**: Tests fail with "fixture not found"  
**Solution**: Check that module's `tests/conftest.py` exists and defines the fixture

**Problem**: Non-deterministic test failures  
**Solution**: Verify `set_seed()` is called before any randomness, not inside tested functions

**Problem**: Path not found errors in tests  
**Solution**: Use absolute paths from repo root: `Path("modules/XX/configs/...")`, not relative paths

### Development Workflow
**Problem**: Pre-commit hooks failing  
**Solution**: Run `make format` to auto-fix, then `make lint` to check remaining issues

**Problem**: MyPy type errors on imports  
**Solution**: MyPy may struggle with `safe_import_from()` - add `# type: ignore` or use `cast()`

**Problem**: Coverage not meeting 80% threshold  
**Solution**: Check `htmlcov/index.html` for uncovered lines, add tests for edge cases

### Experiment Issues
**Problem**: Results not reproducible with same seed  
**Solution**: Check for uncontrolled randomness (e.g., `np.random.rand()` instead of `rng.random()`)

**Problem**: MLflow not tracking experiments  
**Solution**: Verify `config.mlflow_tracking = True` and check `mlruns/` directory exists

**Problem**: Config validation errors  
**Solution**: Ensure YAML matches Pydantic schema, check for typos in field names

### Environment Setup
**Problem**: Dev container not starting  
**Solution**: Check Docker is running, rebuild container with "Dev Containers: Rebuild Container"

**Problem**: Dependencies not installing  
**Solution**: Try `pip install -e ".[dev,docs]"` with quotes, or update pip first: `pip install --upgrade pip`

### Quick Diagnostics
```bash
# Verify Python version (requires 3.11+)
python --version

# Check import system works
python -c "from modules._import_helper import safe_import_from; print('OK')"

# Test specific module imports
python -c "from modules._import_helper import safe_import_from; set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed'); print('OK')"

# Verify pytest can find modules
python -c "import sys; print('modules' in sys.path or any('Computational-ML-lab' in p for p in sys.path))"
```
