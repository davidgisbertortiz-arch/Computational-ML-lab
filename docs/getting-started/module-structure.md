# Module Structure

Every module in this repository follows a consistent structure to ensure clarity and maintainability.

## Directory Layout

```
modules/XX_module_name/
├── README.md           # Module overview, learning objectives, theory
├── src/
│   ├── __init__.py
│   ├── main.py         # CLI entry point
│   ├── core.py         # Core algorithms
│   └── utils.py        # Helper functions
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_experiments.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_integration.py
├── configs/
│   ├── baseline.yaml
│   └── experiment.yaml
└── reports/
    ├── figures/
    ├── results.csv
    └── analysis.md
```

## Module README Template

Each module's README should include:

### 1. Learning Objectives
Clear bullet points on what you'll master

### 2. Minimal Theory
Concise mathematical background (not a textbook!)

### 3. Implementation Plan
- **From Scratch**: Core algorithm implemented manually
- **Library Grade**: Production-quality implementation with edge cases
- **Experiments**: Evaluation on real/synthetic data

### 4. Experiments & Metrics
Specific datasets, evaluation metrics, and baselines

### 5. Failure Modes
Common pitfalls and how to avoid them

### 6. Definition of Done
Checklist for module completion:
- [ ] Theory notes complete
- [ ] From-scratch implementation with tests
- [ ] Library-grade implementation
- [ ] Experiments run with logged metrics
- [ ] Documentation and docstrings
- [ ] Code reviewed and linted

## Source Code (`src/`)

### main.py
CLI entry point using Typer or argparse:

```python
import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def train(
    config: Path = typer.Option(..., help="Config file"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Train the model."""
    # Implementation
    pass

if __name__ == "__main__":
    app()
```

### core.py
Core algorithms with:
- Type hints
- Docstrings (Google style)
- Numerical stability considerations
- Clear separation of concerns

```python
import numpy as np
from typing import Tuple

def gradient_descent(
    x0: np.ndarray,
    grad_fn: callable,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, list]:
    """
    Gradient descent optimizer.
    
    Args:
        x0: Initial point
        grad_fn: Function computing gradient
        lr: Learning rate
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Optimal point and loss history
    """
    # Implementation
    pass
```

## Tests (`tests/`)

### Unit Tests
Test individual functions with known inputs/outputs:

```python
import pytest
import numpy as np
from modules.XX_module.src.core import gradient_descent

def test_gradient_descent_quadratic():
    """Test GD on simple quadratic."""
    def grad_fn(x):
        return 2 * x
    
    x_opt, history = gradient_descent(
        x0=np.array([1.0]),
        grad_fn=grad_fn,
        lr=0.1,
    )
    
    assert np.allclose(x_opt, 0.0, atol=1e-3)
```

### Integration Tests
Test full pipelines:

```python
def test_full_training_pipeline():
    """Test end-to-end training."""
    # Load data, train, evaluate
    pass
```

## Configs (`configs/`)

YAML configuration files for experiments:

```yaml
# baseline.yaml
experiment:
  name: "baseline_v1"
  seed: 42

data:
  path: "data/dataset.csv"
  train_split: 0.8

model:
  type: "LinearRegression"
  hyperparams:
    alpha: 0.01

training:
  max_iter: 1000
  tol: 1e-6
```

## Notebooks (`notebooks/`)

Jupyter notebooks for:
1. **Exploration** - EDA, visualizations, sanity checks
2. **Experiments** - Running configs, comparing models
3. **Analysis** - Interpreting results, failure analysis

Keep notebooks **short** and move reusable code to `src/`.

## Reports (`reports/`)

Generated artifacts:
- `figures/` - Plots, diagrams
- `results.csv` - Metrics tables
- `analysis.md` - Written interpretations

These are **gitignored** by default (except small examples).

## Best Practices

1. **Imports**: Absolute imports from module root
   ```python
   from modules.XX_module.src.core import MyClass
   ```

2. **Reproducibility**: Always accept a `seed` parameter

3. **Logging**: Use Python `logging` module, not `print`

4. **Type Hints**: Add to all public functions

5. **Docstrings**: Google style for all public APIs

6. **Tests**: Aim for >80% coverage on `src/`

## Running a Module

```bash
# CLI
python -m modules.01_numerical_toolbox.src.main train --config configs/baseline.yaml

# Tests
pytest modules/01_numerical_toolbox/tests/

# Notebook
jupyter notebook modules/01_numerical_toolbox/notebooks/01_exploration.ipynb
```

## Next Steps

- See [Module 00: Repo Standards](../../modules/00_repo_standards/README.md) for a concrete example
- Check the [Contributing Guide](../contributing.md) for code style
