# Contributing Guide

Welcome! This guide explains how to contribute to the Computational ML Lab.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/Computational-ML-lab.git
cd Computational-ML-lab
```

### 2. Install Dependencies

```bash
make setup
# Or manually:
pip install -e ".[dev,docs]"
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/my-new-feature
```

## Development Workflow

### Code Style

We use automated formatting and linting:

- **Ruff**: Fast Python linter
- **Black**: Code formatter (100 char line length)
- **MyPy**: Type checking

Run before committing:

```bash
make format  # Auto-fix formatting
make lint    # Check for issues
```

### Type Hints

Add type hints to all public functions:

```python
def compute_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))
```

### Docstrings

Use Google style docstrings:

```python
def gradient_descent(
    x0: np.ndarray,
    grad_fn: callable,
    lr: float = 0.01,
) -> tuple[np.ndarray, list]:
    """
    Gradient descent optimizer.
    
    Args:
        x0: Initial point (shape: [n_features])
        grad_fn: Function computing gradient
        lr: Learning rate (default: 0.01)
        
    Returns:
        Tuple of (optimal_point, loss_history)
        
    Raises:
        ValueError: If lr <= 0
        
    Example:
        >>> def grad(x): return 2*x
        >>> x_opt, history = gradient_descent(np.array([1.0]), grad, lr=0.1)
    """
    pass
```

## Testing

### Writing Tests

Use `pytest` with clear test names:

```python
import pytest
import numpy as np
from modules.XX_module.src.core import my_function

def test_my_function_with_valid_input():
    """Test that function handles valid input correctly."""
    result = my_function(x=1.0, y=2.0)
    assert result == 3.0

def test_my_function_raises_on_negative():
    """Test that function raises ValueError on negative input."""
    with pytest.raises(ValueError, match="x must be positive"):
        my_function(x=-1.0, y=2.0)

@pytest.mark.parametrize("x,y,expected", [
    (0, 0, 0),
    (1, 1, 2),
    (2, 3, 5),
])
def test_my_function_parametrized(x, y, expected):
    """Test multiple input combinations."""
    assert my_function(x, y) == expected
```

### Running Tests

```bash
# All tests
make test

# Specific module
make test-module MODULE=01_numerical_toolbox

# Single test file
pytest modules/01_numerical_toolbox/tests/test_core.py -v

# Single test function
pytest modules/01_numerical_toolbox/tests/test_core.py::test_gradient_descent -v
```

### Coverage

Aim for >80% coverage on `src/` code:

```bash
make test  # Generates htmlcov/index.html
```

## Reproducibility

### Seeds

Always accept and use a seed parameter:

```python
def run_experiment(seed: int = 42):
    """Run experiment with deterministic results."""
    np.random.seed(seed)
    # ... rest of code
```

### Logging

Use MLflow for experiment tracking:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_param("seed", 42)
    mlflow.log_metric("rmse", 0.15)
```

### Configuration

Use YAML configs for experiments:

```yaml
# configs/experiment.yaml
experiment:
  name: "test_run"
  seed: 42

model:
  type: "LinearRegression"
  alpha: 0.01
```

Load with:

```python
import yaml
from pathlib import Path

def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)
```

## Module Development

### Adding a New Module

1. Copy the module template:
   ```bash
   cp -r modules/00_repo_standards modules/XX_new_module
   ```

2. Update `README.md` with:
   - Learning objectives
   - Theory (concise)
   - Implementation plan
   - Experiments & metrics

3. Implement in `src/`:
   - Core algorithms
   - CLI entry point
   - Tests

4. Add notebooks for exploration

5. Update docs:
   - Add to `docs/modules/overview.md`
   - Update main `README.md`

### Module Checklist

- [ ] README with objectives, theory, plan
- [ ] `src/` with type hints and docstrings
- [ ] `tests/` with >80% coverage
- [ ] CLI entry point with Typer
- [ ] Config files in `configs/`
- [ ] Notebooks for exploration
- [ ] Experiments logged to MLflow
- [ ] Documentation updated

## Pull Request Process

### 1. Pre-commit Checks

```bash
make lint
make test
make pre-commit
```

### 2. Commit Messages

Use conventional commits:

```
feat: add Kalman filter implementation
fix: correct numerical stability in matrix inversion
docs: update module 01 README
test: add unit tests for gradient descent
refactor: simplify data loading pipeline
```

### 3. Open PR

- Clear title and description
- Reference any related issues
- Include test results
- Add screenshots for visual changes

### 4. Code Review

Address reviewer feedback and update branch:

```bash
git add .
git commit -m "fix: address review comments"
git push origin feature/my-new-feature
```

## Documentation

### Building Docs

```bash
make serve-docs  # http://localhost:8000
```

### Adding Pages

1. Create markdown in `docs/`
2. Update `mkdocs.yml` nav section
3. Use material for MkDocs features:

```markdown
!!! note "Important"
    This is a callout box.

!!! warning
    Watch out for numerical instability!

```python
# Code blocks with syntax highlighting
def example():
    pass
```

## Style Guidelines

### Python

- Line length: 100 characters
- Imports: grouped (stdlib, third-party, local)
- Use absolute imports from module root
- Avoid `from module import *`

### Naming

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Comments

- Explain **why**, not **what**
- For complex math, reference equations
- Use docstrings for public APIs

```python
# Good
# Use Cholesky decomposition for numerical stability
L = np.linalg.cholesky(A + 1e-6 * np.eye(n))

# Bad
# Compute Cholesky
L = np.linalg.cholesky(A + 1e-6 * np.eye(n))
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing modules for examples

Thank you for contributing! 🎉
