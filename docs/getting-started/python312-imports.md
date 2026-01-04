# Python 3.12+ Import Workaround

## Problem

Python 3.12+ interprets module names starting with digits (e.g., `modules.02_stat_inference_uq`) as invalid octal literals, causing `SyntaxError: invalid decimal literal` during import.

## Solution Strategy

We use **two complementary approaches** depending on where the import occurs:

### 1. Within Same Module Package: Relative Imports

For imports within the same package/module, use **relative imports**:

```python
# ✅ In modules/00_repo_standards/src/__init__.py
from .core import compute_mean
from .utils import set_seed

# ✅ In modules/00_repo_standards/src/mlphys_core/__init__.py
from .config import ExperimentConfig
from .seeding import set_seed
```

**When to use:** Importing sibling modules within the same package.

### 2. Across Modules: Import Helper

For imports **across different modules** (e.g., module 02 importing from module 00), use the `safe_import_from` helper:

```python
# ✅ In modules/02_stat_inference_uq/tests/test_bayesian_regression.py
from modules._import_helper import safe_import_from

# Import from module 00
get_rng = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'get_rng')

# Import from module 02
BayesianLinearRegression, posterior_predictive = safe_import_from(
    '02_stat_inference_uq.src.bayesian_regression',
    'BayesianLinearRegression', 'posterior_predictive'
)
```

**When to use:** Importing from other numeric-prefixed modules.

## Files Modified

### Core Infrastructure
- `modules/_import_helper.py` - Created helper functions
- `modules/__init__.py` - Package initialization
- `conftest.py` (root) - Pytest configuration to add repo to path
- `modules/00_repo_standards/tests/conftest.py` - Module-specific pytest config
- `modules/01_numerical_toolbox/tests/conftest.py` - Module-specific pytest config
- `modules/02_stat_inference_uq/tests/conftest.py` - Module-specific pytest config
- `modules/00_repo_standards/src/__init__.py` - Relative imports
- `modules/00_repo_standards/src/mlphys_core/__init__.py` - Relative imports
- `modules/00_repo_standards/src/mlphys_core/experiment.py` - Relative imports
- `modules/01_numerical_toolbox/src/__init__.py` - Relative imports

### Module 00 (All Files Updated)
- `modules/00_repo_standards/src/main.py` - Uses safe_import_from
- `modules/00_repo_standards/src/demo_experiment.py` - Uses safe_import_from
- `modules/00_repo_standards/run_demo.py` - Uses safe_import_from
- All 6 test files in `modules/00_repo_standards/tests/` - Uses safe_import_from

### Module 01 (All Files Updated)
- `modules/01_numerical_toolbox/experiments/optimizer_benchmark.py` - Uses safe_import_from
- All 3 test files in `modules/01_numerical_toolbox/tests/` - Uses safe_import_from

### Module 02 (Example)
- `modules/02_stat_inference_uq/src/main.py` - Uses safe_import_from
- `modules/02_stat_inference_uq/tests/*.py` - Uses safe_import_from

## Future Module Creation

When creating new modules (03+), follow this pattern:

```python
# In your module's code/tests
from modules._import_helper import safe_import_from

# Import mlphys_core utilities
set_seed, get_rng = safe_import_from(
    '00_repo_standards.src.mlphys_core.seeding',
    'set_seed', 'get_rng'
)

# Import from your module (if needed across files)
MyClass = safe_import_from('XX_my_module.src.my_file', 'MyClass')
```

## Why Not Just Rename Modules?

While renaming modules to avoid numeric prefixes would solve this, we opted to keep the existing naming convention (`00_`, `01_`, etc.) because:

1. **Pedagogical clarity**: Numbers indicate learning progression
2. **Alphabetical ordering**: Natural sorting in file explorers
3. **Consistency**: Matches existing documentation and curriculum design

## Alternative Considered

We could have used `importlib.import_module()` everywhere, but the helper function provides:
- Cleaner syntax
- Type safety
- Centralized workaround logic
- Easier future migration if Python fixes this issue

## Technical Details

The root cause is Python's lexer treating `0X` after a dot as the start of an octal/hex literal before checking if it's a valid numeric literal. The underscore separator makes it invalid, triggering the syntax error.

**Example of the error:**
```
E     File ".../modules/00_repo_standards/src/__init__.py", line 3
E       from modules.00_repo_standards.src.core import compute_mean
E                      ^
E   SyntaxError: invalid decimal literal
```

Our solution bypasses the lexer by using `importlib.import_module()` (which takes a string, not parsed code) or relative imports (which don't include the `modules.XX_` prefix).
