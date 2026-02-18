"""
Workaround for Python 3.12+ octal literal parsing with numeric module names.

Python 3.12+ interprets `from modules.02_name` as an invalid octal literal.
This helper provides safe import functions that bypass the lexer issue.

Usage:
    from modules._import_helper import safe_import_from
    
    # Instead of: from modules.02_stat_inference_uq.src.bayesian_regression import BayesianLinearRegression
    BayesianLinearRegression = safe_import_from('02_stat_inference_uq.src.bayesian_regression', 'BayesianLinearRegression')
"""

import importlib
from typing import Any, Union


def safe_import(module_path: str) -> Any:
    """
    Safely import a module with a numeric prefix using importlib.
    
    Args:
        module_path: Module path WITHOUT the 'modules.' prefix.
                    E.g., '02_stat_inference_uq.src.bayesian_regression'
                         '00_repo_standards.src.mlphys_core.seeding'
    
    Returns:
        The imported module object.
    
    Example:
        >>> _bayes = safe_import('02_stat_inference_uq.src.bayesian_regression')
        >>> BayesianLinearRegression = _bayes.BayesianLinearRegression
    """
    full_path = f'modules.{module_path}'
    try:
        return importlib.import_module(full_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Cannot import '{full_path}'. "
            f"Ensure module path is correct and excludes 'modules.' prefix. "
            f"Example: '00_repo_standards.src.core' not 'modules.00_repo_standards.src.core'"
        ) from e


def safe_import_from(module_path: str, *names: str) -> Union[Any, tuple[Any, ...]]:
    """
    Safely import specific names from a module with numeric prefix.
    
    Args:
        module_path: Module path WITHOUT the 'modules.' prefix
        *names: Names to import from the module
    
    Returns:
        Single object if one name provided, tuple of objects if multiple names
    
    Example:
        >>> # Single import
        >>> set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
        >>> 
        >>> # Multiple imports
        >>> BayesianLinearRegression, posterior_predictive = safe_import_from(
        ...     '02_stat_inference_uq.src.bayesian_regression',
        ...     'BayesianLinearRegression', 'posterior_predictive'
        ... )
    """
    module = safe_import(module_path)
    result = tuple(getattr(module, name) for name in names)
    return result[0] if len(result) == 1 else result
