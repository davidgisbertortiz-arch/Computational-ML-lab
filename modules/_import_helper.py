"""
Workaround for Python 3.12+ octal literal parsing with numeric module names.

Python 3.12+ interprets `from modules.02_name` as an invalid octal literal.
This helper provides safe import functions that bypass the lexer issue.

Usage:
    from modules._import_helper import safe_import
    
    # Instead of: from modules.02_stat_inference_uq.src.bayesian_regression import BayesianLinearRegression
    _mod = safe_import('02_stat_inference_uq.src.bayesian_regression')
    BayesianLinearRegression = _mod.BayesianLinearRegression
"""

import importlib
import sys
from typing import Any


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
    return importlib.import_module(full_path)


def safe_import_from(module_path: str, *names: str) -> tuple:
    """
    Safely import specific names from a module with numeric prefix.
    
    Args:
        module_path: Module path WITHOUT the 'modules.' prefix
        *names: Names to import from the module
    
    Returns:
        Tuple of imported objects (single object if only one name)
    
    Example:
        >>> BayesianLinearRegression, posterior_predictive = safe_import_from(
        ...     '02_stat_inference_uq.src.bayesian_regression',
        ...     'BayesianLinearRegression', 'posterior_predictive'
        ... )
    """
    module = safe_import(module_path)
    result = tuple(getattr(module, name) for name in names)
    return result[0] if len(result) == 1 else result
