"""Numerical toolbox for ML: optimizers, linear algebra, toy problems."""

# Python 3.12+ workaround - use relative imports within package
from .optimizers_from_scratch import (
    GradientDescent,
    MomentumOptimizer,
    AdamOptimizer,
    OptimizationResult,
)
from .linear_algebra import (
    pca_via_svd,
    PCAResult,
    condition_number,
    ridge_regularization,
    demonstrate_ill_conditioning,
)
from .toy_problems import (
    create_quadratic_bowl,
    QuadraticBowl,
    create_linear_regression,
    LinearRegressionProblem,
    linear_regression_closed_form,
    linear_regression_gradient_descent,
    rosenbrock_function,
    rosenbrock_gradient,
    beale_function,
    beale_gradient,
)

__all__ = [
    # Optimizers
    "GradientDescent",
    "MomentumOptimizer",
    "AdamOptimizer",
    "OptimizationResult",
    # Linear algebra
    "pca_via_svd",
    "PCAResult",
    "condition_number",
    "ridge_regularization",
    "demonstrate_ill_conditioning",
    # Toy problems
    "create_quadratic_bowl",
    "QuadraticBowl",
    "create_linear_regression",
    "LinearRegressionProblem",
    "linear_regression_closed_form",
    "linear_regression_gradient_descent",
    "rosenbrock_function",
    "rosenbrock_gradient",
    "beale_function",
    "beale_gradient",
]
