"""Monte Carlo integration with confidence intervals."""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MCResult:
    """Results from Monte Carlo estimation.
    
    Attributes:
        estimate: Point estimate of the integral/expectation
        std_error: Standard error of the estimate
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        n_samples: Number of samples used
        confidence_level: Confidence level (e.g., 0.95)
    """
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_samples: int
    confidence_level: float


class MCIntegrator:
    """
    Monte Carlo integrator for computing integrals/expectations.
    
    Computes:
        I = ∫_a^b f(x) dx ≈ (b-a) * (1/N) Σ f(X_i)
    
    where X_i ~ Uniform(a, b)
    
    Example:
        >>> integrator = MCIntegrator(seed=42)
        >>> result = integrator.integrate(lambda x: x**2, a=0, b=1, n_samples=10000)
        >>> print(f"Estimate: {result.estimate:.4f} ± {result.std_error:.4f}")
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize integrator with optional seed."""
        self.rng = np.random.default_rng(seed)
    
    def integrate(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        a: float,
        b: float,
        n_samples: int = 10000,
        confidence_level: float = 0.95,
    ) -> MCResult:
        """
        Estimate integral using Monte Carlo.
        
        Args:
            func: Function to integrate (must accept array input)
            a: Lower bound
            b: Upper bound
            n_samples: Number of samples
            confidence_level: Confidence level for CI (default 0.95)
            
        Returns:
            MCResult with estimate and confidence interval
        """
        # Sample uniformly in [a, b]
        x = self.rng.uniform(a, b, n_samples)
        
        # Evaluate function
        f_x = func(x)
        
        # Estimate integral: (b-a) * E[f(X)]
        volume = b - a
        estimate = volume * np.mean(f_x)
        
        # Standard error: (b-a) * sqrt(Var[f(X)] / N)
        std_error = volume * np.std(f_x, ddof=1) / np.sqrt(n_samples)
        
        # Confidence interval (assume normal by CLT)
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = estimate - z * std_error
        ci_upper = estimate + z * std_error
        
        return MCResult(
            estimate=estimate,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n_samples,
            confidence_level=confidence_level,
        )
    
    def integrate_multidim(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        bounds: list[Tuple[float, float]],
        n_samples: int = 10000,
        confidence_level: float = 0.95,
    ) -> MCResult:
        """
        Estimate multidimensional integral.
        
        I = ∫...∫ f(x) dx₁...dx_d
        
        Args:
            func: Function taking (n_samples, d) array
            bounds: List of (lower, upper) for each dimension
            n_samples: Number of samples
            confidence_level: Confidence level
            
        Returns:
            MCResult
        """
        d = len(bounds)
        
        # Sample uniformly in hypercube
        x = np.zeros((n_samples, d))
        volume = 1.0
        for i, (a, b) in enumerate(bounds):
            x[:, i] = self.rng.uniform(a, b, n_samples)
            volume *= (b - a)
        
        # Evaluate function
        f_x = func(x)
        
        # Estimate
        estimate = volume * np.mean(f_x)
        std_error = volume * np.std(f_x, ddof=1) / np.sqrt(n_samples)
        
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = estimate - z * std_error
        ci_upper = estimate + z * std_error
        
        return MCResult(
            estimate=estimate,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_samples=n_samples,
            confidence_level=confidence_level,
        )


def estimate_integral(
    func: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Convenience function for quick MC integration.
    
    Args:
        func: Function to integrate
        a: Lower bound
        b: Upper bound
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        (estimate, standard_error)
    """
    integrator = MCIntegrator(seed=seed)
    result = integrator.integrate(func, a, b, n_samples)
    return result.estimate, result.std_error


def convergence_analysis(
    func: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    true_value: float,
    n_samples_list: list[int],
    n_runs: int = 10,
    seed: Optional[int] = None,
) -> dict:
    """
    Analyze convergence of MC estimator.
    
    Args:
        func: Function to integrate
        a, b: Integration bounds
        true_value: True integral value (for error computation)
        n_samples_list: List of sample sizes to test
        n_runs: Number of independent runs per sample size
        seed: Random seed
        
    Returns:
        Dictionary with arrays: n_samples, mean_error, std_error, bias
    """
    rng = np.random.default_rng(seed)
    
    results = {
        'n_samples': np.array(n_samples_list),
        'mean_error': [],
        'std_error': [],
        'bias': [],
        'rmse': [],
    }
    
    for n in n_samples_list:
        estimates = []
        for _ in range(n_runs):
            run_seed = rng.integers(0, 1_000_000)
            est, _ = estimate_integral(func, a, b, n, seed=run_seed)
            estimates.append(est)
        
        estimates = np.array(estimates)
        errors = estimates - true_value
        
        results['mean_error'].append(np.abs(errors).mean())
        results['std_error'].append(np.std(estimates, ddof=1))
        results['bias'].append(np.mean(errors))
        results['rmse'].append(np.sqrt(np.mean(errors**2)))
    
    # Convert to arrays
    for key in ['mean_error', 'std_error', 'bias', 'rmse']:
        results[key] = np.array(results[key])
    
    return results
