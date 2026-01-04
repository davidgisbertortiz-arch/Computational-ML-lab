"""Variance reduction techniques for Monte Carlo."""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VarianceReductionResult:
    """Results from variance-reduced MC estimation."""
    estimate: float
    std_error: float
    variance: float
    n_samples: int
    variance_reduction_factor: Optional[float] = None  # Compared to naive MC


class ImportanceSampler:
    """
    Importance sampling for variance reduction.
    
    Instead of sampling from p(x), sample from q(x):
        E_p[f(x)] = E_q[f(x) * p(x)/q(x)]
    
    Choose q(x) to reduce variance of f(x) * w(x) where w(x) = p(x)/q(x).
    
    Example:
        >>> # Estimate tail probability P(X > 3) for X ~ N(0,1)
        >>> target = lambda x: (x > 3).astype(float)
        >>> # Use N(3, 1) as proposal (shifts mass to tail)
        >>> sampler = ImportanceSampler(seed=42)
        >>> result = sampler.estimate(
        ...     target, 
        ...     proposal_sample=lambda n: np.random.normal(3, 1, n),
        ...     log_target_density=lambda x: -0.5*x**2 - 0.5*np.log(2*np.pi),
        ...     log_proposal_density=lambda x: -0.5*(x-3)**2 - 0.5*np.log(2*np.pi),
        ...     n_samples=10000
        ... )
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def estimate(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        proposal_sample: Callable[[int], np.ndarray],
        log_target_density: Callable[[np.ndarray], np.ndarray],
        log_proposal_density: Callable[[np.ndarray], np.ndarray],
        n_samples: int = 10000,
    ) -> VarianceReductionResult:
        """
        Estimate E[f(x)] using importance sampling.
        
        Args:
            func: Function to estimate expectation of
            proposal_sample: Function that samples from q(x)
            log_target_density: log p(x)
            log_proposal_density: log q(x)
            n_samples: Number of samples
            
        Returns:
            VarianceReductionResult with estimate
        """
        # Sample from proposal
        x = proposal_sample(n_samples)
        
        # Compute importance weights: w = p(x) / q(x)
        log_weights = log_target_density(x) - log_proposal_density(x)
        
        # Normalize weights (for numerical stability)
        log_weights = log_weights - np.max(log_weights)
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights) * n_samples  # Un-normalize for variance
        
        # Evaluate function
        f_x = func(x)
        
        # Weighted estimate
        estimate = np.mean(weights * f_x)
        
        # Variance of weighted estimator
        variance = np.var(weights * f_x, ddof=1)
        std_error = np.sqrt(variance / n_samples)
        
        return VarianceReductionResult(
            estimate=estimate,
            std_error=std_error,
            variance=variance,
            n_samples=n_samples,
        )


class ControlVariates:
    """
    Control variates for variance reduction.
    
    Use correlated variable with known expectation:
        f*(x) = f(x) - c * (g(x) - E[g])
    
    Optimal c = Cov(f, g) / Var(g) minimizes variance.
    
    Example:
        >>> # Estimate E[X^2] for X ~ U(0,1), using control variate g(X) = X
        >>> cv = ControlVariates(seed=42)
        >>> result = cv.estimate(
        ...     func=lambda x: x**2,
        ...     control=lambda x: x,
        ...     control_mean=0.5,  # E[X] for uniform
        ...     sampler=lambda n: np.random.uniform(0, 1, n),
        ...     n_samples=10000
        ... )
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def estimate(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        control: Callable[[np.ndarray], np.ndarray],
        control_mean: float,
        sampler: Callable[[int], np.ndarray],
        n_samples: int = 10000,
        c: Optional[float] = None,
    ) -> VarianceReductionResult:
        """
        Estimate E[f(x)] using control variates.
        
        Args:
            func: Function to estimate expectation of
            control: Control variate g(x) with known expectation
            control_mean: E[g(x)]
            sampler: Function that samples from target distribution
            n_samples: Number of samples
            c: Coefficient (if None, use optimal c)
            
        Returns:
            VarianceReductionResult
        """
        # Sample
        x = sampler(n_samples)
        
        # Evaluate functions
        f_x = func(x)
        g_x = control(x)
        
        # Compute optimal c if not provided
        if c is None:
            cov_fg = np.cov(f_x, g_x)[0, 1]
            var_g = np.var(g_x, ddof=1)
            c = cov_fg / var_g if var_g > 1e-10 else 0.0
        
        # Control variate estimator
        f_star = f_x - c * (g_x - control_mean)
        
        estimate = np.mean(f_star)
        variance = np.var(f_star, ddof=1)
        std_error = np.sqrt(variance / n_samples)
        
        # Compute variance reduction factor
        var_naive = np.var(f_x, ddof=1)
        vrf = var_naive / variance if variance > 1e-10 else float('inf')
        
        return VarianceReductionResult(
            estimate=estimate,
            std_error=std_error,
            variance=variance,
            n_samples=n_samples,
            variance_reduction_factor=vrf,
        )


def antithetic_sampling(
    func: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    n_pairs: int = 5000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Antithetic variates: use (X, -X) or (X, 1-X) for variance reduction.
    
    For symmetric distributions, generate X and use -X as antithetic pair.
    
    Args:
        func: Function to estimate expectation of
        sampler: Samples from symmetric distribution
        n_pairs: Number of antithetic pairs
        seed: Random seed
        
    Returns:
        (estimate, std_error)
    """
    rng = np.random.default_rng(seed)
    
    # Generate samples
    x1 = sampler(n_pairs)
    x2 = -x1  # Antithetic pairs
    
    # Evaluate
    f1 = func(x1)
    f2 = func(x2)
    
    # Average within pairs (reduces variance)
    pair_means = (f1 + f2) / 2
    
    estimate = np.mean(pair_means)
    std_error = np.std(pair_means, ddof=1) / np.sqrt(n_pairs)
    
    return estimate, std_error


def compare_variance_reduction(
    func: Callable[[np.ndarray], np.ndarray],
    naive_sampler: Callable[[int], np.ndarray],
    methods: dict,
    n_samples: int = 10000,
    n_runs: int = 50,
    seed: Optional[int] = None,
) -> dict:
    """
    Compare variance reduction methods.
    
    Args:
        func: Target function
        naive_sampler: Sampling from target distribution
        methods: Dict of {name: sampler_func} for each VR method
        n_samples: Samples per run
        n_runs: Number of independent runs
        seed: Random seed
        
    Returns:
        Dictionary with variance and VRF for each method
    """
    rng = np.random.default_rng(seed)
    
    results = {}
    
    # Naive MC baseline
    naive_estimates = []
    for _ in range(n_runs):
        x = naive_sampler(n_samples)
        naive_estimates.append(np.mean(func(x)))
    
    naive_var = np.var(naive_estimates, ddof=1)
    results['naive'] = {
        'variance': naive_var,
        'vrf': 1.0,
        'estimates': naive_estimates,
    }
    
    # VR methods
    for name, method_func in methods.items():
        estimates = []
        for _ in range(n_runs):
            est = method_func(n_samples)
            estimates.append(est)
        
        var = np.var(estimates, ddof=1)
        vrf = naive_var / var if var > 1e-10 else float('inf')
        
        results[name] = {
            'variance': var,
            'vrf': vrf,
            'estimates': estimates,
        }
    
    return results
