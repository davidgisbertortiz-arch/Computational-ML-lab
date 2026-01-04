"""Rare event estimation techniques."""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class RareEventResult:
    """Results from rare event estimation."""
    probability: float
    std_error: float
    relative_error: float  # std_error / probability
    n_samples: int
    n_events: int  # Number of rare events observed
    method: str


class RareEventEstimator:
    """
    Estimator for rare event probabilities.
    
    Rare events: P(event) << 1, naive MC has high relative error.
    
    Methods:
        1. Naive MC: simple but needs many samples
        2. Importance sampling: shift distribution to rare region
        3. Splitting: multi-stage sampling toward rare region
    
    Example:
        >>> # Estimate P(X > 4) for X ~ N(0,1)
        >>> estimator = RareEventEstimator(seed=42)
        >>> # Naive
        >>> result_naive = estimator.estimate_naive(
        ...     event=lambda x: x > 4,
        ...     sampler=lambda n: np.random.normal(0, 1, n),
        ...     n_samples=100000
        ... )
        >>> # Importance sampling
        >>> result_is = estimator.estimate_importance_sampling(
        ...     event=lambda x: x > 4,
        ...     proposal_sampler=lambda n: np.random.normal(4, 1, n),
        ...     log_target_density=lambda x: -0.5*x**2,
        ...     log_proposal_density=lambda x: -0.5*(x-4)**2,
        ...     n_samples=10000
        ... )
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def estimate_naive(
        self,
        event: Callable[[np.ndarray], np.ndarray],
        sampler: Callable[[int], np.ndarray],
        n_samples: int = 100000,
    ) -> RareEventResult:
        """
        Naive MC estimation: P(event) ≈ (1/N) Σ I(event(X_i)).
        
        Args:
            event: Function returning boolean array (True if event occurs)
            sampler: Samples from target distribution
            n_samples: Number of samples
            
        Returns:
            RareEventResult
        """
        x = sampler(n_samples)
        indicators = event(x).astype(float)
        
        p_hat = np.mean(indicators)
        variance = p_hat * (1 - p_hat)  # Bernoulli variance
        std_error = np.sqrt(variance / n_samples)
        
        relative_error = std_error / p_hat if p_hat > 1e-10 else float('inf')
        n_events = int(np.sum(indicators))
        
        return RareEventResult(
            probability=p_hat,
            std_error=std_error,
            relative_error=relative_error,
            n_samples=n_samples,
            n_events=n_events,
            method='naive_mc',
        )
    
    def estimate_importance_sampling(
        self,
        event: Callable[[np.ndarray], np.ndarray],
        proposal_sampler: Callable[[int], np.ndarray],
        log_target_density: Callable[[np.ndarray], np.ndarray],
        log_proposal_density: Callable[[np.ndarray], np.ndarray],
        n_samples: int = 10000,
    ) -> RareEventResult:
        """
        Importance sampling for rare events.
        
        P(event) = E_q[I(event) * p(x)/q(x)]
        
        Choose q(x) concentrated in rare region.
        
        Args:
            event: Event indicator function
            proposal_sampler: Samples from q(x)
            log_target_density: log p(x)
            log_proposal_density: log q(x)
            n_samples: Number of samples
            
        Returns:
            RareEventResult
        """
        # Sample from proposal
        x = proposal_sampler(n_samples)
        
        # Importance weights: w(x) = p(x) / q(x)
        log_weights = log_target_density(x) - log_proposal_density(x)
        weights = np.exp(log_weights)
        
        # Indicators
        indicators = event(x).astype(float)
        
        # Weighted estimate: E_p[I] = E_q[I * w]
        p_hat = np.mean(weights * indicators)
        
        # Variance of weighted estimator
        variance = np.var(weights * indicators, ddof=1)
        std_error = np.sqrt(variance / n_samples)
        
        relative_error = std_error / p_hat if p_hat > 1e-10 else float('inf')
        n_events = int(np.sum(indicators))
        
        return RareEventResult(
            probability=p_hat,
            std_error=std_error,
            relative_error=relative_error,
            n_samples=n_samples,
            n_events=n_events,
            method='importance_sampling',
        )


def tail_probability(
    threshold: float,
    distribution: str = 'normal',
    loc: float = 0.0,
    scale: float = 1.0,
    n_samples: int = 100000,
    method: str = 'naive',
    seed: Optional[int] = None,
) -> RareEventResult:
    """
    Estimate tail probability P(X > threshold) for standard distributions.
    
    Args:
        threshold: Tail threshold
        distribution: 'normal', 'exponential', etc.
        loc: Location parameter
        scale: Scale parameter
        n_samples: Number of samples
        method: 'naive' or 'importance'
        seed: Random seed
        
    Returns:
        RareEventResult
    """
    estimator = RareEventEstimator(seed=seed)
    
    # Define event
    event = lambda x: x > threshold
    
    # Create RNG once to avoid repeated sequences
    rng = np.random.default_rng(seed)
    
    if distribution == 'normal':
        # Target: N(loc, scale²)
        sampler = lambda n: rng.normal(loc, scale, n)
        log_target = lambda x: stats.norm.logpdf(x, loc, scale)
        
        if method == 'importance':
            # Proposal: N(threshold, scale²) - shifted to tail
            proposal_sampler = lambda n: rng.normal(threshold, scale, n)
            log_proposal = lambda x: stats.norm.logpdf(x, threshold, scale)
            
            return estimator.estimate_importance_sampling(
                event, proposal_sampler, log_target, log_proposal, n_samples
            )
        else:
            return estimator.estimate_naive(event, sampler, n_samples)
    
    elif distribution == 'exponential':
        sampler = lambda n: rng.exponential(scale, n) + loc
        
        if method == 'importance':
            # Proposal: Exponential with higher rate (concentrates in tail)
            new_scale = scale / 2  # Higher rate
            proposal_sampler = lambda n: rng.exponential(new_scale, n) + loc
            log_target = lambda x: stats.expon.logpdf(x - loc, scale=scale)
            log_proposal = lambda x: stats.expon.logpdf(x - loc, scale=new_scale)
            
            return estimator.estimate_importance_sampling(
                event, proposal_sampler, log_target, log_proposal, n_samples
            )
        else:
            return estimator.estimate_naive(event, sampler, n_samples)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def conditional_monte_carlo(
    func: Callable[[np.ndarray], np.ndarray],
    condition: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    n_samples: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Conditional MC: E[f(X) | condition(X) = True].
    
    Useful for rare events: estimate expectation conditioned on event.
    
    Args:
        func: Function to average
        condition: Boolean condition
        sampler: Sampler from distribution
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        (estimate, std_error)
    """
    rng = np.random.default_rng(seed)
    
    # Sample until we get enough conditional samples
    conditional_values = []
    attempts = 0
    max_attempts = n_samples * 100  # Avoid infinite loop
    
    while len(conditional_values) < n_samples and attempts < max_attempts:
        batch = sampler(min(1000, max_attempts - attempts))
        mask = condition(batch)
        conditional_values.extend(func(batch[mask]))
        attempts += len(batch)
    
    if len(conditional_values) < n_samples:
        raise RuntimeError(f"Could not generate {n_samples} conditional samples in {max_attempts} attempts")
    
    conditional_values = np.array(conditional_values[:n_samples])
    
    estimate = np.mean(conditional_values)
    std_error = np.std(conditional_values, ddof=1) / np.sqrt(n_samples)
    
    return estimate, std_error


def adaptive_sampling(
    event: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    target_events: int = 100,
    max_samples: int = 1_000_000,
    seed: Optional[int] = None,
) -> RareEventResult:
    """
    Adaptive sampling: continue until observing target number of events.
    
    Useful when event probability is unknown.
    
    Args:
        event: Event indicator
        sampler: Sampler
        target_events: Desired number of events
        max_samples: Maximum samples to draw
        seed: Random seed
        
    Returns:
        RareEventResult
    """
    rng = np.random.default_rng(seed)
    
    total_samples = 0
    total_events = 0
    
    while total_events < target_events and total_samples < max_samples:
        batch_size = min(10000, max_samples - total_samples)
        x = sampler(batch_size)
        indicators = event(x)
        
        total_samples += batch_size
        total_events += np.sum(indicators)
    
    if total_events < target_events:
        raise RuntimeError(f"Could not observe {target_events} events in {max_samples} samples")
    
    p_hat = total_events / total_samples
    variance = p_hat * (1 - p_hat)
    std_error = np.sqrt(variance / total_samples)
    relative_error = std_error / p_hat if p_hat > 1e-10 else float('inf')
    
    return RareEventResult(
        probability=p_hat,
        std_error=std_error,
        relative_error=relative_error,
        n_samples=total_samples,
        n_events=total_events,
        method='adaptive',
    )
