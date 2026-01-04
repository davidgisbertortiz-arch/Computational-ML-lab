"""MCMC basics: Metropolis-Hastings sampler with diagnostics.

Implements the Metropolis-Hastings algorithm for sampling from arbitrary
distributions, with comprehensive diagnostics (acceptance rate, trace plots,
autocorrelation, effective sample size).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MetropolisHastings:
    """
    Metropolis-Hastings MCMC sampler.
    
    Samples from target distribution π(x) using proposal distribution q(x'|x).
    Uses random walk proposal: x' = x + ε, ε ~ N(0, σ^2 I).
    
    Args:
        log_prob_fn: Log probability function log π(x)
        proposal_std: Proposal standard deviation (σ)
        n_samples: Number of samples to generate
        n_burn: Burn-in samples to discard
        thin: Thinning factor (keep every thin-th sample)
        random_state: Random seed for reproducibility
        
    Attributes:
        samples_: Generated samples (n_samples - n_burn, n_dim)
        accept_rate_: Acceptance rate
        log_probs_: Log probabilities of samples
        
    Example:
        >>> # Sample from 2D Gaussian
        >>> def log_prob(x):
        ...     return -0.5 * np.sum(x**2)
        >>> sampler = MetropolisHastings(log_prob_fn=log_prob, proposal_std=1.0)
        >>> samples = sampler.sample(x0=np.zeros(2), n_samples=5000)
        >>> print(f"Acceptance rate: {sampler.accept_rate_:.2%}")
    """
    
    log_prob_fn: Callable[[np.ndarray], float]
    proposal_std: float = 1.0
    n_samples: int = 10000
    n_burn: int = 1000
    thin: int = 1
    random_state: Optional[int] = None
    
    def __post_init__(self):
        """Initialize state."""
        self.samples_: Optional[np.ndarray] = None
        self.accept_rate_: float = 0.0
        self.log_probs_: Optional[np.ndarray] = None
        self.rng_ = np.random.default_rng(self.random_state)
        
    def sample(
        self,
        x0: np.ndarray,
        n_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Run Metropolis-Hastings sampling.
        
        Args:
            x0: Initial state (n_dim,)
            n_samples: Number of samples (overrides constructor)
            verbose: Print progress every 1000 iterations
            
        Returns:
            samples: MCMC samples after burn-in and thinning
            
        Algorithm:
            1. Start at x0
            2. Propose x' = x + ε, ε ~ N(0, σ^2 I)
            3. Compute acceptance ratio: α = min(1, π(x') / π(x))
            4. Accept x' with probability α, else keep x
            5. Repeat
        """
        if n_samples is None:
            n_samples = self.n_samples
            
        n_dim = x0.shape[0]
        n_total = n_samples + self.n_burn
        
        # Storage (keep all samples for diagnostics, thin later)
        chain = np.zeros((n_total, n_dim))
        log_probs = np.zeros(n_total)
        accepts = np.zeros(n_total, dtype=bool)
        
        # Initialize
        x_current = x0.copy()
        log_prob_current = self.log_prob_fn(x_current)
        
        chain[0] = x_current
        log_probs[0] = log_prob_current
        
        # Main MCMC loop
        for i in range(1, n_total):
            # Propose new state (random walk)
            x_proposed = x_current + self.rng_.normal(0, self.proposal_std, size=n_dim)
            log_prob_proposed = self.log_prob_fn(x_proposed)
            
            # Acceptance ratio (in log space for numerical stability)
            log_alpha = log_prob_proposed - log_prob_current
            
            # Accept/reject
            if log_alpha >= 0 or self.rng_.uniform() < np.exp(log_alpha):
                # Accept
                x_current = x_proposed
                log_prob_current = log_prob_proposed
                accepts[i] = True
            else:
                # Reject (keep current state)
                accepts[i] = False
                
            chain[i] = x_current
            log_probs[i] = log_prob_current
            
            if verbose and (i + 1) % 1000 == 0:
                current_accept_rate = np.mean(accepts[:i+1])
                print(f"Iteration {i+1}/{n_total} | Accept rate: {current_accept_rate:.2%}")
                
        # Compute statistics
        self.accept_rate_ = np.mean(accepts)
        
        # Discard burn-in and apply thinning
        chain_postburn = chain[self.n_burn:]
        log_probs_postburn = log_probs[self.n_burn:]
        
        if self.thin > 1:
            indices = np.arange(0, len(chain_postburn), self.thin)
            self.samples_ = chain_postburn[indices]
            self.log_probs_ = log_probs_postburn[indices]
        else:
            self.samples_ = chain_postburn
            self.log_probs_ = log_probs_postburn
            
        return self.samples_
        
    def get_diagnostics(self) -> dict:
        """
        Compute MCMC diagnostics.
        
        Returns:
            diagnostics: Dictionary with acceptance rate, ESS, etc.
        """
        if self.samples_ is None:
            raise ValueError("No samples yet. Call sample() first.")
            
        diag = MCMCDiagnostics(self.samples_)
        
        return {
            "acceptance_rate": self.accept_rate_,
            "ess": diag.effective_sample_size(),
            "autocorr_time": diag.integrated_autocorr_time(),
            "mean": diag.mean(),
            "std": diag.std(),
        }


@dataclass
class MCMCDiagnostics:
    """
    Diagnostic tools for MCMC chains.
    
    Args:
        samples: MCMC samples (n_samples, n_dim)
        
    Methods:
        - autocorrelation(): Compute autocorrelation function
        - effective_sample_size(): ESS accounting for correlation
        - trace_plot(): Visualize chain evolution
        - marginal_histograms(): 1D and 2D marginals
    """
    
    samples: np.ndarray
    
    def __post_init__(self):
        """Validate inputs."""
        if self.samples.ndim != 2:
            raise ValueError(f"samples must be 2D, got shape {self.samples.shape}")
            
    def mean(self) -> np.ndarray:
        """Sample mean."""
        return np.mean(self.samples, axis=0)
        
    def std(self) -> np.ndarray:
        """Sample standard deviation."""
        return np.std(self.samples, axis=0, ddof=1)
        
    def autocorrelation(self, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute autocorrelation function for each dimension.
        
        ρ(k) = Corr(X_t, X_{t+k})
        
        Args:
            max_lag: Maximum lag (default: n_samples // 2)
            
        Returns:
            acf: Autocorrelation (max_lag, n_dim)
        """
        n_samples, n_dim = self.samples.shape
        
        if max_lag is None:
            max_lag = n_samples // 2
        max_lag = min(max_lag, n_samples - 1)
        
        acf = np.zeros((max_lag, n_dim))
        
        # Center the samples
        centered = self.samples - self.mean()
        var = np.var(self.samples, axis=0, ddof=1)
        
        for lag in range(max_lag):
            if lag == 0:
                acf[lag] = 1.0
            else:
                # ρ(k) = Cov(X_t, X_{t+k}) / Var(X_t)
                cov = np.mean(centered[:-lag] * centered[lag:], axis=0)
                acf[lag] = cov / var
                
        return acf
        
    def integrated_autocorr_time(self, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute integrated autocorrelation time τ_int.
        
        τ_int = 1 + 2 * Σ_{k=1}^∞ ρ(k)
        
        This measures how many correlated samples are needed to get
        one "effective" independent sample.
        
        Args:
            max_lag: Maximum lag for summation
            
        Returns:
            tau_int: Integrated autocorr time per dimension
        """
        acf = self.autocorrelation(max_lag=max_lag)
        
        # Sum until ACF becomes negligible (< 0.05)
        tau_int = np.ones(self.samples.shape[1])
        
        for d in range(self.samples.shape[1]):
            # Find cutoff where ACF drops below threshold
            cutoff = np.where(np.abs(acf[:, d]) < 0.05)[0]
            if len(cutoff) > 0:
                k_max = cutoff[0]
            else:
                k_max = len(acf)
                
            tau_int[d] = 1 + 2 * np.sum(acf[:k_max, d])
            
        return tau_int
        
    def effective_sample_size(self) -> np.ndarray:
        """
        Compute effective sample size ESS = N / τ_int.
        
        Returns:
            ess: Effective sample size per dimension
        """
        tau_int = self.integrated_autocorr_time()
        ess = self.samples.shape[0] / tau_int
        return ess
        
    def trace_plot(self, dims: Optional[list[int]] = None, figsize: tuple = (12, 6)) -> plt.Figure:
        """
        Plot trace of MCMC chain over iterations.
        
        Args:
            dims: Dimensions to plot (default: all)
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        if dims is None:
            dims = list(range(min(4, self.samples.shape[1])))  # Max 4 dims
            
        n_dims = len(dims)
        fig, axes = plt.subplots(n_dims, 1, figsize=figsize, squeeze=False)
        
        for i, d in enumerate(dims):
            ax = axes[i, 0]
            ax.plot(self.samples[:, d], alpha=0.7, linewidth=0.5)
            ax.set_ylabel(f"$x_{d}$", fontsize=11)
            ax.grid(alpha=0.3)
            
            if i == n_dims - 1:
                ax.set_xlabel("Iteration", fontsize=11)
            else:
                ax.set_xticklabels([])
                
        fig.suptitle("MCMC Trace Plot", fontsize=14, fontweight="bold")
        fig.tight_layout()
        
        return fig
        
    def marginal_histograms(
        self,
        dims: Optional[list[int]] = None,
        true_dist: Optional[Callable] = None,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """
        Plot marginal histograms and compare with true distribution.
        
        Args:
            dims: Dimensions to plot (default: first 2)
            true_dist: Optional true density function for comparison
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        if dims is None:
            dims = [0, 1] if self.samples.shape[1] >= 2 else [0]
            
        n_dims = len(dims)
        
        if n_dims == 1:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        elif n_dims == 2:
            fig, axes = plt.subplots(1, 3, figsize=figsize)
        else:
            # Grid layout
            n_cols = 3
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()
            
        # 1D marginals
        for i, d in enumerate(dims):
            if n_dims == 2 and i < 2:
                ax = axes[i]
            else:
                ax = axes[i] if n_dims != 2 else None
                
            if ax is not None:
                ax.hist(self.samples[:, d], bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="black")
                
                # Overlay true distribution if provided
                if true_dist is not None:
                    x_range = np.linspace(self.samples[:, d].min(), self.samples[:, d].max(), 200)
                    # Assume true_dist can evaluate marginals
                    ax.plot(x_range, [true_dist(np.array([x])) for x in x_range], "r-", linewidth=2, label="True")
                    ax.legend()
                    
                ax.set_xlabel(f"$x_{d}$", fontsize=11)
                ax.set_ylabel("Density", fontsize=11)
                ax.set_title(f"Marginal: dim {d}", fontsize=12)
                ax.grid(alpha=0.3)
                
        # 2D joint distribution (if 2+ dims)
        if n_dims >= 2:
            ax_joint = axes[2] if n_dims == 2 else axes[-1]
            ax_joint.hexbin(self.samples[:, dims[0]], self.samples[:, dims[1]], gridsize=30, cmap="Blues")
            ax_joint.set_xlabel(f"$x_{dims[0]}$", fontsize=11)
            ax_joint.set_ylabel(f"$x_{dims[1]}$", fontsize=11)
            ax_joint.set_title(f"Joint: dims {dims[0]}, {dims[1]}", fontsize=12)
            
        fig.suptitle("MCMC Marginal Distributions", fontsize=14, fontweight="bold")
        fig.tight_layout()
        
        return fig
