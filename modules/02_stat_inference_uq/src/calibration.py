"""Calibration metrics and temperature scaling for probabilistic predictions.

Implements reliability diagrams, Expected Calibration Error (ECE), and
temperature scaling for improving classifier calibration.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from dataclasses import dataclass


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    ax: Optional[plt.Axes] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute and plot reliability diagram (calibration plot).
    
    Bins predictions by confidence and compares predicted probability
    vs empirical frequency of positive class.
    
    Args:
        y_true: True binary labels (n_samples,) in {0, 1}
        y_prob: Predicted probabilities (n_samples,) in [0, 1]
        n_bins: Number of bins for grouping predictions
        strategy: Binning strategy - 'uniform' or 'quantile'
        ax: Matplotlib axes for plotting (creates new if None)
        
    Returns:
        bin_centers: Mean predicted probability in each bin
        bin_accuracies: Empirical accuracy in each bin
        bin_counts: Number of samples in each bin
        
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_prob = np.array([0.1, 0.3, 0.6, 0.7, 0.9])
        >>> centers, accs, counts = reliability_diagram(y_true, y_prob, n_bins=3)
    """
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_prob {y_prob.shape}")
    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("y_true must be binary (0 or 1)")
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError("y_prob must be in [0, 1]")
        
    # Create bins
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bins[0] = 0.0  # Ensure endpoints
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    # Assign samples to bins
    bin_indices = np.digitize(y_prob, bins[1:-1])  # Returns 0 to n_bins-1
    
    # Compute statistics per bin
    bin_centers = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    
    for i in range(n_bins):
        mask = bin_indices == i
        bin_counts[i] = np.sum(mask)
        
        if bin_counts[i] > 0:
            bin_centers[i] = np.mean(y_prob[mask])
            bin_accuracies[i] = np.mean(y_true[mask])
        else:
            # Empty bin - use bin center
            bin_centers[i] = (bins[i] + bins[i + 1]) / 2
            bin_accuracies[i] = np.nan
            
    # Plot if axes provided
    if ax is not None:
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
        
        # Reliability diagram (gap plot)
        valid_bins = bin_counts > 0
        ax.bar(
            bin_centers[valid_bins],
            bin_accuracies[valid_bins],
            width=1.0 / n_bins * 0.9,
            alpha=0.6,
            color="steelblue",
            edgecolor="black",
            label="Model calibration",
        )
        
        # Add gap indicators (distance from perfect calibration)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                ax.plot(
                    [bin_centers[i], bin_centers[i]],
                    [bin_centers[i], bin_accuracies[i]],
                    "r-",
                    alpha=0.5,
                )
        
        ax.set_xlabel("Predicted Probability", fontsize=12)
        ax.set_ylabel("Empirical Accuracy", fontsize=12)
        ax.set_title("Reliability Diagram", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
    return bin_centers, bin_accuracies, bin_counts


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE = Σ (n_b / n) |acc_b - conf_b|
    
    where n_b is number of samples in bin b, acc_b is empirical accuracy,
    and conf_b is mean predicted confidence.
    
    Args:
        y_true: True binary labels (n_samples,)
        y_prob: Predicted probabilities (n_samples,)
        n_bins: Number of bins
        strategy: Binning strategy - 'uniform' or 'quantile'
        
    Returns:
        ece: Expected Calibration Error (scalar in [0, 1])
        
    Example:
        >>> # Perfect calibration
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        >>> expected_calibration_error(y_true, y_prob)
        0.0
        
        >>> # Poor calibration (overconfident)
        >>> y_prob_bad = np.array([0.0, 0.1, 0.9, 1.0])
        >>> expected_calibration_error(y_true, y_prob_bad)
        > 0.0
    """
    bin_centers, bin_accuracies, bin_counts = reliability_diagram(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, ax=None
    )
    
    # Weighted average of calibration gaps
    n_total = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            gap = np.abs(bin_accuracies[i] - bin_centers[i])
            ece += (bin_counts[i] / n_total) * gap
            
    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE = max_b |acc_b - conf_b|
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        strategy: Binning strategy
        
    Returns:
        mce: Maximum Calibration Error
    """
    bin_centers, bin_accuracies, bin_counts = reliability_diagram(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, ax=None
    )
    
    # Max gap across bins with data
    valid_bins = bin_counts > 0
    if not np.any(valid_bins):
        return 0.0
        
    gaps = np.abs(bin_accuracies[valid_bins] - bin_centers[valid_bins])
    mce = np.max(gaps)
    
    return mce


@dataclass
class TemperatureScaling:
    """
    Temperature scaling for post-hoc calibration of classifier probabilities.
    
    Applies a single temperature parameter T to logits:
        p_calibrated = softmax(logits / T)
        
    For binary classification:
        p_calibrated = sigmoid(logit / T)
        
    T is learned by minimizing negative log-likelihood on validation set.
    
    Args:
        n_classes: Number of classes (default: 2 for binary)
        max_iter: Maximum optimization iterations
        lr: Learning rate for gradient descent
        tol: Convergence tolerance
        
    Attributes:
        temperature_: Learned temperature parameter (T > 0)
        
    Example:
        >>> # Train classifier and get logits on validation set
        >>> logits_val = model.predict_log_proba(X_val)
        >>> ts = TemperatureScaling()
        >>> ts.fit(logits_val, y_val)
        >>> # Apply to test set
        >>> logits_test = model.predict_log_proba(X_test)
        >>> probs_calibrated = ts.predict_proba(logits_test)
    """
    
    n_classes: int = 2
    max_iter: int = 100
    lr: float = 0.01
    tol: float = 1e-5
    
    def __post_init__(self):
        """Initialize temperature."""
        self.temperature_: float = 1.0
        
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """
        Fit temperature parameter on validation data.
        
        Args:
            logits: Model logits (n_samples,) for binary or (n_samples, n_classes)
            y_true: True labels (n_samples,) in {0, 1, ..., n_classes-1}
            
        Returns:
            self: Fitted scaler
        """
        if self.n_classes == 2:
            # Binary case: logits shape (n_samples,)
            if logits.ndim != 1:
                raise ValueError(f"Binary logits must be 1D, got shape {logits.shape}")
            return self._fit_binary(logits, y_true)
        else:
            # Multi-class: logits shape (n_samples, n_classes)
            if logits.ndim != 2 or logits.shape[1] != self.n_classes:
                raise ValueError(f"Expected shape (n_samples, {self.n_classes}), got {logits.shape}")
            return self._fit_multiclass(logits, y_true)
            
    def _fit_binary(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """Fit temperature for binary classification."""
        # Optimize temperature via gradient descent on NLL
        T = 1.0
        
        for iteration in range(self.max_iter):
            # Scaled probabilities
            probs = 1 / (1 + np.exp(-logits / T))
            
            # Negative log-likelihood
            nll = -np.mean(y_true * np.log(probs + 1e-10) + (1 - y_true) * np.log(1 - probs + 1e-10))
            
            # Gradient w.r.t. T
            # d(NLL)/dT = (1/T^2) * Σ logit_i * (p_i - y_i)
            grad = np.mean(logits * (probs - y_true)) / (T ** 2)
            
            # Update
            T_new = T - self.lr * grad
            T_new = max(T_new, 0.01)  # Ensure T > 0
            
            # Check convergence
            if abs(T_new - T) < self.tol:
                T = T_new
                break
                
            T = T_new
            
        self.temperature_ = T
        return self
        
    def _fit_multiclass(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaling":
        """Fit temperature for multi-class classification."""
        T = 1.0
        
        for iteration in range(self.max_iter):
            # Scaled softmax
            scaled_logits = logits / T
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # NLL
            nll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-10))
            
            # Gradient (simplified for efficiency)
            one_hot = np.zeros_like(logits)
            one_hot[np.arange(len(y_true)), y_true] = 1
            grad = np.mean(np.sum(logits * (probs - one_hot), axis=1)) / (T ** 2)
            
            T_new = T - self.lr * grad
            T_new = max(T_new, 0.01)
            
            if abs(T_new - T) < self.tol:
                T = T_new
                break
                
            T = T_new
            
        self.temperature_ = T
        return self
        
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits (same shape as fit)
            
        Returns:
            probs: Calibrated probabilities
        """
        if self.n_classes == 2:
            # Binary
            return 1 / (1 + np.exp(-logits / self.temperature_))
        else:
            # Multi-class softmax
            scaled = logits / self.temperature_
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
