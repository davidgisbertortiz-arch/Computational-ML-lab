"""Evaluation metrics, calibration, and explainability.

Implements comprehensive evaluation suite:
- Classification metrics: AUC-ROC, AP, precision/recall
- Calibration: ECE, Brier score, reliability diagrams
- Explainability: Permutation importance, optional SHAP
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities for positive class
        y_pred: Predicted labels (computed from threshold if None)
        threshold: Decision threshold
        
    Returns:
        Dictionary of metric name -> value
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= threshold).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'ece': expected_calibration_error(y_true, y_pred_proba),
    }
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
    })
    
    # Recall at precision threshold (operational metric)
    metrics['recall_at_p95'] = recall_at_precision_threshold(
        y_true, y_pred_proba, target_precision=0.95
    )
    
    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies.
    Lower is better (0 = perfect calibration).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for discretization
        
    Returns:
        ECE score
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        
        bin_confidence = y_proba[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        
        ece += bin_weight * abs(bin_confidence - bin_accuracy)
    
    return ece


def recall_at_precision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: float = 0.95,
) -> float:
    """Compute recall at specified precision threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_precision: Minimum acceptable precision
        
    Returns:
        Maximum recall achievable at target precision
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    valid_recalls = recalls[precisions >= target_precision]
    return valid_recalls.max() if len(valid_recalls) > 0 else 0.0


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'Model (AP = {ap:.3f})')
    ax.axhline(y_true.mean(), color='k', linestyle='--', 
               linewidth=1, label=f'Baseline ({y_true.mean():.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Reliability Diagram",
) -> plt.Figure:
    """Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidences.append(y_proba[mask].mean())
            bin_accuracies.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
    
    ece = expected_calibration_error(y_true, y_proba, n_bins)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax1.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, 
             markersize=8, label=f'Model (ECE = {ece:.3f})')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('True Frequency', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predictions
    ax2.hist(y_proba, bins=n_bins, alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance.
    
    Args:
        model: Trained model
        X: Features
        y: Labels
        feature_names: List of feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed
        
    Returns:
        DataFrame with feature importances (sorted)
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    })
    
    return df.sort_values('importance_mean', ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Plot feature importance with error bars.
    
    Args:
        importance_df: DataFrame from compute_permutation_importance
        top_k: Number of top features to plot
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    df_plot = importance_df.head(top_k)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(df_plot))
    
    ax.barh(y_pos, df_plot['importance_mean'], 
            xerr=df_plot['importance_std'],
            align='center', alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (decrease in AUC-ROC)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3, axis='x')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    """Test evaluation functions."""
    print("Testing evaluation metrics...")
    
    # Generate dummy data
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 1000)
    y_proba = np.clip(y_true + np.random.normal(0, 0.3, 1000), 0, 1)
    
    metrics = compute_metrics(y_true, y_proba)
    print("✅ Metrics computed:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.3f}")
    
    # Test ECE
    ece = expected_calibration_error(y_true, y_proba)
    print(f"✅ ECE: {ece:.3f}")
