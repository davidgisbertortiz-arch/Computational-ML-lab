#!/usr/bin/env python
"""UQ in Classification: Comparing Uncertainty Quantification Methods.

This mini-project compares three uncertainty quantification approaches
for classification:
1. Raw predict_proba (baseline)
2. Temperature scaling (post-hoc calibration)
3. Bootstrap ensemble (epistemic uncertainty)

Evaluation metrics:
- Accuracy
- Negative Log-Likelihood (NLL)
- Brier Score
- Expected Calibration Error (ECE)

Generates a report with comparison table and diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
import json
from datetime import datetime

# sklearn imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Import from module utilities
import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
expected_calibration_error, TemperatureScaling, reliability_diagram = safe_import_from(
    '02_stat_inference_uq.src.calibration',
    'expected_calibration_error', 'TemperatureScaling', 'reliability_diagram'
)


@dataclass
class UQMetrics:
    """Container for UQ evaluation metrics."""
    accuracy: float
    nll: float
    brier_score: float
    ece: float
    
    def to_dict(self) -> dict:
        return {
            "Accuracy": f"{self.accuracy:.4f}",
            "NLL": f"{self.nll:.4f}",
            "Brier Score": f"{self.brier_score:.4f}",
            "ECE": f"{self.ece:.4f}",
        }


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute negative log-likelihood (cross-entropy loss).
    
    NLL = -1/N * sum(y_i * log(p_i) + (1-y_i) * log(1-p_i))
    
    Lower is better. Measures how well probabilities match labels.
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    nll = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(nll)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).
    
    Brier = 1/N * sum((p_i - y_i)^2)
    
    Lower is better. Proper scoring rule for probabilistic predictions.
    """
    return float(np.mean((y_prob - y_true) ** 2))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> UQMetrics:
    """Compute all UQ evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    return UQMetrics(
        accuracy=float(np.mean(y_pred == y_true)),
        nll=negative_log_likelihood(y_true, y_prob),
        brier_score=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins=10),
    )


class BootstrapEnsemble:
    """
    Bootstrap ensemble for uncertainty quantification.
    
    Trains multiple classifiers on bootstrap samples to capture
    epistemic (model) uncertainty through disagreement.
    """
    
    def __init__(
        self,
        base_estimator,
        n_estimators: int = 20,
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_: List = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapEnsemble":
        """Fit ensemble on bootstrap samples."""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        
        self.estimators_ = []
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            
            # Clone and fit estimator
            from sklearn.base import clone
            estimator = clone(self.base_estimator)
            estimator.set_params(random_state=self.random_state + i)
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities by averaging ensemble predictions.
        
        Returns average probability for positive class.
        """
        probas = np.array([est.predict_proba(X)[:, 1] for est in self.estimators_])
        return probas.mean(axis=0)
    
    def predict_proba_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Returns:
            mean_prob: Mean probability across ensemble
            std_prob: Standard deviation (epistemic uncertainty)
        """
        probas = np.array([est.predict_proba(X)[:, 1] for est in self.estimators_])
        return probas.mean(axis=0), probas.std(axis=0)


def generate_dataset(
    n_samples: int = 2000,
    n_features: int = 20,
    n_informative: int = 10,
    class_sep: float = 0.8,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset with train/val/test splits."""
    # Ensure n_informative + n_redundant < n_features (with safety margin)
    # Adjust n_informative if needed
    n_informative = min(n_informative, max(2, n_features - 3))
    n_redundant = max(0, min(2, n_features - n_informative - 1))
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        class_sep=class_sep,
        flip_y=0.1,  # Add label noise for realism
        random_state=random_state,
    )
    
    # Split: 60% train, 20% val (for calibration), 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_comparison(
    seed: int = 42,
    output_dir: Path = Path("modules/02_stat_inference_uq/reports/uq_classification"),
) -> Dict[str, UQMetrics]:
    """
    Run the full UQ comparison experiment.
    
    Returns:
        Dictionary mapping method name to UQMetrics
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UQ in Classification: Method Comparison")
    print("=" * 60)
    
    # Generate data
    print("\n📊 Generating synthetic dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_dataset(
        n_samples=2000, random_state=seed
    )
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Results storage
    results: Dict[str, UQMetrics] = {}
    proba_dict: Dict[str, np.ndarray] = {}
    
    # =========================================================================
    # Method 1: Raw predict_proba (Logistic Regression baseline)
    # =========================================================================
    print("\n🔹 Method 1: Raw predict_proba (Logistic Regression)")
    base_clf = LogisticRegression(random_state=seed, max_iter=1000)
    base_clf.fit(X_train, y_train)
    
    y_prob_raw = base_clf.predict_proba(X_test)[:, 1]
    results["Raw predict_proba"] = compute_metrics(y_test, y_prob_raw)
    proba_dict["Raw predict_proba"] = y_prob_raw
    print(f"  Accuracy: {results['Raw predict_proba'].accuracy:.4f}")
    print(f"  ECE: {results['Raw predict_proba'].ece:.4f}")
    
    # =========================================================================
    # Method 2: Temperature Scaling
    # =========================================================================
    print("\n🔹 Method 2: Temperature Scaling")
    
    # Get validation logits for calibration
    y_prob_val = base_clf.predict_proba(X_val)[:, 1]
    
    # Fit temperature scaling on validation set
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(y_val, y_prob_val)
    
    # Apply to test set
    y_prob_temp = temp_scaler.predict_proba(y_prob_raw)
    # Temperature scaling doesn't change predictions, only calibration
    # Use base classifier accuracy
    metrics_temp = compute_metrics(y_test, y_prob_temp)
    results["Temperature Scaling"] = UQMetrics(
        accuracy=results["Raw predict_proba"].accuracy,  # Same as base
        nll=metrics_temp.nll,
        brier_score=metrics_temp.brier_score,
        ece=metrics_temp.ece
    )
    proba_dict["Temperature Scaling"] = y_prob_temp
    print(f"  Optimal temperature: {temp_scaler.temperature_:.3f}")
    print(f"  Accuracy: {results['Temperature Scaling'].accuracy:.4f}")
    print(f"  ECE: {results['Temperature Scaling'].ece:.4f}")
    
    # =========================================================================
    # Method 3: Bootstrap Ensemble
    # =========================================================================
    print("\n🔹 Method 3: Bootstrap Ensemble (20 models)")
    
    ensemble = BootstrapEnsemble(
        base_estimator=LogisticRegression(max_iter=1000),
        n_estimators=20,
        random_state=seed,
    )
    ensemble.fit(X_train, y_train)
    
    y_prob_ensemble, y_std_ensemble = ensemble.predict_proba_with_uncertainty(X_test)
    results["Bootstrap Ensemble"] = compute_metrics(y_test, y_prob_ensemble)
    proba_dict["Bootstrap Ensemble"] = y_prob_ensemble
    print(f"  Mean epistemic uncertainty (std): {y_std_ensemble.mean():.4f}")
    print(f"  Accuracy: {results['Bootstrap Ensemble'].accuracy:.4f}")
    print(f"  ECE: {results['Bootstrap Ensemble'].ece:.4f}")
    
    # =========================================================================
    # Bonus: Ensemble + Temperature Scaling
    # =========================================================================
    print("\n🔹 Bonus: Ensemble + Temperature Scaling")
    
    y_prob_ensemble_val, _ = ensemble.predict_proba_with_uncertainty(X_val)
    temp_scaler_ens = TemperatureScaling()
    temp_scaler_ens.fit(y_val, y_prob_ensemble_val)
    
    y_prob_ens_temp = temp_scaler_ens.predict_proba(y_prob_ensemble)
    metrics_ens_temp = compute_metrics(y_test, y_prob_ens_temp)
    results["Ensemble + TempScale"] = UQMetrics(
        accuracy=results["Bootstrap Ensemble"].accuracy,  # Same as ensemble
        nll=metrics_ens_temp.nll,
        brier_score=metrics_ens_temp.brier_score,
        ece=metrics_ens_temp.ece
    )
    proba_dict["Ensemble + TempScale"] = y_prob_ens_temp
    print(f"  Accuracy: {results['Ensemble + TempScale'].accuracy:.4f}")
    print(f"  ECE: {results['Ensemble + TempScale'].ece:.4f}")
    
    # =========================================================================
    # Generate Report
    # =========================================================================
    print("\n📝 Generating report...")
    
    # Plot 1: Reliability diagrams comparison
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (method, y_prob) in enumerate(proba_dict.items()):
        ax = axes[idx]
        reliability_diagram(y_test, y_prob, n_bins=10, ax=ax)
        ax.set_title(f"{method}\n(ECE={results[method].ece:.4f})", fontsize=11)
    
    plt.suptitle("Reliability Diagrams: UQ Method Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig1.savefig(output_dir / "reliability_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Saved: {output_dir / 'reliability_diagrams.png'}")
    
    # Plot 2: Metric comparison bar chart
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(results.keys())
    metrics = ["NLL", "Brier Score", "ECE"]
    x = np.arange(len(methods))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [getattr(results[m], metric.lower().replace(" ", "_")) for m in methods]
        bars = ax.bar(x + i * width, values, width, label=metric, alpha=0.8)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Method")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("UQ Metrics Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved: {output_dir / 'metrics_comparison.png'}")
    
    # Plot 3: Epistemic uncertainty distribution (bootstrap ensemble)
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Uncertainty vs correctness
    ax = axes[0]
    y_pred = (y_prob_ensemble >= 0.5).astype(int)
    correct = y_pred == y_test
    
    ax.hist(y_std_ensemble[correct], bins=30, alpha=0.7, label="Correct", density=True)
    ax.hist(y_std_ensemble[~correct], bins=30, alpha=0.7, label="Incorrect", density=True)
    ax.set_xlabel("Epistemic Uncertainty (std)")
    ax.set_ylabel("Density")
    ax.set_title("Uncertainty vs Correctness")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Right: Confidence vs uncertainty
    ax = axes[1]
    confidence = np.abs(y_prob_ensemble - 0.5) * 2  # Map to [0, 1]
    ax.scatter(confidence, y_std_ensemble, alpha=0.3, s=10)
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Epistemic Uncertainty (std)")
    ax.set_title("Confidence vs Epistemic Uncertainty")
    ax.grid(alpha=0.3)
    
    # Fit trend line
    z = np.polyfit(confidence, y_std_ensemble, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), 'r--', label=f"Trend (slope={z[0]:.3f})")
    ax.legend()
    
    plt.suptitle("Bootstrap Ensemble: Epistemic Uncertainty Analysis", fontsize=12)
    plt.tight_layout()
    fig3.savefig(output_dir / "epistemic_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  ✓ Saved: {output_dir / 'epistemic_uncertainty.png'}")
    
    # Generate summary table (text)
    table_lines = [
        "=" * 70,
        "UQ IN CLASSIFICATION: METHOD COMPARISON REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random seed: {seed}",
        f"Dataset: 2000 samples, 20 features (60/20/20 train/val/test split)",
        "",
        "-" * 70,
        f"{'Method':<25} {'Accuracy':>10} {'NLL':>10} {'Brier':>10} {'ECE':>10}",
        "-" * 70,
    ]
    
    for method, metrics in results.items():
        table_lines.append(
            f"{method:<25} {metrics.accuracy:>10.4f} {metrics.nll:>10.4f} "
            f"{metrics.brier_score:>10.4f} {metrics.ece:>10.4f}"
        )
    
    table_lines.extend([
        "-" * 70,
        "",
        "INTERPRETATION:",
        "  - Accuracy: Classification accuracy (higher is better)",
        "  - NLL: Negative Log-Likelihood (lower is better)",
        "  - Brier: Brier Score = mean((p - y)^2) (lower is better)",
        "  - ECE: Expected Calibration Error (lower is better)",
        "",
        "KEY FINDINGS:",
    ])
    
    # Find best method for each metric
    best_ece = min(results.items(), key=lambda x: x[1].ece)
    best_nll = min(results.items(), key=lambda x: x[1].nll)
    best_brier = min(results.items(), key=lambda x: x[1].brier_score)
    
    table_lines.extend([
        f"  - Best ECE: {best_ece[0]} ({best_ece[1].ece:.4f})",
        f"  - Best NLL: {best_nll[0]} ({best_nll[1].nll:.4f})",
        f"  - Best Brier: {best_brier[0]} ({best_brier[1].brier_score:.4f})",
        "",
        "CONCLUSIONS:",
        "  1. Temperature scaling improves calibration (ECE) with minimal overhead",
        "  2. Bootstrap ensembles provide epistemic uncertainty estimates",
        "  3. Combining ensemble + temperature scaling often gives best overall UQ",
        "",
        "=" * 70,
    ])
    
    report_text = "\n".join(table_lines)
    
    # Save report
    with open(output_dir / "uq_comparison_report.txt", "w") as f:
        f.write(report_text)
    print(f"  ✓ Saved: {output_dir / 'uq_comparison_report.txt'}")
    
    # Save JSON for programmatic access
    json_results = {method: m.to_dict() for method, m in results.items()}
    with open(output_dir / "uq_metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  ✓ Saved: {output_dir / 'uq_metrics.json'}")
    
    # Print summary
    print("\n" + report_text)
    
    return results


# CLI entry point
if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.callback(invoke_without_command=True)
    def main(
        seed: int = typer.Option(42, help="Random seed"),
        output_dir: Path = typer.Option(
            Path("modules/02_stat_inference_uq/reports/uq_classification"),
            help="Output directory for report"
        ),
    ):
        """
        Run UQ classification comparison experiment.
        
        Compares predict_proba, temperature scaling, and bootstrap ensembles
        on a synthetic classification task.
        """
        run_comparison(seed=seed, output_dir=output_dir)
    
    app()
