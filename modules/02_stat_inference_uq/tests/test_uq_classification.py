"""Tests for UQ classification comparison experiment."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Import experiment components
import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from modules._import_helper import safe_import_from

# Import test subjects
negative_log_likelihood = safe_import_from(
    '02_stat_inference_uq.experiments.uq_classification_comparison',
    'negative_log_likelihood'
)
brier_score = safe_import_from(
    '02_stat_inference_uq.experiments.uq_classification_comparison',
    'brier_score'
)
compute_metrics = safe_import_from(
    '02_stat_inference_uq.experiments.uq_classification_comparison',
    'compute_metrics'
)
BootstrapEnsemble = safe_import_from(
    '02_stat_inference_uq.experiments.uq_classification_comparison',
    'BootstrapEnsemble'
)
generate_dataset = safe_import_from(
    '02_stat_inference_uq.experiments.uq_classification_comparison',
    'generate_dataset'
)
run_comparison = safe_import_from(
    '02_stat_inference_uq.experiments.uq_classification_comparison',
    'run_comparison'
)


class TestMetrics:
    """Tests for UQ evaluation metrics."""
    
    def test_nll_perfect_prediction(self):
        """NLL should be near zero for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.01, 0.01, 0.99, 0.99])  # Near-perfect
        
        nll = negative_log_likelihood(y_true, y_prob)
        assert nll < 0.1  # Should be very low
    
    def test_nll_worst_prediction(self):
        """NLL should be high for wrong confident predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.99, 0.99, 0.01, 0.01])  # Completely wrong
        
        nll = negative_log_likelihood(y_true, y_prob)
        assert nll > 2.0  # Should be high
    
    def test_nll_random_prediction(self):
        """NLL for random predictions should be ~log(2) ≈ 0.693."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])  # Random
        
        nll = negative_log_likelihood(y_true, y_prob)
        assert 0.6 < nll < 0.8  # Should be around log(2)
    
    def test_brier_perfect_prediction(self):
        """Brier score should be near zero for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        
        brier = brier_score(y_true, y_prob)
        assert brier < 0.01
    
    def test_brier_worst_prediction(self):
        """Brier score should be 1 for completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        
        brier = brier_score(y_true, y_prob)
        assert abs(brier - 1.0) < 0.01
    
    def test_brier_random_prediction(self):
        """Brier score for 50% predictions should be 0.25."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        
        brier = brier_score(y_true, y_prob)
        assert abs(brier - 0.25) < 0.01
    
    def test_compute_metrics_returns_all(self):
        """compute_metrics should return all expected metrics."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.6, 0.7, 0.8])
        
        metrics = compute_metrics(y_true, y_prob)
        
        assert hasattr(metrics, 'accuracy')
        assert hasattr(metrics, 'nll')
        assert hasattr(metrics, 'brier_score')
        assert hasattr(metrics, 'ece')
        
        assert 0 <= metrics.accuracy <= 1
        assert metrics.nll > 0
        assert 0 <= metrics.brier_score <= 1
        assert 0 <= metrics.ece <= 1


class TestBootstrapEnsemble:
    """Tests for Bootstrap Ensemble."""
    
    def test_fit_creates_estimators(self):
        """Fit should create n_estimators models."""
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        
        ensemble = BootstrapEnsemble(
            base_estimator=LogisticRegression(max_iter=200),
            n_estimators=5,
            random_state=42,
        )
        ensemble.fit(X, y)
        
        assert len(ensemble.estimators_) == 5
    
    def test_predict_proba_shape(self):
        """predict_proba should return correct shape."""
        from sklearn.linear_model import LogisticRegression
        
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = np.random.randn(20, 5)
        
        ensemble = BootstrapEnsemble(
            base_estimator=LogisticRegression(max_iter=200),
            n_estimators=5,
            random_state=42,
        )
        ensemble.fit(X_train, y_train)
        
        y_prob = ensemble.predict_proba(X_test)
        assert y_prob.shape == (20,)
        assert np.all((y_prob >= 0) & (y_prob <= 1))
    
    def test_uncertainty_estimates(self):
        """predict_proba_with_uncertainty should return mean and std."""
        from sklearn.linear_model import LogisticRegression
        
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = np.random.randn(20, 5)
        
        ensemble = BootstrapEnsemble(
            base_estimator=LogisticRegression(max_iter=200),
            n_estimators=10,
            random_state=42,
        )
        ensemble.fit(X_train, y_train)
        
        mean_prob, std_prob = ensemble.predict_proba_with_uncertainty(X_test)
        
        assert mean_prob.shape == (20,)
        assert std_prob.shape == (20,)
        assert np.all(std_prob >= 0)  # Std should be non-negative
    
    def test_reproducibility(self):
        """Same seed should give same results."""
        from sklearn.linear_model import LogisticRegression
        
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = np.random.randn(10, 5)
        
        # First run
        ens1 = BootstrapEnsemble(
            base_estimator=LogisticRegression(max_iter=200),
            n_estimators=5,
            random_state=42,
        )
        ens1.fit(X_train, y_train)
        prob1 = ens1.predict_proba(X_test)
        
        # Second run with same seed
        ens2 = BootstrapEnsemble(
            base_estimator=LogisticRegression(max_iter=200),
            n_estimators=5,
            random_state=42,
        )
        ens2.fit(X_train, y_train)
        prob2 = ens2.predict_proba(X_test)
        
        np.testing.assert_array_almost_equal(prob1, prob2)


class TestDataset:
    """Tests for dataset generation."""
    
    def test_generate_dataset_shapes(self):
        """Generated dataset should have correct shapes."""
        X_train, X_val, X_test, y_train, y_val, y_test = generate_dataset(
            n_samples=1000, n_features=10, random_state=42
        )
        
        # Check shapes
        assert X_train.shape[0] == 600  # 60% train
        assert X_val.shape[0] == 200    # 20% val
        assert X_test.shape[0] == 200   # 20% test
        
        assert X_train.shape[1] == 10
        assert X_val.shape[1] == 10
        assert X_test.shape[1] == 10
        
        assert len(y_train) == 600
        assert len(y_val) == 200
        assert len(y_test) == 200
    
    def test_generate_dataset_binary_labels(self):
        """Labels should be binary."""
        _, _, _, y_train, y_val, y_test = generate_dataset(
            n_samples=500, random_state=42
        )
        
        for y in [y_train, y_val, y_test]:
            assert set(np.unique(y)).issubset({0, 1})


class TestFullComparison:
    """Integration tests for full comparison experiment."""
    
    def test_run_comparison_completes(self):
        """Full comparison should complete without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_comparison(seed=42, output_dir=Path(tmpdir))
            
            # Should return results for all methods
            assert "Raw predict_proba" in results
            assert "Temperature Scaling" in results
            assert "Bootstrap Ensemble" in results
            assert "Ensemble + TempScale" in results
    
    def test_run_comparison_creates_outputs(self):
        """Comparison should create all expected output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            run_comparison(seed=42, output_dir=output_dir)
            
            # Check files exist
            assert (output_dir / "reliability_diagrams.png").exists()
            assert (output_dir / "metrics_comparison.png").exists()
            assert (output_dir / "epistemic_uncertainty.png").exists()
            assert (output_dir / "uq_comparison_report.txt").exists()
            assert (output_dir / "uq_metrics.json").exists()
    
    def test_metrics_are_reasonable(self):
        """Metrics should be in reasonable ranges."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_comparison(seed=42, output_dir=Path(tmpdir))
            
            for method, metrics in results.items():
                # Accuracy should be reasonable (above random)
                assert metrics.accuracy > 0.5, f"{method} accuracy too low"
                assert metrics.accuracy <= 1.0
                
                # NLL should be positive and finite
                assert metrics.nll > 0
                assert metrics.nll < 10
                
                # Brier score in [0, 1]
                assert 0 <= metrics.brier_score <= 1
                
                # ECE in [0, 1]
                assert 0 <= metrics.ece <= 1
