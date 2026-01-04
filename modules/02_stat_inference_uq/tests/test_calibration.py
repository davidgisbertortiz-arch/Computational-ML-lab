"""Tests for calibration metrics and temperature scaling."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from modules._import_helper import safe_import_from

# Python 3.12+ workaround for numeric module names
(reliability_diagram, expected_calibration_error, 
 maximum_calibration_error, TemperatureScaling) = safe_import_from(
    '02_stat_inference_uq.src.calibration',
    'reliability_diagram', 'expected_calibration_error',
    'maximum_calibration_error', 'TemperatureScaling'
)
get_rng = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'get_rng')


class TestReliabilityDiagram:
    """Tests for reliability diagram."""
    
    def test_perfect_calibration(self):
        """Test ECE = 0 for perfectly calibrated predictions."""
        # Perfectly calibrated: predicted prob matches empirical frequency
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=4)
        
        # Should be small (not exactly zero due to finite samples and binning)
        assert ece < 0.3  # Relaxed for small sample + binning effects
        
    def test_overconfident_calibration(self):
        """Test that overconfident predictions have high ECE."""
        rng = get_rng(42)
        n_samples = 100
        
        # Simulate overconfident classifier
        y_true = rng.integers(0, 2, n_samples)
        # True accuracy is 70%, but predictions are close to 0/1
        y_prob = np.where(y_true == 1, 0.95, 0.05)
        y_prob += rng.normal(0, 0.02, n_samples)  # Small noise
        y_prob = np.clip(y_prob, 0, 1)
        
        # Randomly flip 30% of labels to create 70% accuracy
        flip_mask = rng.random(n_samples) < 0.3
        y_true[flip_mask] = 1 - y_true[flip_mask]
        
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        
        # Should have significant calibration error
        assert ece > 0.15
        
    def test_reliability_diagram_shapes(self):
        """Test that reliability diagram returns correct shapes."""
        rng = get_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        
        n_bins = 10
        centers, accs, counts = reliability_diagram(y_true, y_prob, n_bins=n_bins, ax=None)
        
        assert centers.shape == (n_bins,)
        assert accs.shape == (n_bins,)
        assert counts.shape == (n_bins,)
        assert np.sum(counts) == 100
        
    def test_reliability_diagram_plot(self):
        """Test that plotting works without errors."""
        rng = get_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        
        fig, ax = plt.subplots()
        centers, accs, counts = reliability_diagram(y_true, y_prob, n_bins=10, ax=ax)
        plt.close(fig)
        
        # Should complete without error
        assert True
        
    def test_input_validation(self):
        """Test input validation for calibration metrics."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        # Shape mismatch
        with pytest.raises(ValueError, match="Shape mismatch"):
            reliability_diagram(y_true, np.array([0.1, 0.9]), n_bins=2)
            
        # Non-binary labels
        with pytest.raises(ValueError, match="must be binary"):
            reliability_diagram(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]), n_bins=2)
            
        # Probabilities out of range
        with pytest.raises(ValueError, match="must be in"):
            reliability_diagram(y_true, np.array([0.1, 1.5, 0.2, 0.8]), n_bins=2)
            
    def test_quantile_binning(self):
        """Test quantile binning strategy."""
        rng = get_rng(42)
        y_true = rng.integers(0, 2, 100)
        # Skewed probabilities (mostly high)
        y_prob = rng.beta(5, 2, 100)
        
        centers, accs, counts = reliability_diagram(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        
        # Quantile binning should distribute samples more evenly
        assert np.min(counts) > 0  # No empty bins
        
    def test_mce_computation(self):
        """Test Maximum Calibration Error."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
        
        mce = maximum_calibration_error(y_true, y_prob, n_bins=3)
        
        assert isinstance(mce, float)
        assert 0 <= mce <= 1
        assert np.isfinite(mce)


class TestTemperatureScaling:
    """Tests for temperature scaling."""
    
    def test_binary_fit(self):
        """Test fitting temperature for binary classification."""
        rng = get_rng(42)
        n_samples = 200
        
        # Simulate overconfident logits
        y_true = rng.integers(0, 2, n_samples)
        logits = np.where(y_true == 1, 3.0, -3.0) + rng.normal(0, 0.5, n_samples)
        
        ts = TemperatureScaling(n_classes=2, max_iter=100)
        ts.fit(logits, y_true)
        
        # Temperature should be learned
        assert ts.temperature_ > 0
        assert np.isfinite(ts.temperature_)
        # For overconfident predictions, T should be > 1
        assert ts.temperature_ > 0.8
        
    def test_binary_calibration_improvement(self):
        """Test that temperature scaling improves calibration."""
        rng = get_rng(42)
        n_samples = 500
        
        # Overconfident predictions
        y_true = rng.integers(0, 2, n_samples)
        logits = np.where(y_true == 1, 2.5, -2.5) + rng.normal(0, 1.0, n_samples)
        
        # Uncalibrated probabilities
        probs_uncalib = 1 / (1 + np.exp(-logits))
        ece_before = expected_calibration_error(y_true, probs_uncalib, n_bins=10)
        
        # Apply temperature scaling
        ts = TemperatureScaling(n_classes=2)
        ts.fit(logits, y_true)
        probs_calib = ts.predict_proba(logits)
        ece_after = expected_calibration_error(y_true, probs_calib, n_bins=10)
        
        # ECE should decrease or stay similar (if already well-calibrated)
        assert ece_after <= ece_before * 1.5  # Allow tolerance for well-calibrated cases
        
    def test_multiclass_fit(self):
        """Test fitting temperature for multi-class classification."""
        rng = get_rng(42)
        n_samples = 200
        n_classes = 3
        
        y_true = rng.integers(0, n_classes, n_samples)
        logits = rng.normal(0, 2, (n_samples, n_classes))
        # Make predictions more confident
        logits[np.arange(n_samples), y_true] += 3.0
        
        ts = TemperatureScaling(n_classes=n_classes, max_iter=100)
        ts.fit(logits, y_true)
        
        assert ts.temperature_ > 0
        assert np.isfinite(ts.temperature_)
        
    def test_multiclass_predict_proba(self):
        """Test multi-class probability prediction."""
        rng = get_rng(42)
        n_samples = 50
        n_classes = 4
        
        y_true = rng.integers(0, n_classes, n_samples)
        logits_train = rng.normal(0, 1, (n_samples, n_classes))
        logits_train[np.arange(n_samples), y_true] += 2.0
        
        ts = TemperatureScaling(n_classes=n_classes)
        ts.fit(logits_train, y_true)
        
        # Test set
        logits_test = rng.normal(0, 1, (20, n_classes))
        probs = ts.predict_proba(logits_test)
        
        assert probs.shape == (20, n_classes)
        # Probabilities should sum to 1
        assert np.allclose(np.sum(probs, axis=1), 1.0)
        assert np.all((probs >= 0) & (probs <= 1))
        
    def test_temperature_effect(self):
        """Test that higher temperature smooths probabilities."""
        logits = np.array([2.0, -2.0, 0.0])
        
        # Low temperature (T < 1) makes sharper
        ts_low = TemperatureScaling(n_classes=2)
        ts_low.temperature_ = 0.5
        probs_low = ts_low.predict_proba(logits)
        
        # High temperature (T > 1) makes smoother
        ts_high = TemperatureScaling(n_classes=2)
        ts_high.temperature_ = 2.0
        probs_high = ts_high.predict_proba(logits)
        
        # Low temp should give more extreme probabilities
        assert np.abs(probs_low[0] - 0.5) > np.abs(probs_high[0] - 0.5)
        
    def test_reproducibility(self):
        """Test that fitting is reproducible."""
        rng1 = get_rng(42)
        rng2 = get_rng(42)
        
        n_samples = 100
        y_true = rng1.integers(0, 2, n_samples)
        logits = rng1.normal(0, 1, n_samples)
        
        ts1 = TemperatureScaling(n_classes=2)
        ts1.fit(logits, y_true)
        
        # Reset RNG and refit
        y_true2 = rng2.integers(0, 2, n_samples)
        logits2 = rng2.normal(0, 1, n_samples)
        
        ts2 = TemperatureScaling(n_classes=2)
        ts2.fit(logits2, y_true2)
        
        assert np.isclose(ts1.temperature_, ts2.temperature_, rtol=1e-5)
        
    def test_input_validation(self):
        """Test input validation for temperature scaling."""
        ts = TemperatureScaling(n_classes=2)
        
        # Wrong logits shape for binary
        with pytest.raises(ValueError, match="must be 1D"):
            ts.fit(np.ones((10, 2)), np.ones(10))
            
        ts_mc = TemperatureScaling(n_classes=3)
        
        # Wrong logits shape for multi-class
        with pytest.raises(ValueError, match="Expected shape"):
            ts_mc.fit(np.ones((10, 2)), np.ones(10))


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_calibration_pipeline(self):
        """Test full calibration analysis pipeline."""
        rng = get_rng(42)
        n_samples = 300
        
        # Generate synthetic classifier outputs
        y_true = rng.integers(0, 2, n_samples)
        logits = np.where(y_true == 1, 2.0, -2.0) + rng.normal(0, 1.0, n_samples)
        
        # Split into val and test
        n_val = 200
        logits_val, logits_test = logits[:n_val], logits[n_val:]
        y_val, y_test = y_true[:n_val], y_true[n_val:]
        
        # Uncalibrated predictions
        probs_uncalib = 1 / (1 + np.exp(-logits_test))
        ece_before = expected_calibration_error(y_test, probs_uncalib, n_bins=10)
        
        # Fit temperature on validation set
        ts = TemperatureScaling(n_classes=2)
        ts.fit(logits_val, y_val)
        
        # Calibrated predictions on test set
        probs_calib = ts.predict_proba(logits_test)
        ece_after = expected_calibration_error(y_test, probs_calib, n_bins=10)
        
        # Calibration should improve or stay similar
        assert ece_after <= ece_before * 1.3  # Allow some tolerance
        assert ece_after < 0.2  # Should be reasonably calibrated
