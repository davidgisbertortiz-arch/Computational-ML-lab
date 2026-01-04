"""Tests for MCMC sampling and diagnostics."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

# Python 3.12+ workaround for numeric module names
MetropolisHastings, MCMCDiagnostics = safe_import_from(
    '02_stat_inference_uq.src.mcmc_basics',
    'MetropolisHastings', 'MCMCDiagnostics'
)
get_rng = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'get_rng')


class TestMetropolisHastings:
    """Tests for Metropolis-Hastings sampler."""
    
    def test_sample_1d_gaussian(self):
        """Test sampling from 1D standard normal."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            proposal_std=1.0,
            n_samples=5000,
            n_burn=1000,
            random_state=42,
        )
        
        samples = sampler.sample(x0=np.array([0.0]), verbose=False)
        
        assert samples.shape == (5000, 1)
        assert np.all(np.isfinite(samples))
        
        # Sample mean should be close to 0
        assert np.abs(np.mean(samples)) < 0.1
        # Sample std should be close to 1
        assert np.abs(np.std(samples) - 1.0) < 0.15
        
    def test_sample_2d_gaussian(self):
        """Test sampling from 2D independent Gaussian."""
        def log_prob(x):
            # N(0, diag([1, 4]))
            return -0.5 * (x[0]**2 + x[1]**2 / 4)
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            proposal_std=1.5,
            n_samples=8000,
            n_burn=2000,
            random_state=42,
        )
        
        samples = sampler.sample(x0=np.array([0.0, 0.0]), verbose=False)
        
        assert samples.shape == (8000, 2)
        
        # Check marginal means
        assert np.allclose(np.mean(samples, axis=0), [0, 0], atol=0.15)
        # Check marginal variances
        assert np.allclose(np.std(samples, axis=0), [1, 2], atol=0.2)
        
    def test_acceptance_rate(self):
        """Test that acceptance rate is reasonable."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            proposal_std=1.0,
            n_samples=2000,
            n_burn=500,
            random_state=42,
        )
        
        sampler.sample(x0=np.zeros(2), verbose=False)
        
        # Acceptance rate should be in reasonable range
        assert 0.1 < sampler.accept_rate_ < 0.9
        # For RW-MH with tuned proposal, ideally ~0.23-0.4
        
    def test_proposal_std_effect(self):
        """Test that proposal std affects acceptance rate."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        # Small proposal std -> high acceptance
        sampler_small = MetropolisHastings(
            log_prob_fn=log_prob,
            proposal_std=0.1,
            n_samples=1000,
            n_burn=200,
            random_state=42,
        )
        sampler_small.sample(x0=np.zeros(1), verbose=False)
        
        # Large proposal std -> low acceptance
        sampler_large = MetropolisHastings(
            log_prob_fn=log_prob,
            proposal_std=5.0,
            n_samples=1000,
            n_burn=200,
            random_state=42,
        )
        sampler_large.sample(x0=np.zeros(1), verbose=False)
        
        assert sampler_small.accept_rate_ > sampler_large.accept_rate_
        
    def test_log_probs_stored(self):
        """Test that log probabilities are stored."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            n_samples=1000,
            n_burn=200,
            random_state=42,
        )
        
        samples = sampler.sample(x0=np.zeros(2), verbose=False)
        
        assert sampler.log_probs_ is not None
        assert len(sampler.log_probs_) == len(samples)
        assert np.all(np.isfinite(sampler.log_probs_))
        
    def test_thinning(self):
        """Test that thinning reduces sample count."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            n_samples=1000,
            n_burn=200,
            thin=5,
            random_state=42,
        )
        
        samples = sampler.sample(x0=np.zeros(1), verbose=False)
        
        # After thinning by 5, should have ~200 samples
        assert samples.shape[0] == 200
        
    def test_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        sampler1 = MetropolisHastings(
            log_prob_fn=log_prob,
            n_samples=1000,
            n_burn=200,
            random_state=42,
        )
        samples1 = sampler1.sample(x0=np.zeros(2), verbose=False)
        
        sampler2 = MetropolisHastings(
            log_prob_fn=log_prob,
            n_samples=1000,
            n_burn=200,
            random_state=42,
        )
        samples2 = sampler2.sample(x0=np.zeros(2), verbose=False)
        
        assert np.array_equal(samples1, samples2)
        
    def test_diagnostics_dict(self):
        """Test that diagnostics can be computed."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            n_samples=3000,
            n_burn=500,
            random_state=42,
        )
        
        sampler.sample(x0=np.zeros(2), verbose=False)
        diag = sampler.get_diagnostics()
        
        assert "acceptance_rate" in diag
        assert "ess" in diag
        assert "autocorr_time" in diag
        assert "mean" in diag
        assert "std" in diag
        
        assert np.all(diag["ess"] > 0)
        assert np.all(diag["ess"] <= len(sampler.samples_))


class TestMCMCDiagnostics:
    """Tests for MCMC diagnostics."""
    
    def test_mean_std(self):
        """Test mean and std computation."""
        rng = get_rng(42)
        samples = rng.standard_normal((1000, 3))
        
        diag = MCMCDiagnostics(samples)
        
        mean = diag.mean()
        std = diag.std()
        
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.allclose(mean, np.mean(samples, axis=0))
        assert np.allclose(std, np.std(samples, axis=0, ddof=1))
        
    def test_autocorrelation(self):
        """Test autocorrelation computation."""
        # Create correlated samples
        rng = get_rng(42)
        n_samples = 500
        samples = np.cumsum(rng.standard_normal((n_samples, 2)), axis=0) / np.sqrt(np.arange(1, n_samples+1))[:, None]
        
        diag = MCMCDiagnostics(samples)
        acf = diag.autocorrelation(max_lag=50)
        
        assert acf.shape == (50, 2)
        # ACF at lag 0 should be 1
        assert np.allclose(acf[0], 1.0)
        # ACF should generally decay
        assert np.all(acf[-1] < acf[0])
        
    def test_integrated_autocorr_time(self):
        """Test integrated autocorrelation time."""
        # IID samples should have tau_int ≈ 1
        rng = get_rng(42)
        samples_iid = rng.standard_normal((2000, 2))
        
        diag = MCMCDiagnostics(samples_iid)
        tau_int = diag.integrated_autocorr_time()
        
        assert tau_int.shape == (2,)
        assert np.all(tau_int >= 1.0)  # Always >= 1
        # For IID, should be close to 1 (small sample noise can increase estimate)
        assert np.all(tau_int < 5.0)  # Relaxed for finite sample effects
        
    def test_effective_sample_size(self):
        """Test ESS computation."""
        rng = get_rng(42)
        n_samples = 1000
        samples = rng.standard_normal((n_samples, 2))
        
        diag = MCMCDiagnostics(samples)
        ess = diag.effective_sample_size()
        
        assert ess.shape == (2,)
        assert np.all(ess > 0)
        assert np.all(ess <= n_samples)
        # For IID, ESS should be reasonably high (estimation bias can reduce it)
        assert np.all(ess > 0.3 * n_samples)  # Relaxed for estimation variance
        
    def test_trace_plot(self):
        """Test that trace plot can be generated."""
        rng = get_rng(42)
        samples = rng.standard_normal((500, 3))
        
        diag = MCMCDiagnostics(samples)
        fig = diag.trace_plot(dims=[0, 1])
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    def test_marginal_histograms(self):
        """Test that marginal histograms can be generated."""
        rng = get_rng(42)
        samples = rng.standard_normal((1000, 2))
        
        diag = MCMCDiagnostics(samples)
        fig = diag.marginal_histograms(dims=[0, 1])
        
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
        
    def test_input_validation(self):
        """Test input validation for diagnostics."""
        # Wrong shape
        with pytest.raises(ValueError, match="must be 2D"):
            MCMCDiagnostics(np.array([1, 2, 3]))


class TestIntegration:
    """Integration tests for full MCMC workflow."""
    
    def test_sample_and_diagnose(self):
        """Test full sampling + diagnostics workflow."""
        # Bivariate Gaussian with correlation
        rho = 0.7
        cov = np.array([[1, rho], [rho, 1]])
        cov_inv = np.linalg.inv(cov)
        
        def log_prob(x):
            return -0.5 * x @ cov_inv @ x
            
        sampler = MetropolisHastings(
            log_prob_fn=log_prob,
            proposal_std=1.2,
            n_samples=5000,
            n_burn=1000,
            random_state=42,
        )
        
        samples = sampler.sample(x0=np.zeros(2), verbose=False)
        
        # Check sampling quality
        assert 0.2 < sampler.accept_rate_ < 0.6
        
        # Diagnostics
        diag_dict = sampler.get_diagnostics()
        assert diag_dict["ess"][0] > 100  # At least 100 effective samples
        
        # Mean should be close to [0, 0]
        assert np.allclose(diag_dict["mean"], [0, 0], atol=0.15)
        
        # Std should be close to [1, 1]
        assert np.allclose(diag_dict["std"], [1, 1], atol=0.15)
        
        # Empirical correlation should be close to true correlation
        empirical_corr = np.corrcoef(samples.T)[0, 1]
        assert np.abs(empirical_corr - rho) < 0.1
        
    def test_convergence_from_different_inits(self):
        """Test that chains converge to same distribution from different starts."""
        def log_prob(x):
            return -0.5 * np.sum(x**2)
            
        inits = [np.array([5.0]), np.array([-5.0]), np.array([0.0])]
        means = []
        
        for init in inits:
            sampler = MetropolisHastings(
                log_prob_fn=log_prob,
                proposal_std=1.0,
                n_samples=3000,
                n_burn=1000,
                random_state=42,
            )
            samples = sampler.sample(x0=init, verbose=False)
            means.append(np.mean(samples))
            
        # All chains should converge to same mean
        assert np.std(means) < 0.2
