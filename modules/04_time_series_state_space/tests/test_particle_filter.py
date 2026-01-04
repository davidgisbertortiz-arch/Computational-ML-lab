"""Tests for Particle Filter implementation."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

ParticleFilter, gaussian_likelihood, create_process_noise_wrapper = safe_import_from(
    '04_time_series_state_space.src.particle_filter',
    'ParticleFilter', 'gaussian_likelihood', 'create_process_noise_wrapper'
)

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestParticleFilter:
    """Test Particle Filter implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        
        # Simple linear dynamics for testing
        self.F = np.array([[1, 0.1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.1]])
        
        def f(x, u, rng):
            return self.F @ x + rng.multivariate_normal(np.zeros(2), self.Q)
        
        def h(x):
            return self.H @ x
        
        self.f = f
        self.h = h
        self.likelihood = gaussian_likelihood(self.R)
        
    def test_initialization(self):
        """Test particle filter initialization."""
        pf = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        assert pf.particles.shape == (100, 2)
        assert pf.weights.shape == (100,)
        assert np.allclose(np.sum(pf.weights), 1.0)
        
    def test_predict_shapes(self):
        """Test prediction step output shapes."""
        pf = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        particles = pf.predict()
        
        assert particles.shape == (100, 2)
        
    def test_update_weights_normalized(self):
        """Test that update step normalizes weights."""
        pf = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        pf.predict()
        weights = pf.update(np.array([1.0]))
        
        assert np.allclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        
    def test_resample_preserves_particles(self):
        """Test that resampling maintains particle count."""
        pf = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        pf.predict()
        pf.update(np.array([1.0]))
        pf.resample()
        
        assert pf.particles.shape == (100, 2)
        assert pf.weights.shape == (100,)
        
    def test_state_estimate(self):
        """Test getting state estimate from particles."""
        pf = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        mean, cov = pf.get_state_estimate()
        
        assert mean.shape == (2,)
        assert cov.shape == (2, 2)
        # Mean should be close to initialization mean
        assert np.allclose(mean, np.array([0, 1]), atol=0.5)
        
    def test_reproducibility_with_seed(self):
        """Test that particle filter is reproducible with same seed."""
        observations = np.array([1.0, 1.1, 1.2])
        
        # Run 1
        pf1 = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        rng1 = np.random.default_rng(42)
        pf1.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng1)
        
        states1 = []
        for z in observations:
            pf1.predict()
            pf1.update(np.array([z]))
            pf1.resample()
            mean, _ = pf1.get_state_estimate()
            states1.append(mean)
        
        # Run 2 with same seed
        pf2 = ParticleFilter(self.f, self.h, self.likelihood, n_particles=100)
        rng2 = np.random.default_rng(42)
        pf2.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng2)
        
        states2 = []
        for z in observations:
            pf2.predict()
            pf2.update(np.array([z]))
            pf2.resample()
            mean, _ = pf2.get_state_estimate()
            states2.append(mean)
        
        assert np.allclose(states1, states2)
        
    def test_more_particles_better_estimate(self):
        """Sanity test: more particles should give better estimate."""
        set_seed(42)
        
        # Generate true trajectory
        true_states = []
        observations = []
        x = np.array([0.0, 1.0])
        
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = self.F @ x + rng.multivariate_normal(np.zeros(2), self.Q)
            true_states.append(x.copy())
            z = self.H @ x + rng.normal(0, np.sqrt(self.R[0, 0]))
            observations.append(z[0])
        
        true_states = np.array(true_states)
        
        # Run with 50 particles
        pf_small = ParticleFilter(self.f, self.h, self.likelihood, n_particles=50)
        rng_small = np.random.default_rng(42)
        pf_small.initialize(mean=np.array([0, 0]), cov=np.eye(2), rng=rng_small)
        
        estimates_small = []
        for z in observations:
            pf_small.predict()
            pf_small.update(np.array([z]))
            pf_small.resample()
            mean, _ = pf_small.get_state_estimate()
            estimates_small.append(mean)
        
        estimates_small = np.array(estimates_small)
        rmse_small = np.sqrt(np.mean((true_states - estimates_small)**2))
        
        # Run with 500 particles
        pf_large = ParticleFilter(self.f, self.h, self.likelihood, n_particles=500)
        rng_large = np.random.default_rng(42)
        pf_large.initialize(mean=np.array([0, 0]), cov=np.eye(2), rng=rng_large)
        
        estimates_large = []
        for z in observations:
            pf_large.predict()
            pf_large.update(np.array([z]))
            pf_large.resample()
            mean, _ = pf_large.get_state_estimate()
            estimates_large.append(mean)
        
        estimates_large = np.array(estimates_large)
        rmse_large = np.sqrt(np.mean((true_states - estimates_large)**2))
        
        # More particles should give equal or better estimate (not strict due to randomness)
        assert rmse_large <= rmse_small * 1.5


class TestHelpers:
    """Test helper functions."""
    
    def test_gaussian_likelihood(self):
        """Test Gaussian likelihood function creation."""
        R = np.array([[1.0]])
        likelihood = gaussian_likelihood(R)
        
        # Test on some points
        z_obs = np.array([0.0])
        z_pred = np.array([0.0])
        prob = likelihood(z_obs, z_pred)
        
        assert prob > 0
        # At mean, probability should be maximized
        assert prob > likelihood(z_obs, np.array([1.0]))
        
    def test_process_noise_wrapper(self):
        """Test process noise wrapper."""
        def f_det(x, u):
            return np.array([x[0] + 1, x[1] + 2])
        
        Q = np.eye(2) * 0.1
        f_stoch = create_process_noise_wrapper(f_det, Q)
        
        x = np.array([0.0, 0.0])
        rng = np.random.default_rng(42)
        
        x_next = f_stoch(x, None, rng)
        
        assert x_next.shape == (2,)
        # Should be close to deterministic result but with noise
        assert np.abs(x_next[0] - 1.0) < 1.0  # Allowing for noise
        assert np.abs(x_next[1] - 2.0) < 1.0
