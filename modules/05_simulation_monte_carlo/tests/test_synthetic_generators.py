"""Tests for synthetic data generators."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(generate_brownian_motion, generate_ou_process, generate_levy_flight,
 PhysicsDataGenerator, generate_correlated_noise) = safe_import_from(
    '05_simulation_monte_carlo.src.synthetic_generators',
    'generate_brownian_motion', 'generate_ou_process', 'generate_levy_flight',
    'PhysicsDataGenerator', 'generate_correlated_noise'
)


class TestStochasticProcesses:
    """Test stochastic process generators."""
    
    def test_brownian_motion_shapes(self):
        """Test Brownian motion output shapes."""
        t, x = generate_brownian_motion(n_steps=100, seed=42)
        
        assert len(t) == 101  # n_steps + 1
        assert len(x) == 101
        assert x[0] == 0.0  # Initial value
    
    def test_brownian_drift(self):
        """Test that Brownian motion has correct drift."""
        t, x = generate_brownian_motion(n_steps=10000, dt=0.01, mu=0.5, sigma=0.1, seed=42)
        
        # Mean should grow approximately as μ*t
        expected_mean = 0.5 * t[-1]
        assert abs(x[-1] - expected_mean) < 2.0  # Within reasonable range (increased tolerance)
    
    def test_ou_process_mean_reversion(self):
        """Test OU process mean reversion."""
        t, x = generate_ou_process(n_steps=5000, dt=0.01, theta=1.0, mu=2.0, x0=0.0, seed=42)
        
        # Long-term mean should be close to mu
        long_term_mean = np.mean(x[2500:])  # Second half
        assert abs(long_term_mean - 2.0) < 0.5
    
    def test_levy_flight_shapes(self):
        """Test Lévy flight shapes."""
        steps, positions = generate_levy_flight(n_steps=100, alpha=1.5, seed=42)
        
        assert len(steps) == 101
        assert len(positions) == 101
        assert positions[0] == 0.0
    
    def test_reproducibility(self):
        """Test reproducibility with seeds."""
        t1, x1 = generate_brownian_motion(n_steps=100, seed=42)
        t2, x2 = generate_brownian_motion(n_steps=100, seed=42)
        
        assert np.array_equal(t1, t2)
        assert np.array_equal(x1, x2)


class TestPhysicsDataGenerator:
    """Test physics dataset generator."""
    
    def test_damped_oscillator(self):
        """Test damped harmonic oscillator dataset."""
        gen = PhysicsDataGenerator(seed=42)
        dataset = gen.damped_harmonic_oscillator(n_samples=100, noise_level=0.1)
        
        assert dataset.features.shape == (100, 2)  # (time, velocity)
        assert dataset.targets.shape == (100,)
        assert dataset.noise_level == 0.1
        assert 'oscillator' in dataset.description.lower()
    
    def test_projectile_motion(self):
        """Test projectile motion dataset."""
        gen = PhysicsDataGenerator(seed=42)
        dataset = gen.projectile_motion(n_samples=50, noise_level=0.5)
        
        assert dataset.features.shape == (50, 2)  # (v0, angle)
        assert dataset.targets.shape == (50,)
        # Range should be positive
        assert np.all(dataset.targets > 0)
    
    def test_heat_diffusion(self):
        """Test heat diffusion dataset."""
        gen = PhysicsDataGenerator(seed=42)
        dataset = gen.heat_diffusion_1d(n_samples=100, noise_level=0.05)
        
        assert dataset.features.shape == (100, 2)  # (x, t)
        assert dataset.targets.shape == (100,)
    
    def test_pendulum_energy(self):
        """Test pendulum energy dataset."""
        gen = PhysicsDataGenerator(seed=42)
        dataset = gen.pendulum_energy(n_samples=100, noise_level=0.1)
        
        assert dataset.features.shape == (100, 2)  # (theta, theta_dot)
        assert dataset.targets.shape == (100,)
        # Energy should be non-negative
        assert np.all(dataset.targets >= -0.5)  # Allow small negative due to noise
    
    def test_ground_truth_callable(self):
        """Test that ground truth function works."""
        gen = PhysicsDataGenerator(seed=42)
        dataset = gen.damped_harmonic_oscillator(n_samples=10)
        
        # Ground truth should be callable
        test_input = dataset.features[:5]
        predictions = dataset.ground_truth_func(test_input[:, 0])  # Just time
        
        assert len(predictions) == 5
    
    def test_reproducibility(self):
        """Test reproducibility."""
        gen1 = PhysicsDataGenerator(seed=42)
        gen2 = PhysicsDataGenerator(seed=42)
        
        ds1 = gen1.damped_harmonic_oscillator(n_samples=50)
        ds2 = gen2.damped_harmonic_oscillator(n_samples=50)
        
        assert np.array_equal(ds1.features, ds2.features)
        assert np.array_equal(ds1.targets, ds2.targets)


class TestCorrelatedNoise:
    """Test correlated noise generator."""
    
    def test_correlated_noise_shape(self):
        """Test output shape."""
        noise = generate_correlated_noise(n_samples=100, correlation_length=5.0, seed=42)
        
        assert len(noise) == 100
    
    def test_correlation_structure(self):
        """Test that noise has expected correlation."""
        noise = generate_correlated_noise(
            n_samples=500, correlation_length=10.0, sigma=1.0, seed=42
        )
        
        # Compute autocorrelation
        mean_noise = np.mean(noise)
        autocorr = np.correlate(noise - mean_noise, noise - mean_noise, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        
        # Should decay with lag
        assert autocorr[0] == 1.0
        assert autocorr[20] < autocorr[5]
    
    def test_reproducibility(self):
        """Test reproducibility."""
        noise1 = generate_correlated_noise(n_samples=100, seed=42)
        noise2 = generate_correlated_noise(n_samples=100, seed=42)
        
        assert np.array_equal(noise1, noise2)
