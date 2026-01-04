"""Tests for seeding utilities."""

import pytest
import numpy as np
import random
from modules._import_helper import safe_import_from

set_seed, get_rng, check_determinism = safe_import_from(
    '00_repo_standards.src.mlphys_core.seeding',
    'set_seed', 'get_rng', 'check_determinism'
)


class TestSetSeed:
    """Tests for set_seed function."""
    
    def test_numpy_reproducibility(self):
        """Test that numpy is reproducible with set_seed."""
        set_seed(42)
        x1 = np.random.randn(10)
        
        set_seed(42)
        x2 = np.random.randn(10)
        
        assert np.array_equal(x1, x2)
    
    def test_python_random_reproducibility(self):
        """Test that Python random is reproducible with set_seed."""
        set_seed(42)
        vals1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        vals2 = [random.random() for _ in range(10)]
        
        assert vals1 == vals2
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        x1 = np.random.randn(10)
        
        set_seed(43)
        x2 = np.random.randn(10)
        
        assert not np.array_equal(x1, x2)
    
    def test_multiple_calls_same_seed(self):
        """Test multiple set_seed calls with same seed."""
        set_seed(42)
        set_seed(42)
        x1 = np.random.randn(5)
        
        set_seed(42)
        set_seed(42)
        x2 = np.random.randn(5)
        
        assert np.array_equal(x1, x2)


class TestGetRng:
    """Tests for get_rng function."""
    
    def test_returns_generator(self):
        """Test that get_rng returns a Generator instance."""
        rng = get_rng(42)
        assert isinstance(rng, np.random.Generator)
    
    def test_reproducibility(self):
        """Test that same seed produces same sequence."""
        rng1 = get_rng(42)
        x1 = rng1.standard_normal(10)
        
        rng2 = get_rng(42)
        x2 = rng2.standard_normal(10)
        
        assert np.array_equal(x1, x2)
    
    def test_isolation_from_global_state(self):
        """Test that RNG is isolated from global numpy state."""
        # Set global seed
        np.random.seed(123)
        
        # Get isolated RNG
        rng = get_rng(42)
        x1 = rng.standard_normal(5)
        
        # Global state shouldn't affect RNG
        np.random.seed(456)
        rng2 = get_rng(42)
        x2 = rng2.standard_normal(5)
        
        assert np.array_equal(x1, x2)
    
    def test_none_seed(self):
        """Test that None seed creates different sequences."""
        rng1 = get_rng(None)
        x1 = rng1.standard_normal(10)
        
        rng2 = get_rng(None)
        x2 = rng2.standard_normal(10)
        
        # Should be different (very high probability)
        assert not np.array_equal(x1, x2)


class TestCheckDeterminism:
    """Tests for check_determinism function."""
    
    def test_deterministic_function(self):
        """Test with a deterministic function."""
        def deterministic_fn():
            set_seed(42)
            return np.random.randn(10).tolist()
        
        result = check_determinism(deterministic_fn, seed=42, n_runs=3)
        assert result is True
    
    def test_non_deterministic_function(self):
        """Test with a non-deterministic function."""
        def non_deterministic_fn():
            # Doesn't set seed internally
            return np.random.randn(10).tolist()
        
        # Note: check_determinism sets seed before each call
        # So this should actually be deterministic
        result = check_determinism(non_deterministic_fn, seed=42, n_runs=3)
        assert result is True
    
    def test_dict_results(self):
        """Test with function returning dict."""
        def dict_fn():
            set_seed(42)
            return {"a": 1, "b": np.random.randn(5).tolist()}
        
        result = check_determinism(dict_fn, seed=42, n_runs=3)
        assert result is True
    
    def test_list_results(self):
        """Test with function returning list."""
        def list_fn():
            set_seed(42)
            return [1, 2, 3, np.random.randn()]
        
        result = check_determinism(list_fn, seed=42, n_runs=3)
        assert result is True
    
    def test_array_results(self):
        """Test with function returning numpy array."""
        def array_fn():
            set_seed(42)
            return np.random.randn(10)
        
        result = check_determinism(array_fn, seed=42, n_runs=3)
        assert result is True
    
    def test_single_run(self):
        """Test with n_runs=1 (always returns True)."""
        def any_fn():
            return np.random.randn()
        
        result = check_determinism(any_fn, seed=42, n_runs=1)
        assert result is True
    
    def test_many_runs(self):
        """Test with many runs."""
        def deterministic_fn():
            set_seed(42)
            return np.random.randn()
        
        result = check_determinism(deterministic_fn, seed=42, n_runs=10)
        assert result is True


@pytest.mark.parametrize("seed", [0, 1, 42, 100, 12345])
def test_set_seed_various_seeds(seed):
    """Test set_seed with various seed values."""
    set_seed(seed)
    x1 = np.random.randn(5)
    
    set_seed(seed)
    x2 = np.random.randn(5)
    
    assert np.array_equal(x1, x2)


@pytest.mark.parametrize("seed", [0, 1, 42, 100, 12345])
def test_get_rng_various_seeds(seed):
    """Test get_rng with various seed values."""
    rng1 = get_rng(seed)
    x1 = rng1.standard_normal(5)
    
    rng2 = get_rng(seed)
    x2 = rng2.standard_normal(5)
    
    assert np.array_equal(x1, x2)
