"""Tests for utility functions."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from modules._import_helper import safe_import_from

set_seed, load_config, log_experiment, save_results = safe_import_from(
    '00_repo_standards.src.utils',
    'set_seed', 'load_config', 'log_experiment', 'save_results'
)


class TestSetSeed:
    """Tests for set_seed function."""
    
    def test_reproducible_numpy(self):
        """Test that numpy random is reproducible with same seed."""
        set_seed(42)
        x1 = np.random.randn(5)
        
        set_seed(42)
        x2 = np.random.randn(5)
        
        assert np.array_equal(x1, x2)
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        x1 = np.random.randn(5)
        
        set_seed(43)
        x2 = np.random.randn(5)
        
        assert not np.array_equal(x1, x2)


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_valid_config(self):
        """Test loading a valid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("seed: 42\nlr: 0.01\nmax_iter: 100\n")
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert config["seed"] == 42
            assert config["lr"] == 0.01
            assert config["max_iter"] == 100
        finally:
            config_path.unlink()
    
    def test_load_nested_config(self):
        """Test loading config with nested structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("optimizer:\n  lr: 0.01\n  max_iter: 100\n")
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert config["optimizer"]["lr"] == 0.01
            assert config["optimizer"]["max_iter"] == 100
        finally:
            config_path.unlink()
    
    def test_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        config_path = Path("nonexistent_config.yaml")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(config_path)


class TestLogExperiment:
    """Tests for log_experiment function."""
    
    def test_logs_without_error(self, capsys):
        """Test that logging doesn't raise errors."""
        log_experiment(
            experiment_name="test",
            params={"lr": 0.01, "seed": 42},
            metrics={"loss": 0.15},
        )
        
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "lr" in captured.out
        assert "loss" in captured.out


class TestSaveResults:
    """Tests for save_results function."""
    
    def test_save_and_load(self):
        """Test saving results to YAML and loading back."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.yaml"
            results = {"loss": 0.15, "accuracy": 0.95}
            
            save_results(results, output_path)
            
            assert output_path.exists()
            loaded = load_config(output_path)
            assert loaded["loss"] == 0.15
            assert loaded["accuracy"] == 0.95
    
    def test_creates_parent_dirs(self):
        """Test that parent directories are created if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "results.yaml"
            results = {"loss": 0.15}
            
            save_results(results, output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()
