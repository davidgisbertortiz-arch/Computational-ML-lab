"""Integration tests for the demo experiment."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from modules._import_helper import safe_import_from

DemoConfig, DemoExperiment = safe_import_from('00_repo_standards.src.demo_experiment', 'DemoConfig', 'DemoExperiment')
set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')


class TestDemoExperimentReproducibility:
    """Test that demo experiment is fully reproducible."""
    
    def test_same_seed_same_metrics(self):
        """Test that same seed produces identical metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DemoConfig(
                name="repro_test",
                seed=42,
                output_dir=Path(tmpdir) / "run1",
                save_artifacts=False,
            )
            
            exp1 = DemoExperiment(config)
            metrics1 = exp1.run()
            
            # Run again with same seed
            config2 = DemoConfig(
                name="repro_test",
                seed=42,
                output_dir=Path(tmpdir) / "run2",
                save_artifacts=False,
            )
            
            exp2 = DemoExperiment(config2)
            metrics2 = exp2.run()
            
            # Metrics should be identical
            assert metrics1 == metrics2
    
    def test_different_seeds_different_metrics(self):
        """Test that different seeds produce different metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = DemoConfig(
                name="test1",
                seed=42,
                output_dir=Path(tmpdir) / "run1",
                save_artifacts=False,
            )
            
            exp1 = DemoExperiment(config1)
            metrics1 = exp1.run()
            
            config2 = DemoConfig(
                name="test2",
                seed=43,
                output_dir=Path(tmpdir) / "run2",
                save_artifacts=False,
            )
            
            exp2 = DemoExperiment(config2)
            metrics2 = exp2.run()
            
            # Metrics should differ (with high probability)
            assert metrics1 != metrics2
    
    def test_multiple_runs_same_seed(self):
        """Test reproducibility across multiple runs."""
        n_runs = 3
        all_metrics = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_runs):
                config = DemoConfig(
                    name=f"run_{i}",
                    seed=42,
                    output_dir=Path(tmpdir) / f"run_{i}",
                    save_artifacts=False,
                )
                
                exp = DemoExperiment(config)
                metrics = exp.run()
                all_metrics.append(metrics)
            
            # All metrics should be identical
            for i in range(1, n_runs):
                assert all_metrics[i] == all_metrics[0]


class TestDemoExperimentOutputs:
    """Test that demo experiment produces expected outputs."""
    
    def test_creates_output_directory(self):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            
            config = DemoConfig(
                name="test",
                seed=42,
                output_dir=output_dir,
                save_artifacts=True,
            )
            
            exp = DemoExperiment(config)
            exp.run()
            
            assert output_dir.exists()
            assert output_dir.is_dir()
    
    def test_saves_config(self):
        """Test that config is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            
            config = DemoConfig(
                name="test",
                seed=42,
                output_dir=output_dir,
                save_artifacts=True,
            )
            
            exp = DemoExperiment(config)
            exp.run()
            
            config_path = output_dir / "config.yaml"
            assert config_path.exists()
    
    def test_saves_metrics_json(self):
        """Test that metrics are saved to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            
            config = DemoConfig(
                name="test",
                seed=42,
                output_dir=output_dir,
                save_artifacts=True,
            )
            
            exp = DemoExperiment(config)
            metrics = exp.run()
            
            metrics_path = output_dir / "metrics.json"
            assert metrics_path.exists()
            
            # Load and verify
            with open(metrics_path) as f:
                saved_metrics = json.load(f)
            
            assert "metrics" in saved_metrics
            assert "timestamp" in saved_metrics
            assert saved_metrics["metrics"] == metrics
    
    def test_saves_confusion_matrix_plot(self):
        """Test that confusion matrix plot is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            
            config = DemoConfig(
                name="test",
                seed=42,
                output_dir=output_dir,
                save_artifacts=True,
            )
            
            exp = DemoExperiment(config)
            exp.run()
            
            plot_path = output_dir / "figures" / "confusion_matrix.png"
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0  # File is not empty
    
    def test_saves_log_file(self):
        """Test that log file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            
            config = DemoConfig(
                name="test_log",
                seed=42,
                output_dir=output_dir,
                save_artifacts=True,
            )
            
            exp = DemoExperiment(config)
            exp.run()
            
            log_path = output_dir / "test_log.log"
            assert log_path.exists()
            
            # Check log contains expected content
            with open(log_path) as f:
                log_content = f.read()
            
            assert "Set random seed" in log_content
            assert "Preparing data" in log_content
            assert "Building model" in log_content
            assert "Training model" in log_content
            assert "Evaluating model" in log_content


class TestDemoExperimentMetrics:
    """Test that metrics are within expected ranges."""
    
    def test_metrics_structure(self):
        """Test that returned metrics have expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DemoConfig(
                name="test",
                seed=42,
                output_dir=Path(tmpdir),
                save_artifacts=False,
            )
            
            exp = DemoExperiment(config)
            metrics = exp.run()
            
            # Check expected keys
            assert "train_accuracy" in metrics
            assert "test_accuracy" in metrics
            assert "test_f1_score" in metrics
            assert "n_train" in metrics
            assert "n_test" in metrics
            assert "n_iterations" in metrics
    
    def test_accuracy_in_valid_range(self):
        """Test that accuracy values are between 0 and 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DemoConfig(
                name="test",
                seed=42,
                output_dir=Path(tmpdir),
                save_artifacts=False,
            )
            
            exp = DemoExperiment(config)
            metrics = exp.run()
            
            assert 0 <= metrics["train_accuracy"] <= 1
            assert 0 <= metrics["test_accuracy"] <= 1
            assert 0 <= metrics["test_f1_score"] <= 1
    
    def test_sample_counts(self):
        """Test that sample counts match configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DemoConfig(
                name="test",
                seed=42,
                n_samples=500,
                test_size=0.2,
                output_dir=Path(tmpdir),
                save_artifacts=False,
            )
            
            exp = DemoExperiment(config)
            metrics = exp.run()
            
            assert metrics["n_train"] == 400  # 80% of 500
            assert metrics["n_test"] == 100   # 20% of 500
            assert metrics["n_train"] + metrics["n_test"] == config.n_samples


class TestDemoExperimentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DemoConfig(
                name="test",
                seed=42,
                n_samples=50,
                n_features=5,
                n_informative=3,  # Ensure n_redundant = 5 - 3 = 2 >= 0
                output_dir=Path(tmpdir),
                save_artifacts=False,
            )
            
            exp = DemoExperiment(config)
            metrics = exp.run()
            
            assert metrics is not None
            assert "test_accuracy" in metrics
    
    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DemoConfig(
                name="test",
                seed=42,
                n_samples=200,
                n_features=100,
                n_informative=50,
                output_dir=Path(tmpdir),
                save_artifacts=False,
            )
            
            exp = DemoExperiment(config)
            metrics = exp.run()
            
            assert metrics is not None
            assert "test_accuracy" in metrics


@pytest.mark.parametrize("seed", [0, 1, 42, 100, 12345])
def test_different_seeds_all_work(seed):
    """Test that experiment works with various seeds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DemoConfig(
            name="test",
            seed=seed,
            output_dir=Path(tmpdir),
            save_artifacts=False,
        )
        
        exp = DemoExperiment(config)
        metrics = exp.run()
        
        assert metrics is not None
        assert 0 <= metrics["test_accuracy"] <= 1
