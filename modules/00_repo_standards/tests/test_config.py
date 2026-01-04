"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from modules._import_helper import safe_import_from

ExperimentConfig, load_config, save_config = safe_import_from(
    '00_repo_standards.src.mlphys_core.config',
    'ExperimentConfig', 'load_config', 'save_config'
)


class TestExperimentConfig:
    """Tests for ExperimentConfig model."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ExperimentConfig(name="test")
        
        assert config.name == "test"
        assert config.description == ""
        assert config.seed == 42
        assert config.output_dir == Path("reports")
        assert config.save_artifacts is True
        assert config.log_level == "INFO"
        assert config.mlflow_tracking is False
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = ExperimentConfig(
            name="custom",
            description="Custom experiment",
            seed=123,
            output_dir=Path("custom_output"),
            log_level="DEBUG",
        )
        
        assert config.name == "custom"
        assert config.description == "Custom experiment"
        assert config.seed == 123
        assert config.output_dir == Path("custom_output")
        assert config.log_level == "DEBUG"
    
    def test_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = ExperimentConfig(
            name="test",
            output_dir="string/path",
        )
        
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("string/path")
    
    def test_log_level_validation(self):
        """Test that log level is validated."""
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = ExperimentConfig(name="test", log_level=level)
            assert config.log_level == level
        
        # Case insensitive
        config = ExperimentConfig(name="test", log_level="info")
        assert config.log_level == "INFO"
        
        # Invalid level
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test", log_level="INVALID")
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        config = ExperimentConfig(
            name="test",
            custom_field="custom_value",
            another_field=123,
        )
        
        assert config.custom_field == "custom_value"
        assert config.another_field == 123


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_valid_config(self):
        """Test loading a valid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: test\nseed: 42\nlog_level: DEBUG\n")
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert config.name == "test"
            assert config.seed == 42
            assert config.log_level == "DEBUG"
        finally:
            config_path.unlink()
    
    def test_load_nested_config(self):
        """Test loading config with nested structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: test\nseed: 42\noutput_dir: reports/test\n")
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path)
            assert config.output_dir == Path("reports/test")
        finally:
            config_path.unlink()
    
    def test_load_custom_config_class(self):
        """Test loading with a custom config class."""
        class CustomConfig(ExperimentConfig):
            learning_rate: float = 0.01
            n_epochs: int = 100
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: test\nlearning_rate: 0.001\nn_epochs: 50\n")
            config_path = Path(f.name)
        
        try:
            config = load_config(config_path, CustomConfig)
            assert isinstance(config, CustomConfig)
            assert config.learning_rate == 0.001
            assert config.n_epochs == 50
        finally:
            config_path.unlink()
    
    def test_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        config_path = Path("nonexistent_config.yaml")
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(config_path)
    
    def test_invalid_yaml_raises(self):
        """Test that invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: test\ninvalid yaml: {[}]\n")
            config_path = Path(f.name)
        
        try:
            with pytest.raises(Exception):  # yaml.YAMLError
                load_config(config_path)
        finally:
            config_path.unlink()
    
    def test_validation_error(self):
        """Test that invalid config raises ValidationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("name: test\nlog_level: INVALID\n")
            config_path = Path(f.name)
        
        try:
            with pytest.raises(ValidationError):
                load_config(config_path)
        finally:
            config_path.unlink()


class TestSaveConfig:
    """Tests for save_config function."""
    
    def test_save_and_reload(self):
        """Test saving config and reloading it."""
        config = ExperimentConfig(
            name="test",
            seed=123,
            log_level="DEBUG",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config, config_path)
            
            assert config_path.exists()
            
            # Reload
            loaded_config = load_config(config_path)
            assert loaded_config.name == config.name
            assert loaded_config.seed == config.seed
            assert loaded_config.log_level == config.log_level
    
    def test_creates_parent_dirs(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.yaml"
            config = ExperimentConfig(name="test")
            
            save_config(config, config_path)
            
            assert config_path.exists()
            assert config_path.parent.exists()
    
    def test_path_objects_converted(self):
        """Test that Path objects are converted to strings."""
        config = ExperimentConfig(
            name="test",
            output_dir=Path("reports/test"),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config, config_path)
            
            # Read raw YAML
            with open(config_path) as f:
                content = f.read()
            
            # Should contain string representation
            assert "reports/test" in content
            # Should not contain "Path" or "PosixPath"
            assert "Path" not in content


class TestCustomConfigSubclass:
    """Tests for creating custom config subclasses."""
    
    def test_subclass_with_extra_fields(self):
        """Test creating a subclass with additional fields."""
        class ModelConfig(ExperimentConfig):
            learning_rate: float = Field(default=0.01, description="Learning rate")
            n_layers: int = Field(default=3, description="Number of layers")
            activation: str = Field(default="relu", description="Activation function")
        
        config = ModelConfig(
            name="model_exp",
            learning_rate=0.001,
            n_layers=5,
        )
        
        assert config.name == "model_exp"
        assert config.learning_rate == 0.001
        assert config.n_layers == 5
        assert config.activation == "relu"
    
    def test_subclass_save_and_load(self):
        """Test saving and loading a custom config subclass."""
        class ModelConfig(ExperimentConfig):
            learning_rate: float = 0.01
            batch_size: int = 32
        
        config = ModelConfig(
            name="test",
            learning_rate=0.001,
            batch_size=64,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config, config_path)
            
            loaded = load_config(config_path, ModelConfig)
            assert loaded.learning_rate == 0.001
            assert loaded.batch_size == 64
