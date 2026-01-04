"""Engineering tests for training framework."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from modules._import_helper import safe_import_from

# Import module components
TrainingConfig = safe_import_from('06_deep_learning_systems.src.config', 'TrainingConfig')
SimpleMLP = safe_import_from('06_deep_learning_systems.src.models', 'SimpleMLP')
CNNMnist = safe_import_from('06_deep_learning_systems.src.models', 'CNNMnist')
Trainer = safe_import_from('06_deep_learning_systems.src.trainer', 'Trainer')
train_on_tiny_batch = safe_import_from('06_deep_learning_systems.src.trainer', 'train_on_tiny_batch')
create_tiny_dataset = safe_import_from('06_deep_learning_systems.src.datasets', 'create_tiny_dataset')
TinyDataset = safe_import_from('06_deep_learning_systems.src.datasets', 'TinyDataset')

# Import seeding utility
set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')


class TestOverfitTinyBatch:
    """
    Test 1: Model can overfit (memorize) a tiny batch.
    
    Purpose: Verify gradients are flowing correctly and model has
    sufficient capacity. If this fails, there's a fundamental issue
    with the architecture or training loop.
    
    Acceptance criteria:
    - Final loss < 0.01 (near-perfect memorization)
    - Final loss < 1% of initial loss (99% reduction)
    """
    
    def test_simple_mlp_overfits_tiny_batch(self):
        """SimpleMLP should memorize 8 samples perfectly."""
        set_seed(42)
        
        # Create tiny dataset
        X, y = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2)
        
        # Create model with sufficient capacity
        model = SimpleMLP(input_dim=10, hidden_dims=[20, 20], output_dim=2)
        
        # Train to overfit
        model, losses = train_on_tiny_batch(
            model, X, y,
            num_steps=200,
            lr=1e-2,
            device="cpu"
        )
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        # Assertions
        assert final_loss < 0.01, \
            f"Model failed to overfit: final_loss={final_loss:.4f} (expected <0.01)"
        assert final_loss < initial_loss * 0.01, \
            f"Loss reduction insufficient: {final_loss:.4f} vs {initial_loss:.4f} " \
            f"(expected 99% reduction)"
        
        # Check predictions are correct
        model.eval()
        with torch.no_grad():
            preds = model(X).argmax(dim=1)
            accuracy = (preds == y).float().mean().item()
        
        assert accuracy == 1.0, \
            f"Model didn't memorize perfectly: accuracy={accuracy:.2%} (expected 100%)"
    
    def test_cnn_overfits_tiny_images(self):
        """CNNMnist should memorize 8 images perfectly."""
        set_seed(42)
        
        # Create tiny image dataset (MNIST-like)
        X = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        
        model = CNNMnist(num_classes=10, dropout=0.0)  # No dropout for overfitting
        
        # Train to overfit
        model, losses = train_on_tiny_batch(
            model, X, y,
            num_steps=200,
            lr=1e-3,
            device="cpu"
        )
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        # Assertions
        assert final_loss < 0.01, \
            f"CNN failed to overfit: final_loss={final_loss:.4f} (expected <0.01)"
        assert final_loss < initial_loss * 0.01, \
            f"Loss reduction insufficient: {final_loss:.4f} vs {initial_loss:.4f}"


class TestGradientsFinite:
    """
    Test 2: Gradients are finite (no NaN/Inf).
    
    Purpose: Verify numerical stability - exploding/vanishing gradients
    will cause NaN/Inf values that break training.
    
    Acceptance criteria:
    - All gradients are finite (not NaN, not Inf)
    - Gradients exist for all parameters
    """
    
    def test_simple_mlp_gradients_finite(self):
        """SimpleMLP gradients should be finite."""
        set_seed(42)
        
        model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        X, y = torch.randn(4, 10), torch.randint(0, 2, (4,))
        
        # Forward + backward
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check all gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"No gradient for {name}"
            assert torch.all(torch.isfinite(param.grad)), \
                f"Non-finite gradient for {name}: {param.grad}"
            assert not torch.any(torch.isnan(param.grad)), \
                f"NaN gradient for {name}"
            assert not torch.any(torch.isinf(param.grad)), \
                f"Inf gradient for {name}"
    
    def test_cnn_gradients_finite(self):
        """CNNMnist gradients should be finite."""
        set_seed(42)
        
        model = CNNMnist(num_classes=10, dropout=0.5)
        X = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        
        # Forward + backward
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check all gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"No gradient for {name}"
            assert torch.all(torch.isfinite(param.grad)), \
                f"Non-finite gradient for {name}"
    
    def test_gradients_with_large_inputs(self):
        """Gradients should remain finite even with large inputs."""
        set_seed(42)
        
        model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        
        # Deliberately large inputs
        X = torch.randn(4, 10) * 100
        y = torch.randint(0, 2, (4,))
        
        output = model(X)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        
        # Check gradients are still finite
        for param in model.parameters():
            assert torch.all(torch.isfinite(param.grad)), \
                "Gradients became non-finite with large inputs"


class TestDeterministicTraining:
    """
    Test 3: Training is deterministic on CPU with same seed.
    
    Purpose: Ensure reproducibility - same seed should give identical
    results. Critical for debugging and scientific validity.
    
    Acceptance criteria:
    - Bit-identical losses with same seed
    - Different results with different seeds
    """
    
    def test_same_seed_gives_same_results(self):
        """Same seed should produce identical training."""
        # Run 1
        set_seed(42)
        model1 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        X, y = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2, seed=42)
        model1, losses1 = train_on_tiny_batch(model1, X, y, num_steps=10, device="cpu")
        
        # Run 2 with same seed
        set_seed(42)
        model2 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        X2, y2 = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2, seed=42)
        model2, losses2 = train_on_tiny_batch(model2, X2, y2, num_steps=10, device="cpu")
        
        # Check bit-identical
        assert torch.allclose(torch.tensor(losses1), torch.tensor(losses2), atol=1e-6), \
            f"Losses differ with same seed: {losses1[-1]:.6f} vs {losses2[-1]:.6f}"
        
        # Check model weights are identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6), \
                "Model weights differ with same seed"
    
    def test_different_seeds_give_different_results(self):
        """Different seeds should produce different training."""
        # Run 1
        set_seed(42)
        model1 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        X1, y1 = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2, seed=42)
        model1, losses1 = train_on_tiny_batch(model1, X1, y1, num_steps=10, device="cpu")
        
        # Run 2 with different seed
        set_seed(123)
        model2 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        X2, y2 = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2, seed=123)
        model2, losses2 = train_on_tiny_batch(model2, X2, y2, num_steps=10, device="cpu")
        
        # Check results differ
        assert not torch.allclose(torch.tensor(losses1), torch.tensor(losses2), atol=1e-4), \
            "Losses are identical despite different seeds"


class TestCheckpointing:
    """
    Test 4: Checkpointing saves and loads state correctly.
    
    Purpose: Verify model state can be persisted and restored for
    resumable training and inference.
    
    Acceptance criteria:
    - Loaded model produces same outputs as original
    - Training can resume from checkpoint
    """
    
    def test_save_and_load_checkpoint(self, temp_checkpoint_dir):
        """Checkpoint should restore model state exactly."""
        set_seed(42)
        
        # Create and train model
        config = TrainingConfig(
            name="test",
            checkpoint_dir=temp_checkpoint_dir,
            num_epochs=1,
        )
        model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        trainer = Trainer(config, model, device="cpu")
        
        # Get initial predictions
        X_test = torch.randn(4, 10)
        with torch.no_grad():
            original_output = model(X_test)
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pt"
        trainer.save_checkpoint("test_checkpoint.pt")
        
        # Create new model and load checkpoint
        model2 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        trainer2 = Trainer(config, model2, device="cpu")
        trainer2.load_checkpoint(checkpoint_path)
        
        # Check predictions match
        with torch.no_grad():
            loaded_output = model2(X_test)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-6), \
            "Loaded model produces different outputs"
    
    def test_checkpoint_contains_all_state(self, temp_checkpoint_dir):
        """Checkpoint should contain all necessary state."""
        config = TrainingConfig(
            name="test",
            checkpoint_dir=temp_checkpoint_dir,
        )
        model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        trainer = Trainer(config, model, device="cpu")
        
        # Save checkpoint
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pt"
        trainer.save_checkpoint("test_checkpoint.pt")
        
        # Load and check contents
        checkpoint = torch.load(checkpoint_path)
        
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        assert 'best_val_loss' in checkpoint
        assert 'history' in checkpoint
        assert 'config' in checkpoint


class TestEarlyStopping:
    """
    Test 5: Early stopping triggers correctly.
    
    Purpose: Verify training stops when validation loss plateaus,
    preventing overfitting and wasted compute.
    
    Acceptance criteria:
    - Training stops before max epochs when loss plateaus
    - Best model is saved before stopping
    """
    
    def test_early_stopping_triggers(self, temp_checkpoint_dir, tiny_dataset):
        """Early stopping should trigger when val loss doesn't improve."""
        set_seed(42)
        
        X, y = tiny_dataset
        dataset = TinyDataset(X, y)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset, batch_size=8)
        val_loader = DataLoader(dataset, batch_size=8)  # Same data for testing
        
        config = TrainingConfig(
            name="test_early_stop",
            checkpoint_dir=temp_checkpoint_dir,
            num_epochs=50,  # Set high, but should stop early
            early_stop_patience=3,  # Stop after 3 epochs without improvement
            learning_rate=1e-2,
            model_type="SimpleMLP",
            input_dim=10,
            hidden_dims=[20],
            output_dim=2,
        )
        
        model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        trainer = Trainer(config, model, device="cpu")
        
        # Train (should overfit quickly and trigger early stopping)
        history = trainer.train(train_loader, val_loader)
        
        # Check early stopping triggered
        epochs_trained = len(history['train_loss'])
        assert epochs_trained < 50, \
            f"Early stopping didn't trigger (trained {epochs_trained}/50 epochs)"
        
        # Best model should be saved
        best_checkpoint = temp_checkpoint_dir / "best_model.pt"
        assert best_checkpoint.exists(), \
            "Best model checkpoint not saved"


class TestLossDecreasing:
    """
    Test 6: Training loss decreases monotonically.
    
    Purpose: Sanity check that optimizer is working - loss should
    decrease over training.
    
    Acceptance criteria:
    - Final loss < initial loss
    - Loss decreases by at least 50%
    """
    
    def test_training_decreases_loss(self, tiny_dataset):
        """Training should decrease loss."""
        set_seed(42)
        
        X, y = tiny_dataset
        
        model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        model, losses = train_on_tiny_batch(model, X, y, num_steps=50, device="cpu")
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.4f} → {final_loss:.4f}"
        assert final_loss < initial_loss * 0.5, \
            f"Loss reduction insufficient: {initial_loss:.4f} → {final_loss:.4f} " \
            f"(expected >50% reduction)"


class TestModelShapes:
    """
    Test 7: Model outputs have correct shapes.
    
    Purpose: Verify architecture produces expected tensor dimensions.
    
    Acceptance criteria:
    - Output shape matches expected (batch_size, num_classes)
    - Works with variable batch sizes
    """
    
    def test_simple_mlp_output_shape(self):
        """SimpleMLP should produce correct output shape."""
        model = SimpleMLP(input_dim=10, hidden_dims=[20, 15], output_dim=3)
        
        # Test with batch size 1
        x1 = torch.randn(1, 10)
        out1 = model(x1)
        assert out1.shape == (1, 3), f"Wrong shape: {out1.shape}"
        
        # Test with batch size 32
        x32 = torch.randn(32, 10)
        out32 = model(x32)
        assert out32.shape == (32, 3), f"Wrong shape: {out32.shape}"
    
    def test_cnn_mnist_output_shape(self):
        """CNNMnist should produce correct output shape."""
        model = CNNMnist(num_classes=10)
        
        # Test with batch size 1
        x1 = torch.randn(1, 1, 28, 28)
        out1 = model(x1)
        assert out1.shape == (1, 10), f"Wrong shape: {out1.shape}"
        
        # Test with batch size 64
        x64 = torch.randn(64, 1, 28, 28)
        out64 = model(x64)
        assert out64.shape == (64, 10), f"Wrong shape: {out64.shape}"
    
    def test_flattening_handles_batches(self):
        """SimpleMLP should auto-flatten image inputs."""
        model = SimpleMLP(input_dim=784, hidden_dims=[128], output_dim=10)
        
        # Input as flattened
        x_flat = torch.randn(32, 784)
        out_flat = model(x_flat)
        
        # Input as image (should be auto-flattened)
        x_img = torch.randn(32, 1, 28, 28)
        out_img = model(x_img)
        
        # Both should produce same shape
        assert out_flat.shape == out_img.shape == (32, 10)
