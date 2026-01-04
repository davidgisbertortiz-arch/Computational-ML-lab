# Module 06: Deep Learning Systems

**Focus**: Engineering a production-grade PyTorch training framework with best practices.

## Learning Objectives

After completing this module, you will understand:

1. **Training Framework Design**:
   - Config-driven training loops (Pydantic validation)
   - Checkpointing and model persistence
   - Mixed precision training (AMP)
   - Early stopping strategies
   - Comprehensive metric logging

2. **Engineering Best Practices**:
   - Overfitting sanity checks (tiny batch memorization)
   - Gradient finite difference tests
   - Deterministic training on CPU
   - Learning curve visualization
   - Reproducibility guarantees

3. **Production Patterns**:
   - Separation of concerns (trainer/model/data)
   - Stateful training resumption
   - Device-agnostic code (CPU/GPU)
   - Hyperparameter management

## Module Structure

```
06_deep_learning_systems/
├── src/
│   ├── __init__.py
│   ├── config.py           # TrainingConfig (Pydantic)
│   ├── trainer.py          # Core Trainer class
│   ├── models.py           # SimpleMLP, CNNMnist
│   ├── datasets.py         # MNIST + tabular loaders
│   └── main.py             # CLI: train_torch, evaluate_torch
├── tests/
│   ├── test_training.py    # Engineering tests (overfit, gradients, determinism)
│   ├── test_checkpointing.py
│   └── test_models.py
├── configs/
│   ├── mnist_baseline.yaml
│   └── tabular_mlp.yaml
├── notebooks/
│   └── 01_training_framework_walkthrough.ipynb
└── reports/
    ├── mnist_learning_curves.png
    └── metrics.json
```

## Quick Start

### 1. Train MNIST baseline
```bash
python -m modules.06_deep_learning_systems.src.main train \
    --config modules/06_deep_learning_systems/configs/mnist_baseline.yaml \
    --seed 42
```

### 2. Run engineering tests
```bash
pytest modules/06_deep_learning_systems/tests/test_training.py -v
```

### 3. Evaluate checkpointed model
```bash
python -m modules.06_deep_learning_systems.src.main evaluate \
    --checkpoint modules/06_deep_learning_systems/reports/checkpoints/best_model.pt \
    --config modules/06_deep_learning_systems/configs/mnist_baseline.yaml
```

## Key Components

### `TrainingConfig` (Pydantic)
```python
class TrainingConfig(ExperimentConfig):
    # Model
    model_type: str = "SimpleMLP"
    hidden_dims: List[int] = [128, 64]
    
    # Training
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # Engineering
    use_amp: bool = False  # Mixed precision
    gradient_clip_norm: Optional[float] = None
    early_stop_patience: int = 5
    checkpoint_every: int = 1  # Save every N epochs
```

### `Trainer` Class
```python
trainer = Trainer(config, model, device)
trainer.train(train_loader, val_loader)
trainer.save_checkpoint(path)
trainer.load_checkpoint(path)
metrics = trainer.evaluate(test_loader)
```

## Engineering Tests

### 1. Overfit Tiny Batch
**Purpose**: Verify model can memorize 8 samples (sanity check for gradient flow).

```python
def test_overfit_tiny_batch():
    model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    X, y = create_tiny_dataset(n=8)
    trainer = Trainer(config, model, device="cpu")
    
    initial_loss = compute_loss(model, X, y)
    trainer.train_on_batch(X, y, num_steps=200)
    final_loss = compute_loss(model, X, y)
    
    assert final_loss < 0.01  # Should memorize perfectly
    assert final_loss < initial_loss * 0.01  # 99% reduction
```

### 2. Gradient Finite Difference Check
**Purpose**: Validate autograd correctness.

```python
def test_gradients_are_finite():
    model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    X, y = torch.randn(4, 10), torch.randint(0, 2, (4,))
    
    output = model(X)
    loss = F.cross_entropy(output, y)
    loss.backward()
    
    for param in model.parameters():
        assert torch.all(torch.isfinite(param.grad))
        assert not torch.any(torch.isnan(param.grad))
```

### 3. Deterministic CPU Training
**Purpose**: Ensure reproducibility with same seed.

```python
def test_deterministic_training():
    set_seed(42)
    model1 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    losses1 = train_n_steps(model1, data, steps=10)
    
    set_seed(42)
    model2 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    losses2 = train_n_steps(model2, data, steps=10)
    
    assert torch.allclose(losses1, losses2)  # Bit-identical
```

## Expected Outputs

### reports/mnist_learning_curves.png
- Training loss vs epoch (should decrease monotonically)
- Validation accuracy vs epoch (should plateau ~98%)
- Clear separation if overfitting occurs

### reports/metrics.json
```json
{
  "final_train_loss": 0.05,
  "final_val_loss": 0.12,
  "best_val_accuracy": 0.982,
  "best_epoch": 7,
  "total_epochs": 10,
  "training_time_seconds": 45.3
}
```

### reports/checkpoints/
- `best_model.pt`: Best validation loss checkpoint
- `last_model.pt`: Final epoch checkpoint
- Includes: model state_dict, optimizer state_dict, epoch, metrics

## Implementation Checklist

- [ ] **Core Framework**
  - [ ] `config.py`: TrainingConfig with Pydantic validation
  - [ ] `trainer.py`: Trainer class with train/eval/checkpoint methods
  - [ ] `models.py`: SimpleMLP, CNNMnist
  - [ ] `datasets.py`: MNIST loaders with transforms

- [ ] **Engineering Tests** (test_training.py)
  - [ ] Overfit tiny batch (<0.01 loss on 8 samples)
  - [ ] Gradient finite checks (no NaN/Inf)
  - [ ] Deterministic CPU training (bit-identical with same seed)
  - [ ] Checkpointing saves/loads state correctly
  - [ ] Early stopping triggers when val loss plateaus

- [ ] **CLI Commands** (main.py)
  - [ ] `train`: Full training loop with logging
  - [ ] `evaluate`: Load checkpoint and compute test metrics
  - [ ] Both accept --config, --seed, --device

- [ ] **Optional Features**
  - [ ] Mixed precision (torch.cuda.amp.autocast)
  - [ ] Gradient clipping (torch.nn.utils.clip_grad_norm_)
  - [ ] Learning rate scheduling
  - [ ] TensorBoard logging

- [ ] **Validation**
  - [ ] MNIST baseline achieves >97% test accuracy in <5 min
  - [ ] All engineering tests pass (pytest)
  - [ ] Learning curves saved to reports/
  - [ ] README documents all components

## References

- PyTorch Training Loop: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Mixed Precision: https://pytorch.org/docs/stable/amp.html
- Checkpointing Best Practices: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

## Notes

- **Why MNIST?** Fast sanity check for framework correctness (~10k samples, 28x28 images).
- **Why engineering tests?** Catch common bugs (dead ReLUs, exploding gradients, non-reproducible seeds).
- **Why config-driven?** Enables hyperparameter sweeps without code changes.
- **Device handling**: Always use `tensor.to(device)` for CPU/GPU compatibility.

## Time Estimate
- Implementation: 3-4 hours
- Testing: 1 hour
- Documentation: 30 min
- **Total: ~5 hours**
