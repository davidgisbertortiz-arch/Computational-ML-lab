# Module 06: Deep Learning Systems

This directory contains the expected outputs from training experiments.

## Directory Structure

```
reports/
├── checkpoints/          # Model checkpoints
│   ├── best_model.pt    # Best validation loss checkpoint
│   └── last_model.pt    # Final epoch checkpoint
├── learning_curves.png   # Training/validation curves
└── metrics.json          # Final metrics summary
```

## Expected Outputs

### 1. Checkpoints (`checkpoints/`)

Each checkpoint contains:
- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state (for resumable training)
- `epoch`: Training epoch number
- `best_val_loss`: Best validation loss achieved
- `history`: Training history (losses, accuracies)
- `config`: Training configuration

**Usage**:
```bash
# Resume training
python -m modules.06_deep_learning_systems.src.main train \
    --config configs/mnist_baseline.yaml \
    --resume reports/checkpoints/last_model.pt

# Evaluate
python -m modules.06_deep_learning_systems.src.main evaluate \
    --checkpoint reports/checkpoints/best_model.pt \
    --config configs/mnist_baseline.yaml
```

### 2. Learning Curves (`learning_curves.png`)

Two-panel plot showing:
- **Left panel**: Training and validation loss vs epoch
  - Training loss should decrease monotonically
  - Validation loss should decrease initially, may plateau
  - Gap between curves indicates overfitting
  
- **Right panel**: Validation accuracy vs epoch
  - Should increase and plateau near 98% for MNIST
  - Should increase and plateau near 85-90% for FashionMNIST

**Expected patterns**:
- **Healthy training**: Both losses decrease, accuracy increases
- **Overfitting**: Train loss << val loss, accuracy plateaus/decreases
- **Underfitting**: Both losses high, accuracy low
- **Early stopping**: Curves stop before max epochs

### 3. Metrics (`metrics.json`)

Final metrics summary:
```json
{
  "final_train_loss": 0.05,
  "final_val_loss": 0.12,
  "best_val_accuracy": 0.982,
  "best_epoch": 7,
  "total_epochs": 10,
  "test_loss": 0.13,
  "test_accuracy": 0.980,
  "config": { ... }
}
```

**Expected ranges** (MNIST, 10 epochs):
- `test_accuracy`: 0.97 - 0.99 (97-99%)
- `test_loss`: 0.05 - 0.15
- `total_epochs`: 5-10 (may trigger early stopping)

**Expected ranges** (FashionMNIST, 15 epochs):
- `test_accuracy`: 0.85 - 0.90 (85-90%)
- `test_loss`: 0.20 - 0.40
- `total_epochs`: 10-15

## Baseline Results

### MNIST (CNNMnist, 10 epochs)
- **Test Accuracy**: ~98%
- **Training Time**: ~2-3 min (CPU), ~30 sec (GPU)
- **Parameters**: ~100k
- **Typical Early Stop**: Epoch 7-8

### FashionMNIST (CNNMnist, 15 epochs)
- **Test Accuracy**: ~88%
- **Training Time**: ~3-4 min (CPU), ~45 sec (GPU)
- **Parameters**: ~100k
- **Typical Early Stop**: Epoch 10-12

### SimpleMLP (MNIST, 5 epochs)
- **Test Accuracy**: ~96-97%
- **Training Time**: ~1-2 min (CPU)
- **Parameters**: ~200k
- **Typical Early Stop**: Epoch 4-5

## Troubleshooting

### Low accuracy (<90% on MNIST)
- Check learning rate (try 1e-3 to 1e-4)
- Increase epochs or reduce early_stop_patience
- Check data normalization

### Training too slow
- Reduce num_workers if CPU bottlenecked
- Use GPU (`device: cuda`)
- Reduce batch_size if OOM
- Use SimpleMLP instead of CNN for fast iteration

### Model not overfitting tiny batch
- Check gradient flow (test_gradients_finite)
- Increase model capacity
- Remove regularization (dropout, weight_decay)
- Check loss function matches task

### Non-reproducible results
- Ensure `set_seed()` called before training
- Use CPU device for bit-identical results
- Check for non-deterministic ops (GPU atomics)
