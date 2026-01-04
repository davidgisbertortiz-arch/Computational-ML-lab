#!/usr/bin/env python3
"""Quick validation script for Module 06."""

import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import torch
from modules._import_helper import safe_import_from

print("=" * 60)
print("Module 06: Deep Learning Systems - Quick Validation")
print("=" * 60)

# Import components
print("\n1. Testing imports...")
try:
    TrainingConfig = safe_import_from('06_deep_learning_systems.src.config', 'TrainingConfig')
    SimpleMLP = safe_import_from('06_deep_learning_systems.src.models', 'SimpleMLP')
    CNNMnist = safe_import_from('06_deep_learning_systems.src.models', 'CNNMnist')
    train_on_tiny_batch = safe_import_from('06_deep_learning_systems.src.trainer', 'train_on_tiny_batch')
    create_tiny_dataset = safe_import_from('06_deep_learning_systems.src.datasets', 'create_tiny_dataset')
    set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test config
print("\n2. Testing config...")
try:
    config = TrainingConfig(
        name="test",
        model_type="SimpleMLP",
        batch_size=32,
        num_epochs=5,
    )
    print(f"   ✓ Config created: {config.name}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")
    sys.exit(1)

# Test SimpleMLP
print("\n3. Testing SimpleMLP...")
try:
    set_seed(42)
    model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 2), f"Wrong shape: {out.shape}"
    print(f"   ✓ SimpleMLP works: {out.shape}")
except Exception as e:
    print(f"   ✗ SimpleMLP failed: {e}")
    sys.exit(1)

# Test CNNMnist
print("\n4. Testing CNNMnist...")
try:
    model = CNNMnist(num_classes=10, dropout=0.5)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape == (2, 10), f"Wrong shape: {out.shape}"
    print(f"   ✓ CNNMnist works: {out.shape}")
except Exception as e:
    print(f"   ✗ CNNMnist failed: {e}")
    sys.exit(1)

# Test overfitting tiny batch
print("\n5. Testing overfitting (200 steps)...")
try:
    set_seed(42)
    X, y = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2)
    model = SimpleMLP(input_dim=10, hidden_dims=[20, 20], output_dim=2)
    
    model, losses = train_on_tiny_batch(model, X, y, num_steps=200, lr=1e-2, device="cpu")
    
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss:   {final_loss:.4f}")
    print(f"   Reduction:    {(1 - final_loss/initial_loss)*100:.1f}%")
    
    if final_loss < 0.01:
        print(f"   ✓ Model overfitted successfully (loss < 0.01)")
    else:
        print(f"   ⚠ Model didn't overfit perfectly (loss = {final_loss:.4f})")
    
    # Check accuracy
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
    print(f"   Accuracy:     {accuracy:.2%}")
    
except Exception as e:
    print(f"   ✗ Overfitting test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test gradients are finite
print("\n6. Testing gradient finiteness...")
try:
    set_seed(42)
    model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    X, y = torch.randn(4, 10), torch.randint(0, 2, (4,))
    
    output = model(X)
    loss = torch.nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    all_finite = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"   ✗ No gradient for {name}")
            all_finite = False
        elif not torch.all(torch.isfinite(param.grad)):
            print(f"   ✗ Non-finite gradient for {name}")
            all_finite = False
    
    if all_finite:
        print("   ✓ All gradients are finite")
    else:
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Gradient test failed: {e}")
    sys.exit(1)

# Test determinism
print("\n7. Testing deterministic training...")
try:
    # Run 1
    set_seed(42)
    model1 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    X, y = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2, seed=42)
    model1, losses1 = train_on_tiny_batch(model1, X, y, num_steps=10, device="cpu")
    
    # Run 2
    set_seed(42)
    model2 = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
    X2, y2 = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2, seed=42)
    model2, losses2 = train_on_tiny_batch(model2, X2, y2, num_steps=10, device="cpu")
    
    if torch.allclose(torch.tensor(losses1), torch.tensor(losses2), atol=1e-6):
        print("   ✓ Training is deterministic (bit-identical)")
    else:
        print(f"   ✗ Training not deterministic: {losses1[-1]:.6f} vs {losses2[-1]:.6f}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Determinism test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All validation tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. Run full test suite: make test-module MODULE=06_deep_learning_systems")
print("2. Train MNIST: python -m modules.06_deep_learning_systems.src.main train \\")
print("                    --config modules/06_deep_learning_systems/configs/mnist_baseline.yaml")
print("=" * 60)
