#!/usr/bin/env python
"""Quick test script to verify module 00 implementation."""

import sys
from pathlib import Path

import numpy as np

from modules._import_helper import safe_import_from

print("=" * 60)
print("Testing Module 00: Repository Standards")
print("=" * 60)
print()

# Test 1: Import mlphys_core
print("✓ Test 1: Importing mlphys_core...")
try:
    ExperimentConfig, load_config, set_seed, setup_logger, BaseExperiment = safe_import_from(
        "00_repo_standards.src.mlphys_core",
        "ExperimentConfig",
        "load_config",
        "set_seed",
        "setup_logger",
        "BaseExperiment",
    )
    print("  SUCCESS: All imports working")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 2: Config creation
print("\n✓ Test 2: Creating configuration...")
try:
    config = ExperimentConfig(
        name="test",
        seed=42,
        output_dir=Path("test_output"),
    )
    assert config.name == "test"
    assert config.seed == 42
    print("  SUCCESS: Config created and validated")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 3: Seeding
print("\n✓ Test 3: Testing deterministic seeding...")
try:
    set_seed(42)
    x1 = np.random.randn(10)

    set_seed(42)
    x2 = np.random.randn(10)

    assert np.array_equal(x1, x2), "Seeding not deterministic!"
    print("  SUCCESS: Seeding is deterministic")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 4: Logging
print("\n✓ Test 4: Testing logging setup...")
try:
    logger = setup_logger("test_logger", level="INFO")
    logger.info("Test message")
    print("  SUCCESS: Logger configured")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 5: Demo experiment import
print("\n✓ Test 5: Importing demo experiment...")
try:
    DemoConfig, DemoExperiment = safe_import_from(
        "00_repo_standards.src.demo_experiment",
        "DemoConfig",
        "DemoExperiment",
    )
    print("  SUCCESS: Demo experiment imports working")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Test 6: Demo config validation
print("\n✓ Test 6: Creating demo configuration...")
try:
    demo_config = DemoConfig(
        name="quick_test",
        seed=42,
        n_samples=100,
        n_features=10,
    )
    assert demo_config.n_samples == 100
    assert demo_config.C == 1.0  # default value
    print("  SUCCESS: Demo config validated")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✓ All quick tests passed!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Run full test suite: pytest modules/00_repo_standards/tests/ -v")
print("  2. Run demo: python -m modules.run demo --module 00 --seed 42")
print("  3. Check linting: ruff check modules/00_repo_standards/")
print("  4. Format code: black modules/00_repo_standards/")
