# Optimizer Benchmark Experiment

## Purpose

Reproducible benchmark comparing GD, Momentum, and Adam optimizers on:
1. **Quadratic bowl (κ=5)** – well-conditioned, all optimizers should converge quickly
2. **Quadratic bowl (κ=100)** – ill-conditioned, tests optimizer robustness
3. **Logistic regression** – realistic ML problem with 200 samples, 10 features

## Quick Start

```bash
# Run with defaults
python -m modules.01_numerical_toolbox.experiments.optimizer_benchmark run

# Custom settings
python -m modules.01_numerical_toolbox.experiments.optimizer_benchmark run \
    --max-iter 500 \
    --seed 123 \
    --mlflow-experiment "my-benchmark"

# Without MLflow
python -m modules.01_numerical_toolbox.experiments.optimizer_benchmark run --no-mlflow
```

## Outputs

| File | Description |
|------|-------------|
| `reports/benchmark/convergence_comparison.png` | Side-by-side convergence plots |
| `reports/benchmark/summary_table.txt` | Formatted results table |
| `reports/benchmark/benchmark_results.json` | Raw results for analysis |
| `mlruns/` | MLflow tracking data |

## Metrics Logged

- `iterations` – steps to convergence (or max_iter)
- `final_loss` – objective value at termination
- `grad_norm` – final gradient magnitude
- `wall_time` – execution time in seconds
- `converged` – whether tolerance was reached

## View MLflow Results

```bash
cd /workspaces/Computational-ML-lab
mlflow ui --backend-store-uri mlruns
# Open http://127.0.0.1:5000 in browser
```

## Expected Results

| Problem | Optimizer | Typical Iters | Notes |
|---------|-----------|---------------|-------|
| Quadratic κ=5 | GD | ~50 | Fast convergence |
| Quadratic κ=5 | Momentum | ~30 | Momentum helps |
| Quadratic κ=5 | Adam | ~40 | Adaptive LR |
| Quadratic κ=100 | GD | 300 (max) | Struggles |
| Quadratic κ=100 | Momentum | ~150 | Much better |
| Quadratic κ=100 | Adam | ~80 | Best for ill-cond |
| Logistic | GD | ~100 | Standard |
| Logistic | Momentum | ~60 | Faster |
| Logistic | Adam | ~50 | Fastest |

## Reproducibility

- Seed controls: NumPy RNG, problem generation, initial points
- Default seed: 42
- All results are deterministic given the same seed
