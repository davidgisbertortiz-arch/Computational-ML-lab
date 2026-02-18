# MLflow Tracking Guide

## Overview

MLflow is an **optional** experiment tracking tool used across modules in this repository. It provides a convenient UI for comparing experiment runs, hyperparameters, and metrics.

## Key Points

- **Optional**: Experiments run normally without MLflow enabled
- **Local-only**: Uses SQLite backend (`mlflow.db`) - no server setup required
- **Not committed**: All MLflow artifacts are gitignored and generated at runtime
- **Clean up**: Use `make clean` to remove local tracking artifacts

## Local Artifacts (gitignored)

The following MLflow-related files are generated at runtime and automatically ignored by git:

- `mlflow.db` - SQLite database storing experiment metadata
- `*.db-shm`, `*.db-wal` - SQLite write-ahead log files
- `mlruns/` - Directory containing run artifacts and metrics
- `mlartifacts/` - Directory for model artifacts
- `.mlflow/` - MLflow cache directory

## Enabling Tracking

### Per-Experiment Configuration

Enable MLflow tracking in your experiment YAML config:

```yaml
name: "my-experiment"
seed: 42
mlflow_tracking: true  # Enable MLflow logging
mlflow_experiment_name: "numerical-optimization"  # Optional: custom experiment name
```

### Programmatic Usage

```python
from modules._import_helper import safe_import_from

BaseExperiment = safe_import_from('00_repo_standards.src.mlphys_core', 'BaseExperiment')

class MyExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config, module_dir=Path(__file__).parent.parent)
        # MLflow logging handled automatically if config.mlflow_tracking = True
    
    def evaluate(self, model, data):
        metrics = {"accuracy": 0.95, "loss": 0.15}
        # Metrics logged to MLflow automatically by BaseExperiment
        return metrics
```

## Viewing Results

### Start MLflow UI

```bash
# From repository root
make mlflow-ui

# Or directly
mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000 in your browser.

### UI Features

- **Experiment comparison**: Compare metrics across runs
- **Parameter tracking**: View hyperparameters for each run
- **Artifact browser**: Download model checkpoints and plots
- **Run metadata**: Git commit hash, timestamps, config files

## Cleaning Up

### Remove All MLflow Artifacts

```bash
make clean
```

This removes:
- Local database files (`mlflow.db`, `*.db-shm`, `*.db-wal`)
- Run directories (`mlruns/`, `mlartifacts/`)
- Cache directories (`.mlflow/`)

### Manual Cleanup

```bash
rm -rf mlruns/ mlartifacts/ mlflow.db *.db-shm *.db-wal .mlflow/
```

## CI/CD Considerations

- CI jobs do not require MLflow artifacts
- Tests run with `mlflow_tracking: false` by default
- No MLflow server setup needed for local development
- Artifacts are never committed to version control

## Best Practices

1. **Development**: Enable tracking for experiments you want to compare
2. **CI**: Keep tracking disabled to avoid artifact accumulation
3. **Cleanup**: Run `make clean` periodically to reclaim disk space
4. **Sharing**: Export specific runs if you need to share results (MLflow supports export/import)

## Troubleshooting

### "Database is locked" error

This usually means multiple processes are accessing the database. Close any running MLflow UI instances or experiments.

### Missing experiments after `git pull`

This is expected - MLflow data is local and not synced. Each developer has their own local tracking database.

### Large `mlruns/` directory

Use `make clean` to remove old experiment artifacts. For selective cleanup, manually delete specific experiment directories in `mlruns/`.

## Alternative Tracking

If you prefer not to use MLflow, experiments automatically:
- Save config files to `reports/config.yaml`
- Log metrics to `reports/metrics.json`
- Save plots to `reports/figures/`

These files are also gitignored but provide a lightweight alternative for tracking individual runs.
