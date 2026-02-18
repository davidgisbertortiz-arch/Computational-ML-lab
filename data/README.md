# Data policy

This repository does **not** track runtime datasets or large binary artifacts in git.

## What belongs in `data/`

- Lightweight metadata and docs (`README`, schema notes, source manifests)
- Fetch/generate scripts (`*.py`) that reproduce datasets locally
- Optional placeholders (`.gitkeep`) to preserve folder structure

## What must not be committed

- Downloaded raw datasets
- Preprocessed data dumps and feature stores
- Large binaries (`.csv`, `.parquet`, `.npz`, `.npy`, `.pt`, `.pth`, etc.)

The root `.gitignore` is configured to ignore `data/**` by default, while keeping this README
and scripts/documentation trackable.

## Recommended workflow

1. Add a module-local script to fetch data from the original source.
2. Save local outputs under `data/<module_name>/`.
3. Document source URL, expected version/date, and checksum in the module README.
4. For synthetic experiments, generate data from code with a fixed seed.

## Example fetch script

```python
from pathlib import Path
import urllib.request


def fetch_data() -> Path:
    output_dir = Path("data/03_ml_tabular_foundations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.csv"

    if not output_path.exists():
        urllib.request.urlretrieve("https://example.com/dataset.csv", output_path)

    return output_path
```

## Reproducibility checklist

- Keep data acquisition fully scripted
- Record dataset version/checksum
- Keep preprocessing in code (not manual steps)
- Use deterministic seeds (`set_seed(...)`) for generated data
