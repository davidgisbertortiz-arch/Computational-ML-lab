# Data Directory

This directory is for storing datasets used in experiments.

## Guidelines

1. **Git Ignore**: By default, this directory is gitignored to prevent large files from being committed
2. **Small Data**: If you have small (<1MB) sample datasets, they can be committed with git
3. **Large Data**: For large datasets:
   - Store externally (e.g., cloud storage, public repositories)
   - Provide download scripts in module's `src/` directory
   - Document download instructions in module README

## Structure

Organize data by module:

```
data/
├── 01_numerical_toolbox/
│   └── sample_matrices.npz
├── 03_ml_tabular_foundations/
│   ├── download.py
│   └── preprocessed/
├── 04_time_series_state_space/
│   └── synthetic_timeseries.csv
└── README.md (this file)
```

## Example Download Script

```python
# modules/XX_module/src/download_data.py
import urllib.request
from pathlib import Path

def download_dataset(output_dir: Path):
    """Download dataset from public source."""
    url = "https://example.com/dataset.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.csv"
    
    if not output_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")
    else:
        print(f"Dataset already exists at {output_path}")

if __name__ == "__main__":
    download_dataset(Path("data/XX_module"))
```

## Best Practices

- **Reproducibility**: Document exact data sources and versions
- **Preprocessing**: Save preprocessing steps as code, not just preprocessed data
- **Validation**: Include data checksums (MD5/SHA256) to verify integrity
- **Synthetic Data**: When possible, generate synthetic data in code for full reproducibility
