# 📓 Notebook Quick-Start Guide

## Prerequisites

1. **Environment setup**:
   ```bash
   cd /workspaces/Computational-ML-lab
   pip install -e ".[dev]"  # Install all dependencies
   ```

2. **Generate dataset** (if not exists):
   ```bash
   python -m modules.03_ml_tabular_foundations.src.data
   # Creates: data/particle_collisions.csv (100K samples)
   ```

## Running Notebooks

### Option 1: VS Code (Recommended)

1. Open VS Code in workspace root
2. Navigate to `modules/03_ml_tabular_foundations/notebooks/`
3. Click any `.ipynb` file
4. Select Python kernel (should auto-detect conda/venv)
5. Run cells with `Shift+Enter`

### Option 2: Jupyter Notebook

```bash
cd modules/03_ml_tabular_foundations
jupyter notebook notebooks/
# Opens browser → select notebook
```

### Option 3: JupyterLab

```bash
cd modules/03_ml_tabular_foundations
jupyter lab
# Navigate to notebooks/ folder
```

## Notebook Sequence

Run in order for best learning experience:

1. **`01_eda_and_data_quality.ipynb`** (15-20 min)
   - Start here: Learn systematic EDA
   - Generates: EDA plots + summary JSON

2. **`02_splitting_cv_and_leakage.ipynb`** (15-20 min)
   - Critical: Demonstrates leakage with broken pipelines
   - Generates: Leakage comparison plot + checklist

3. **`03_baselines_and_metrics_that_matter.ipynb`** (20-25 min)
   - Establishes baselines (majority, logistic regression)
   - Generates: ROC/PR curves, confusion matrix, bootstrap CIs

4. **`04_gbm_model_and_hyperparameter_search.ipynb`** (20-25 min)
   - Trains production model (LightGBM)
   - Generates: Learning curves, feature importance, tuning results

5. **`05_calibration_explainability_and_error_analysis.ipynb`** (20-25 min)
   - Calibrates predictions + error analysis
   - Generates: Reliability diagrams, slice analysis, error summary

**Total time**: ~2 hours (hands-on learning)

## Outputs

All notebooks save outputs to `reports/`:

```
reports/
├── 01_target_distribution.png
├── 01_eda_summary.json
├── 02_leaky_vs_correct.png
├── 02_leakage_checklist.md
├── 03_roc_curve_baseline.png
├── 03_baseline_metrics.csv
├── 04_learning_curve.png
├── 04_tuning_comparison.csv
├── 05_reliability_comparison.png
└── 05_error_analysis_summary.md
```

## Troubleshooting

### Import errors (Python 3.12+)

**Error**: `SyntaxError: invalid syntax` on module imports

**Fix**: Use the import helper pattern:
```python
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
```

See [docs/getting-started/python312-imports.md](../../docs/getting-started/python312-imports.md)

### Missing dependencies

**Error**: `ModuleNotFoundError: No module named 'lightgbm'`

**Fix**:
```bash
pip install lightgbm typer pyyaml
# Or reinstall full environment:
pip install -e ".[dev]"
```

### Data not found

**Error**: `FileNotFoundError: data/particle_collisions.csv`

**Fix**:
```bash
python -m modules.03_ml_tabular_foundations.src.data
```

### Kernel not found

**Issue**: VS Code doesn't show Python kernel

**Fix**:
1. Open Command Palette (`Cmd+Shift+P`)
2. Type: "Python: Select Interpreter"
3. Choose your environment (conda/venv)
4. Reopen notebook

## Learning Tips

1. **Read before running**: Each notebook has theory sections—don't skip!

2. **Run cells in order**: Notebooks depend on previous cell outputs

3. **Do exercises**: Each notebook has 3-6 exercises with solutions
   - Try them before looking at solutions
   - Solutions are at the end of each notebook

4. **Experiment**: After finishing, try:
   - Changing hyperparameters
   - Adding new features
   - Testing different thresholds

5. **Portfolio use**: These notebooks demonstrate:
   - Production ML patterns (pipelines, CV, calibration)
   - Error analysis rigor
   - Clear communication (plots, tables, summaries)
   
   → Show to employers/mentors as evidence of ML engineering skill

## Next Steps

After completing all notebooks:

1. **Apply to your data**:
   - Replace `load_data()` with your dataset
   - Adapt feature engineering to your domain
   - Follow same workflow (EDA → splits → baselines → GBM → calibration)

2. **Extend functionality**:
   - Add SHAP for instance-level explanations
   - Try other models (XGBoost, CatBoost)
   - Implement online learning / model updates

3. **Production deployment**:
   - Package as REST API (FastAPI)
   - Add monitoring (Evidently, Arize)
   - Implement A/B testing framework

## Questions?

- Check [README.md](README.md) for module overview
- Review [MODEL_CARD.md](MODEL_CARD.md) for model details
- See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for results

**Happy learning!** 🚀
