# Module 03: ML Tabular Foundations

**Status**: ✅ Complete

Portfolio-grade tabular ML benchmark demonstrating production-quality patterns: proper sklearn pipelines, train/val/test splits, cross-validation, calibration, threshold selection, and explainability.

---

## 📋 Documentation

| Document | Description |
|----------|-------------|
| **[Notebook Quick-Start](NOTEBOOK_QUICKSTART.md)** | How to run notebooks, troubleshooting, learning tips |
| **[Model Card](MODEL_CARD.md)** | Intended use, limitations, ethical concerns, uncertainty quantification |
| **[Experiment Report](EXPERIMENT_REPORT.md)** | Final metrics, calibration analysis, error patterns, ablation study |

---

## 📓 Educational Notebooks

**Learning Path**: 5 notebooks teaching tabular ML with professional rigor

| Notebook | Topic | Learning Objectives | Runtime |
|----------|-------|---------------------|---------|
| **[01_eda_and_data_quality.ipynb](notebooks/01_eda_and_data_quality.ipynb)** | EDA & Data Quality | Systematic EDA, schema validation, leakage detection, outlier analysis | ~2 min |
| **[02_splitting_cv_and_leakage.ipynb](notebooks/02_splitting_cv_and_leakage.ipynb)** | Splits, CV & Leakage | Proper train/val/test splits, stratification, demonstrate leakage with wrong pipelines, fix with sklearn Pipelines | ~2 min |
| **[03_baselines_and_metrics_that_matter.ipynb](notebooks/03_baselines_and_metrics_that_matter.ipynb)** | Baselines & Metrics | Establish baselines, choose metrics aligned with business goals, threshold selection, bootstrap confidence intervals | ~2 min |
| **[04_gbm_model_and_hyperparameter_search.ipynb](notebooks/04_gbm_model_and_hyperparameter_search.ipynb)** | GBM & Hyperparameter Tuning | Train LightGBM, hyperparameter search with RandomizedSearchCV, learning curves, feature importance | ~3 min |
| **[05_calibration_explainability_and_error_analysis.ipynb](notebooks/05_calibration_explainability_and_error_analysis.ipynb)** | Calibration & Error Analysis | Calibrate predictions (Platt scaling, ECE), permutation importance, slice analysis, identify failure modes | ~2 min |

**Teaching Style**: Professor + lab manual with theory, implementation, experiments, failure modes, and exercises with solutions.

**Prerequisites**: Basic Python, NumPy, Pandas, scikit-learn

**To run notebooks**:
```bash
cd modules/03_ml_tabular_foundations
jupyter notebook notebooks/
# Or use VS Code's Jupyter extension
```

---

## 🎯 Problem Framing

**Task**: Binary classification of synthetic particle collision events

**Use case**: Detect signal events (e.g., rare particle decays) from background noise in high-energy physics experiments.

**Why this problem?**
- Representative of real-world scientific ML (CERN, particle physics)
- Class imbalance (signal:background ≈ 1:9)
- Physics-inspired features with correlations and nonlinearities
- Interpretability matters (physicists need to understand decisions)

**Business metrics**:
- **Recall @ 95% precision**: Maximize signal detection while controlling false positives
- **AUC-ROC**: Overall discrimination ability
- **Calibration (ECE)**: Trust probabilistic predictions for decision-making

---

## 📊 Dataset

**Synthetic Particle Collision Dataset** (`data/particle_collisions.csv`)

**Generation**: Physics-inspired feature correlations
- **16 features**: kinematic variables (momentum, energy, angles), derived quantities
  - `p_T`: Transverse momentum (GeV/c)
  - `eta`, `phi`: Pseudorapidity and azimuthal angle
  - `E_total`: Total energy (GeV)
  - `m_inv`: Invariant mass (GeV/c²)
  - `missing_E_T`: Missing transverse energy (GeV)
  - Plus 10 additional derived features with physics-motivated correlations

**Target**: `is_signal` (0 = background, 1 = signal event)

**Statistics**:
- Total samples: 100,000
- Class distribution: 10% signal, 90% background
- Train/val/test split: 60% / 20% / 20%
- No missing values (particles always measured)

**Why synthetic?**
- Self-contained (no external downloads)
- Reproducible data generation with seed
- Controlled feature correlations for teaching
- No privacy/licensing concerns

---

## 📏 Metrics & Evaluation

### Primary Metrics

1. **AUC-ROC**: Overall discrimination (threshold-agnostic)
2. **Average Precision (AP)**: Summary of precision-recall curve (better for imbalanced classes)
3. **Recall @ Precision=0.95**: Operational metric (maximize signal detection with <5% false positives)

### Calibration Metrics

4. **Expected Calibration Error (ECE)**: Measures probability calibration quality
5. **Brier Score**: Mean squared error of probabilistic predictions

### Model Interpretability

6. **Permutation Importance**: Feature contribution to model performance
7. **SHAP values** (optional): Instance-level feature attribution

**Leaderboard priority**: AUC-ROC (tiebreaker: Recall@P95)

---

## ⚠️ Leakage Pitfalls

**Common mistakes we avoid**:

1. ❌ **Preprocessing leakage**: Fitting scaler on full dataset before split
   - ✅ **Fix**: Fit scaler only on train set, transform val/test
   - ✅ **Implementation**: Use sklearn Pipeline

2. ❌ **Validation leakage**: Tuning on test set
   - ✅ **Fix**: Three-way split (train/val/test), tune on val, evaluate on test
   - ✅ **Implementation**: Stratified splits with fixed seed

3. ❌ **Temporal leakage**: Not applicable (no time ordering in particle events)

4. ❌ **Target leakage**: Features computed using target
   - ✅ **Fix**: Synthetic generation ensures features computed from physics only

5. ❌ **Cross-validation leakage**: Different preprocessing per fold
   - ✅ **Fix**: Pipeline ensures consistent preprocessing in CV

**Tests**: `tests/test_no_leakage.py` verifies:
- Preprocessor fit only on train data
- No data overlap between splits
- Reproducible results with seed

---

## 🧪 Experiment Design

### Baseline Model

**Logistic Regression** with L2 regularization
- Simple, interpretable, fast
- Strong baseline for tabular data
- Provides calibrated probabilities by default

**Pipeline**:
```python
StandardScaler → LogisticRegression(C=1.0, max_iter=1000)
```

### Production Model

**LightGBM Classifier**
- State-of-art for tabular data
- Handles feature interactions and nonlinearities
- Fast training with early stopping
- Built-in feature importance

**Pipeline**:
```python
StandardScaler → LightGBM(n_estimators=500, learning_rate=0.05, max_depth=7)
```

**Hyperparameters** (tuned via 5-fold CV on validation set):
- `learning_rate`: [0.01, 0.05, 0.1]
- `num_leaves`: [31, 63, 127]
- `max_depth`: [5, 7, 10]
- `min_child_samples`: [20, 50, 100]

### Post-processing

1. **Calibration**: Platt scaling (logistic calibration on validation set)
2. **Threshold selection**: Optimize for Recall@P95 on validation set

### Cross-Validation Strategy

- **5-fold stratified CV** on train set for model selection
- **Validation set** for threshold tuning and calibration
- **Test set** for final evaluation (report once)

---

## 🏗️ Implementation

### Core Modules

#### `src/data.py`
- `generate_particle_collision_data()`: Create synthetic dataset
- `load_data()`: Load from disk with validation
- `split_data()`: Stratified train/val/test splits

#### `src/models.py`
- `LogisticRegressionPipeline`: Baseline sklearn pipeline
- `LightGBMPipeline`: Production GBM pipeline
- `CalibratedModel`: Wrapper for Platt scaling

#### `src/eval.py`
- `compute_metrics()`: All evaluation metrics
- `plot_roc_curve()`, `plot_precision_recall()`: Visualizations
- `plot_calibration_curve()`: Reliability diagram
- `compute_permutation_importance()`: Feature importance

#### `src/train.py`
- Typer CLI with `train` and `evaluate` commands
- Config-driven experiments (YAML configs)
- Saves models, metrics, and plots to `reports/`

### Configuration

**Example**: `configs/baseline.yaml`
```yaml
name: logistic_baseline
seed: 42
model:
  type: logistic
  C: 1.0
  max_iter: 1000
data:
  n_samples: 100000
  signal_fraction: 0.1
training:
  cv_folds: 5
  calibrate: true
  optimize_threshold: true
```

---

## 🚀 Usage

### 1. Generate Data
```bash
python -m modules.03_ml_tabular_foundations.src.data
```
Creates `data/particle_collisions.csv`

### 2. Train Baseline
```bash
python -m modules.03_ml_tabular_foundations.src.train train \
  --config configs/baseline.yaml
```

### 3. Train Production Model
```bash
python -m modules.03_ml_tabular_foundations.src.train train \
  --config configs/lightgbm.yaml
```

### 4. Evaluate on Test Set
```bash
python -m modules.03_ml_tabular_foundations.src.train evaluate \
  --model-path reports/models/lightgbm_model.pkl \
  --output reports/test_metrics.json
```

### 5. Run Notebooks
```bash
jupyter notebook notebooks/
```
- `01_eda.ipynb`: Exploratory data analysis
- `02_error_analysis.ipynb`: Error patterns and feature importance

---

## 📊 Expected Results

### Performance

| Model | AUC-ROC | AP | Recall@P95 | ECE | Brier |
|-------|---------|----|-----------|----|-------|
| Logistic (baseline) | 0.92 | 0.75 | 0.65 | 0.08 | 0.15 |
| LightGBM (uncalibrated) | 0.96 | 0.88 | 0.82 | 0.12 | 0.12 |
| LightGBM (calibrated) | 0.96 | 0.88 | 0.82 | 0.04 | 0.10 |

**Insights**:
- GBM significantly outperforms logistic (AUC: 0.96 vs 0.92)
- Calibration reduces ECE by ~3x without hurting discrimination
- Threshold tuning boosts recall@P95 from default 0.5 threshold

### Feature Importance (Top 5)

1. `m_inv` (invariant mass): Strongest signal discriminator
2. `missing_E_T`: Indicates undetected particles
3. `p_T` (transverse momentum): Kinematic signature
4. `E_total`: Energy conservation constraint
5. `eta` (pseudorapidity): Angular distribution

---

## 🧪 Tests

Run tests to verify correctness:

```bash
# All tests
pytest modules/03_ml_tabular_foundations/tests/

# Specific tests
pytest modules/03_ml_tabular_foundations/tests/test_no_leakage.py -v
pytest modules/03_ml_tabular_foundations/tests/test_reproducibility.py -v
```

**Test coverage**:
- ✅ No preprocessing leakage (scaler fit only on train)
- ✅ Deterministic training (same seed → same results)
- ✅ Pipeline correctness (end-to-end predictions)
- ✅ Calibration improvement (ECE reduction)
- ✅ Data split integrity (no overlap, correct sizes)

---

## 📚 Key Takeaways

✅ **Pipelines prevent leakage**: Always use sklearn Pipeline for preprocessing + model

✅ **Three-way split essential**: Train for fitting, validation for tuning/calibration, test for final eval

✅ **Calibration matters**: GBMs often overconfident, Platt scaling fixes this

✅ **Threshold selection**: Default 0.5 is rarely optimal for imbalanced classes

✅ **Explainability builds trust**: Permutation importance + SHAP reveal what model learned

✅ **Reproducibility is non-negotiable**: Fixed seeds, versioned configs, test determinism

---

## 🔗 References

- **sklearn Pipelines**: https://scikit-learn.org/stable/modules/compose.html
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Calibration**: Platt (1999), "Probabilistic Outputs for SVMs"
- **SHAP**: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model Predictions"
- **Threshold selection**: Chawla et al. (2004), "Threshold-moving approaches for imbalanced data"

---

## 📁 Project Structure

```
modules/03_ml_tabular_foundations/
├── src/
│   ├── __init__.py
│   ├── data.py           # Data generation and loading
│   ├── models.py         # Pipeline definitions
│   ├── eval.py           # Metrics and visualization
│   └── train.py          # CLI for training and evaluation
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_no_leakage.py
│   └── test_reproducibility.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_error_analysis.ipynb
├── configs/
│   ├── baseline.yaml
│   └── lightgbm.yaml
├── reports/          # Generated outputs (gitignored)
│   ├── models/
│   ├── metrics/
│   └── plots/
└── README.md
```

---

## 🎓 Learning Objectives

After completing this module, you will:

1. ✅ Build production-quality ML pipelines with sklearn
2. ✅ Implement proper train/val/test splits without leakage
3. ✅ Tune hyperparameters with cross-validation
4. ✅ Calibrate probabilistic predictions (Platt scaling)
5. ✅ Select optimal decision thresholds for imbalanced data
6. ✅ Explain model predictions with permutation importance
7. ✅ Write tests to catch common ML bugs
8. ✅ Structure experiments with configs and reproducible seeds

**Next module**: Time series & state-space models (Module 04)
