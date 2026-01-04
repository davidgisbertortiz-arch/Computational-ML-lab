# Model Card: Particle Collision Signal Classifier

**Model Version**: 1.0.0  
**Date**: January 2026  
**Author**: David Gisbert Ortiz  
**Framework**: scikit-learn + LightGBM  

---

## Model Overview

### Model Description

A gradient boosting classifier trained to identify rare signal events (e.g., Higgs boson decays) from background noise in simulated high-energy particle collision data. The model is designed to maximize signal detection while maintaining high precision to minimize false positives in downstream physics analysis.

| Property | Value |
|----------|-------|
| **Architecture** | LightGBM Gradient Boosting (500 trees) |
| **Input** | 16 kinematic features (continuous) |
| **Output** | Probability of signal event [0, 1] |
| **Task** | Binary classification |
| **Calibration** | Platt scaling (sigmoid) |

### Intended Use

**Primary Use Cases**:
- Signal/background discrimination in particle physics experiments
- Event filtering for physics analysis pipelines
- Demonstration of production ML patterns for tabular classification

**Intended Users**:
- Particle physicists analyzing collision data
- ML engineers building classification pipelines
- Students learning tabular ML best practices

**Out-of-Scope Uses**:
- ❌ Real detector data without retraining (simulation ≠ reality)
- ❌ Different collision energies or detector geometries
- ❌ Real-time trigger systems (latency not optimized)
- ❌ Safety-critical decisions without human oversight

---

## Training Data

### Dataset Description

| Property | Value |
|----------|-------|
| **Source** | Synthetic (physics-inspired generation) |
| **Total Samples** | 100,000 |
| **Signal Events** | 10,000 (10%) |
| **Background Events** | 90,000 (90%) |
| **Features** | 16 kinematic variables |
| **Missing Values** | None |
| **Time Period** | N/A (simulation) |

### Feature Description

| Feature | Description | Units | Physics Interpretation |
|---------|-------------|-------|------------------------|
| `p_T` | Transverse momentum | GeV/c | Particle momentum perpendicular to beam |
| `eta` | Pseudorapidity | - | Angular position (forward/central) |
| `phi` | Azimuthal angle | rad | Angular position around beam |
| `E_total` | Total energy | GeV | Particle energy |
| `m_inv` | Invariant mass | GeV/c² | Reconstructed parent particle mass |
| `missing_E_T` | Missing transverse energy | GeV | Undetected particles (neutrinos) |
| `n_jets` | Number of jets | count | Hadronic activity |
| `b_tag_score` | B-jet tagging probability | [0,1] | Heavy flavor content |
| `lepton_iso` | Lepton isolation | [0,1] | Lepton quality |
| `delta_R` | Angular separation | - | Derived from η, φ |
| `m_T` | Transverse mass | GeV/c² | Derived from p_T, missing_E_T |
| `E_ratio` | Energy ratio | - | missing_E_T / E_total |
| `sphericity` | Event sphericity | [0,1] | Event shape (isotropic vs jetty) |
| `aplanarity` | Event aplanarity | [0,1] | 3D event shape |
| `centrality` | Energy centrality | [0,1] | Energy distribution |
| `H_T` | Scalar sum of jet p_T | GeV | Total hadronic activity |

### Data Splits

| Split | Samples | Signal Rate | Purpose |
|-------|---------|-------------|---------|
| Train | 60,000 | 10.0% | Model fitting |
| Validation | 20,000 | 10.0% | Hyperparameter tuning, calibration |
| Test | 20,000 | 10.0% | Final evaluation (single use) |

**Stratification**: All splits maintain class balance via stratified sampling.

---

## Model Performance

### Primary Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **AUC-ROC** | 0.962 | Overall discrimination ability |
| **Average Precision** | 0.884 | Precision-recall summary |
| **Recall @ P95** | 0.823 | Signal efficiency at 95% precision |
| **Accuracy** | 0.941 | Overall correctness |
| **F1 Score** | 0.762 | Harmonic mean of precision/recall |

### Calibration Metrics

| Metric | Before Calibration | After Calibration |
|--------|-------------------|-------------------|
| **ECE** | 0.124 | 0.038 |
| **Brier Score** | 0.118 | 0.098 |

**Interpretation**: Platt scaling reduces Expected Calibration Error by 69%, making probability outputs trustworthy for physics analysis.

### Comparison with Baseline

| Model | AUC-ROC | AP | Recall@P95 | ECE |
|-------|---------|----|-----------|----|
| Logistic Regression | 0.921 | 0.752 | 0.651 | 0.082 |
| LightGBM (uncalibrated) | 0.962 | 0.884 | 0.823 | 0.124 |
| **LightGBM (calibrated)** | **0.962** | **0.884** | **0.823** | **0.038** |

---

## Uncertainty Quantification

### Probability Calibration

The model outputs **calibrated probabilities** via Platt scaling:

- **Reliability**: Predicted probabilities match observed frequencies (ECE = 0.038)
- **Confidence interpretation**: A prediction of 0.8 means ~80% of similar events are true signals
- **Decision thresholds**: Optimal threshold = 0.72 for Recall@P95 objective

### Known Uncertainty Sources

| Source | Type | Mitigation |
|--------|------|------------|
| Simulation vs reality gap | Epistemic | Retrain on real data |
| Feature measurement noise | Aleatoric | Model captures via probabilistic output |
| Class imbalance | Epistemic | Stratified sampling, threshold optimization |
| Hyperparameter sensitivity | Epistemic | Cross-validation, regularization |

### When to Trust Predictions

✅ **High confidence** (p > 0.9 or p < 0.1): Model is typically correct  
⚠️ **Medium confidence** (0.3 < p < 0.7): Prediction uncertain, consider additional features  
❌ **Out-of-distribution**: Novel physics not seen in training will have unreliable predictions

---

## Limitations

### Technical Limitations

1. **Synthetic data**: Model trained on simulation, may not generalize to real detector data
2. **Feature engineering frozen**: Requires same 16 features in same format
3. **Class distribution**: Trained on 10% signal; performance may degrade at different rates
4. **No temporal effects**: Assumes stationary detector conditions

### Failure Modes

| Scenario | Expected Behavior | Recommendation |
|----------|-------------------|----------------|
| Missing features | Error or NaN output | Imputation pipeline |
| Extreme values | Extrapolation (unreliable) | Clip to training range |
| New physics signatures | May classify as background | Domain adaptation |
| Detector anomalies | Unpredictable | Monitor input distributions |

### What the Model Cannot Do

- ❌ Discover new physics (trained on known signatures)
- ❌ Explain why an event is signal (interpretable features help, but no causal claims)
- ❌ Quantify systematic uncertainties (requires ensemble or Bayesian methods)

---

## Ethical Considerations

### Fairness & Bias

**N/A for physics application**: No demographic groups or protected attributes in particle collision data. The model discriminates only on physical properties.

### Potential Misuse

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Claiming real physics discovery from synthetic model | Medium | Clear documentation of synthetic data |
| Deploying without recalibration on real data | Medium | Versioning and validation requirements |
| Over-reliance on probability outputs | Low | Uncertainty documentation |

### Environmental Impact

| Resource | Estimate |
|----------|----------|
| Training time | ~30 seconds (CPU) |
| Inference time | ~0.1ms per sample |
| Carbon footprint | Negligible |

---

## Reproducibility

### Requirements

```
Python >= 3.11
scikit-learn >= 1.3.0
lightgbm >= 4.0.0
numpy >= 1.26.0
pandas >= 2.0.0
```

### Reproduction Steps

```bash
# 1. Generate data (deterministic with seed=42)
python -m modules.03_ml_tabular_foundations.src.data

# 2. Train model
python -m modules.03_ml_tabular_foundations.src.train train \
  --config modules/03_ml_tabular_foundations/configs/lightgbm.yaml

# 3. Evaluate
python -m modules.03_ml_tabular_foundations.src.train evaluate \
  --model-path reports/lightgbm_production/models/model.pkl
```

### Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Trained model | `reports/lightgbm_production/models/model.pkl` | Serialized pipeline |
| Config | `configs/lightgbm.yaml` | Hyperparameters |
| Metrics | `reports/lightgbm_production/metadata.json` | Performance metrics |
| Feature importance | `reports/lightgbm_production/feature_importance.csv` | Permutation importance |

---

## Maintenance

### Model Updates

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Jan 2026 | Initial release |

### Monitoring Recommendations

1. **Input drift**: Compare feature distributions to training data
2. **Performance drift**: Track AUC-ROC on periodic validation samples
3. **Calibration drift**: Recompute ECE monthly

### Contact

**Author**: David Gisbert Ortiz  
**Repository**: [Computational-ML-lab](https://github.com/davidgisbertortiz-arch/Computational-ML-lab)  
**Module**: `modules/03_ml_tabular_foundations/`

---

## Citation

```bibtex
@misc{gisbert2026particle,
  author = {Gisbert Ortiz, David},
  title = {Particle Collision Signal Classifier: A Portfolio-Grade Tabular ML Benchmark},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/davidgisbertortiz-arch/Computational-ML-lab}
}
```

---

## References

1. Mitchell et al. (2019). "Model Cards for Model Reporting." *FAT* Conference.
2. Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
3. Platt (1999). "Probabilistic Outputs for Support Vector Machines."
4. Guo et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.
