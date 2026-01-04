# Module 02: Statistical Inference & Uncertainty Quantification

**Status**: ✅ Complete

Treats uncertainty as a first-class object throughout statistical inference and machine learning. Implements Bayesian methods, calibration techniques, and MCMC sampling from scratch.

---

## � Educational Notebooks

**Series**: Comprehensive introduction to statistical inference and uncertainty quantification for ML practitioners.

1. **[Probability & Uncertainty Basics](notebooks/01_probability_and_uncertainty_basics.ipynb)**  
   Aleatoric vs epistemic uncertainty, confidence intervals, measurement error, sampling distributions

2. **[Bayesian Linear Regression & UQ](notebooks/02_bayesian_linear_regression_uq.ipynb)**  
   Posterior inference, predictive distributions, parameter uncertainty, prior effects, Bayesian vs frequentist comparison

3. **[Calibration & Temperature Scaling](notebooks/03_calibration_reliability_temperature_scaling.ipynb)**  
   Reliability diagrams, ECE/Brier metrics, temperature scaling, threshold decision analysis, calibration vs accuracy

4. **[MCMC: Metropolis-Hastings & Diagnostics](notebooks/04_mcmc_metropolis_hastings_diagnostics.ipynb)**  
   Markov Chain Monte Carlo, proposal tuning, acceptance rates, trace plots, autocorrelation, ESS, failure modes

**To run the notebooks**:
```bash
cd modules/02_stat_inference_uq/notebooks
jupyter notebook  # or use VS Code's notebook interface
```

Each notebook is self-contained (~2-3 min runtime) with reproducibility seed=42. Outputs saved to `reports/`.

---

## �📚 What You'll Learn

- **Bayesian Inference**: Posterior distributions, conjugate priors, parameter uncertainty
- **Calibration**: Reliability diagrams, ECE/MCE, temperature scaling
- **MCMC Sampling**: Metropolis-Hastings algorithm, convergence diagnostics
- **Uncertainty Propagation**: How uncertainty flows from data → parameters → predictions
- **Bayesian vs Frequentist**: Credible intervals vs confidence intervals

---

## 🎯 Core Implementations

### 1. Bayesian Linear Regression (`bayesian_regression.py`)
- **Conjugate Gaussian prior** for closed-form posterior
- **Posterior predictive distribution** with mean and variance
- **Parameter sampling** from posterior
- **Log marginal likelihood** for model comparison

**Key insight**: Bayesian approach naturally provides uncertainty estimates for predictions, accounting for both parameter uncertainty and observation noise.

### 2. Calibration Metrics (`calibration.py`)
- **Reliability diagrams** binning predictions by confidence
- **Expected Calibration Error (ECE)** quantifying miscalibration
- **Temperature scaling** for post-hoc calibration improvement

**Key insight**: Modern neural networks are often miscalibrated (overconfident). Temperature scaling provides a simple, effective fix without retraining.

### 3. MCMC Sampling (`mcmc_basics.py`)
- **Metropolis-Hastings** random-walk sampler
- **Diagnostics**: acceptance rate, trace plots, autocorrelation, ESS
- **Visual tools** for assessing convergence

**Key insight**: MCMC enables Bayesian inference for models without closed-form posteriors. Diagnostics are essential to verify sampling quality.

---

## 🧪 Experiments

### Demo 1: Bayesian vs Frequentist Uncertainty
```bash
python -m modules.02_stat_inference_uq.src.main run-bayes-demo --seed 42
```

Compares:
- **Bayesian credible intervals** from posterior predictive
- **Frequentist confidence intervals** from bootstrap

Outputs: `reports/bayes_vs_frequentist.png`

### Demo 2: Calibration with Temperature Scaling
```bash
python -m modules.02_stat_inference_uq.src.main run-calibration-demo --seed 42
```

Shows:
- Reliability diagram before/after temperature scaling
- ECE reduction
- Learned temperature parameter

Outputs: `reports/calibration_before_after.png`

### Demo 3: MCMC Diagnostics
```bash
python -m modules.02_stat_inference_uq.src.main run-mcmc-demo --n-samples 10000
```

Visualizes:
- Trace plots showing chain mixing
- Marginal histograms vs true distribution
- Autocorrelation decay

Outputs: `reports/mcmc_trace.png`, `reports/mcmc_marginals.png`

---

## 📖 Theory (Minimal)

### Bayesian Linear Regression
Model: $y = X w + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

With Gaussian prior $p(w) = \mathcal{N}(w_0, \Lambda_0^{-1})$, posterior is:

$$p(w | X, y) = \mathcal{N}(w_N, \Lambda_N^{-1})$$

where:
- $\Lambda_N = \Lambda_0 + \frac{1}{\sigma^2} X^T X$ (posterior precision)
- $w_N = \Lambda_N^{-1} (\Lambda_0 w_0 + \frac{1}{\sigma^2} X^T y)$ (posterior mean)

Predictive distribution for new $x_*$:

$$p(y_* | x_*, X, y) = \mathcal{N}(x_*^T w_N, \sigma^2 + x_*^T \Lambda_N^{-1} x_*)$$

### Expected Calibration Error

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} | \text{acc}(B_m) - \text{conf}(B_m) |$$

where $B_m$ are bins of predictions grouped by confidence.

### Metropolis-Hastings

1. Propose: $x' \sim q(x' | x_t)$ (e.g., $x' = x_t + \mathcal{N}(0, \sigma^2)$)
2. Accept with probability: $\alpha = \min(1, \frac{\pi(x')q(x_t|x')}{\pi(x_t)q(x'|x_t)})$
3. If accepted: $x_{t+1} = x'$, else $x_{t+1} = x_t$

For symmetric proposals (random walk), simplifies to $\alpha = \min(1, \frac{\pi(x')}{\pi(x_t)})$.

---

## ⚠️ Failure Modes & Pitfalls

1. **Small Sample Size**
   - Problem: Bayesian posterior overconfident with few data points
   - Fix: Use informative priors or report posterior uncertainty honestly

2. **Miscalibration Detection**
   - Problem: Standard metrics (accuracy, F1) don't catch miscalibration
   - Fix: Always compute ECE for probabilistic models; visualize reliability diagrams

3. **MCMC Non-Convergence**
   - Problem: Chain stuck in local mode or mixing poorly
   - Fix: Check diagnostics (acceptance rate 0.2-0.5, ACF decay, ESS > 100)
   - Try: Adjust proposal std, run longer, use better sampler (HMC/NUTS)

4. **Temperature Scaling Overfitting**
   - Problem: Fitting temperature on test set inflates calibration metrics
   - Fix: Always use separate validation set for temperature fitting

5. **Assuming Gaussian Posterior**
   - Problem: Not all posteriors are Gaussian (e.g., multimodal)
   - Fix: Use MCMC or variational inference for complex posteriors

---

## ✅ Definition of Done

- [x] Bayesian regression with posterior predictive
- [x] Calibration metrics (ECE, reliability diagrams)
- [x] Temperature scaling implementation
- [x] Metropolis-Hastings MCMC sampler
- [x] MCMC diagnostics (ACF, ESS, trace plots)
- [x] Comprehensive tests (>80% coverage)
- [x] 3 CLI demos with visualizations
- [x] Notebooks comparing Bayesian/frequentist approaches
- [x] README with theory and usage

---

## 🔗 Resources

**Papers**:
- Guo et al. (2017): "On Calibration of Modern Neural Networks" - Temperature scaling
- Neal (1993): "Probabilistic Inference Using Markov Chain Monte Carlo Methods" - MCMC fundamentals
- Bishop (2006): "Pattern Recognition and Machine Learning", Ch. 3 - Bayesian regression

**Tutorials**:
- [MCMC Sampling for Dummies](http://twiecki.github.io/blog/2015/11/10/mcmc-sampling-for-dummies/)
- [Calibration in Deep Learning](https://calibration-in-deep-learning.readthedocs.io/)

---

## 🚀 Next Steps

After completing this module:
1. Review outputs in `reports/`
2. Run tests: `make test-module MODULE=02_stat_inference_uq`
3. Explore notebooks for interactive demos
4. Move to **Module 03: Tabular ML Foundations** (gradient boosting, trees, ensembles)

---

## 📁 File Structure

```
modules/02_stat_inference_uq/
├── src/
│   ├── __init__.py
│   ├── bayesian_regression.py     # Bayesian linear regression (362 lines)
│   ├── calibration.py              # ECE, reliability, temp scaling (334 lines)
│   ├── mcmc_basics.py              # Metropolis-Hastings + diagnostics (332 lines)
│   └── main.py                     # CLI entry points (245 lines)
├── tests/
│   ├── test_bayesian_regression.py  # 180 lines, 13 tests
│   ├── test_calibration.py          # 210 lines, 17 tests
│   └── test_mcmc_basics.py          # 220 lines, 15 tests
├── notebooks/
│   └── 01_bayesian_vs_frequentist.ipynb  # Interactive demos
├── configs/
│   └── (future: experiment configs)
└── reports/                         # Generated plots (gitignored)
```

**Total**: ~1900 lines of production code + tests
