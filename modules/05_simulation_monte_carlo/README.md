# Module 05: Simulation & Monte Carlo Methods

**Status:** ✅ Complete

**Learning Objectives:**
- Master Monte Carlo integration with confidence intervals
- Apply variance reduction techniques (importance sampling, control variates, antithetic sampling)
- Estimate rare event probabilities reliably
- Generate physics-inspired synthetic datasets with ground truth
- Understand when variance reduction provides 10x-100x speedups

---

## � Notebooks (Interactive Learning)

Work through these in order for a complete MC methods education:

### [01_monte_carlo_integration_and_error_bars.ipynb](notebooks/01_monte_carlo_integration_and_error_bars.ipynb)
**What you'll learn:**
- Why MC works: LLN + CLT intuition
- Implement MC integration with confidence intervals
- Diagnose convergence: error vs N (O(N^{-1/2}) scaling)
- When MC beats deterministic methods (high dimensions)
- Common pitfalls: confusing variance with SE, poor seeds

**Key experiment:** Estimate 5D unit sphere volume (advantage over grid methods)

---

### [02_variance_reduction_importance_sampling.ipynb](notebooks/02_variance_reduction_importance_sampling.ipynb)
**What you'll learn:**
- Importance sampling derivation and implementation
- See 10-50x speedup on tail probability problems
- Sensitivity to proposal choice (good vs bad)
- Failure modes: weight degeneracy, infinite variance

**Key experiment:** Estimate P(X > 3) for N(0,1) with 20-50x variance reduction

---

### [03_control_variates_and_antithetic_sampling.ipynb](notebooks/03_control_variates_and_antithetic_sampling.ipynb)
**What you'll learn:**
- Control variates with optimal coefficient
- Correlation drives VRF: 1/(1-ρ²)
- Antithetic sampling for symmetric functions
- When to combine multiple variance reduction methods

**Key experiment:** Asian option pricing with geometric mean as control (2-5x VRF)

---

### [04_rare_event_probability_estimation.ipynb](notebooks/04_rare_event_probability_estimation.ipynb)
**What you'll learn:**
- Why rare events (P < 10^-3) break naive MC
- Exponential tilting for tail probabilities
- Adaptive sampling to determine needed N
- Avoiding false confidence from zero hits

**Key experiment:** Estimate P(Z > 4) ≈ 3×10^-5 with 100x variance reduction

---

### [05_synthetic_physics_data_generator.ipynb](notebooks/05_synthetic_physics_data_generator.ipynb)
**What you'll learn:**
- Generate datasets with known ground truth for ML debugging
- Simulate stochastic processes: Brownian, OU, Lévy flights
- Physics-inspired regression problems
- Correlated noise vs IID noise

**Key experiment:** Test ML models on synthetic pendulum energy data, measure true error vs noisy error

---

**How to run:**
```bash
# Launch Jupyter
jupyter notebook modules/05_simulation_monte_carlo/notebooks/

# Or use VS Code Jupyter extension (recommended)
code modules/05_simulation_monte_carlo/notebooks/
```

**Runtime:** Each notebook ~2-3 minutes on CPU. All outputs saved to `reports/`.

---

## �📚 Theory Overview

### Monte Carlo Integration

Estimate $I = \mathbb{E}[f(X)]$ for $X \sim p(x)$:

$$\hat{I} = \frac{1}{N} \sum_{i=1}^N f(X_i), \quad X_i \sim p(x)$$

**Error:** $\text{SE}(\hat{I}) = \frac{\sigma}{\sqrt{N}}$ where $\sigma^2 = \text{Var}[f(X)]$

**Convergence:** $O(N^{-1/2})$ regardless of dimension (curse of dimensionality avoided!)

### Variance Reduction Techniques

#### 1. Importance Sampling
Sample from $q(x)$ instead of $p(x)$:
$$\hat{I}_{\text{IS}} = \frac{1}{N} \sum_{i=1}^N f(X_i) w(X_i), \quad w(x) = \frac{p(x)}{q(x)}$$

**Optimal $q^*$:** Proportional to $|f(x)|p(x)$ → zero variance (impractical but guides design)

#### 2. Control Variates
Use correlated variable $g(X)$ with known mean $\mu_g$:
$$\hat{I}_{\text{CV}} = \hat{I}_f - c(\hat{I}_g - \mu_g)$$

**Optimal $c^*$:** $c^* = \frac{\text{Cov}[f,g]}{\text{Var}[g]}$

**Variance reduction:** $\text{Var}[\hat{I}_{\text{CV}}] = \text{Var}[\hat{I}_f](1 - \rho^2)$

#### 3. Antithetic Sampling
Generate pairs $(X, -X)$ to induce negative correlation:
$$\hat{I}_{\text{AS}} = \frac{1}{2N} \sum_{i=1}^N [f(X_i) + f(-X_i)]$$

Works when $f$ is monotonic.

### Rare Event Estimation

**Challenge:** For $P < 10^{-3}$, naive MC needs $N \sim 10^6$ for 10% relative error.

**Relative error:** $\frac{\text{SE}(\hat{P})}{P} \approx \frac{1}{\sqrt{NP}}$

**Solution:** Importance sampling with shifted proposal centered on rare event region.

**Exponential tilting:** For Gaussian tails, optimal shift is $\mu^* = c$ (threshold).

---

## 🗂️ Module Structure

```
05_simulation_monte_carlo/
├── src/
│   ├── __init__.py              # Module exports
│   ├── mc_integration.py        # MC integration + CIs
│   ├── variance_reduction.py    # IS, control variates, antithetic
│   ├── rare_events.py           # Rare event estimators
│   ├── synthetic_generators.py  # Physics-inspired datasets
│   ├── config.py                # Experiment configs
│   └── main.py                  # CLI (run_mc_demo, generate_physics_data)
├── tests/
│   ├── test_mc_integration.py   # Integration accuracy tests
│   ├── test_variance_reduction.py # VRF verification
│   ├── test_rare_events.py      # Tail probability tests
│   └── test_synthetic_generators.py # Process properties tests
├── notebooks/
│   ├── 01_variance_reduction_effectiveness.ipynb  # VR comparison
│   └── 02_rare_events_pitfalls.ipynb             # Rare event pitfalls + solutions
├── configs/                     # YAML experiment configs
├── reports/                     # Generated plots and results (gitignored)
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Installation
```bash
# From repository root
pip install -e ".[dev]"

# Or just core dependencies
pip install -e .
```

### CLI Commands

**Run comprehensive demo:**
```bash
# Generate error vs N plots, CI coverage, variance reduction comparison
python -m modules.05_simulation_monte_carlo.src.main run-mc-demo \
    --n-samples 10000 \
    --seed 42 \
    --output-dir reports/mc_demo

# Or use Makefile
make run-module MODULE=05_simulation_monte_carlo ARGS="run-mc-demo --n-samples 20000"
```

**Generate physics datasets:**
```bash
# Damped harmonic oscillator
python -m modules.05_simulation_monte_carlo.src.main generate-physics-data \
    --dataset oscillator \
    --n-samples 1000 \
    --noise-level 0.1 \
    --output-file data.csv

# Available: oscillator, projectile, heat, pendulum
```

**Generate stochastic processes:**
```bash
# Brownian motion
python -m modules.05_simulation_monte_carlo.src.main generate-stochastic-process \
    --process brownian \
    --n-steps 1000 \
    --output-plot brownian.png

# Available: brownian, ou, levy
```

### Running Tests
```bash
# All tests
make test-module MODULE=05_simulation_monte_carlo

# Fast tests
pytest modules/05_simulation_monte_carlo/tests/ -x

# With coverage
pytest modules/05_simulation_monte_carlo/tests/ --cov=modules.05_simulation_monte_carlo
```

---

## 📊 Key Implementations

### 1. Monte Carlo Integration (`mc_integration.py`)

```python
from modules._import_helper import safe_import_from

MCIntegrator = safe_import_from(
    '05_simulation_monte_carlo.src.mc_integration',
    'MCIntegrator'
)

# Estimate ∫₀¹ e^x dx = e - 1
integrator = MCIntegrator(
    integrand=lambda x: np.exp(x),
    lower=0.0,
    upper=1.0,
    seed=42
)

result = integrator.estimate(n_samples=10000)
print(f"Estimate: {result.estimate:.6f} ± {result.std_error:.6f}")

# Convergence analysis
convergence = integrator.convergence_analysis(
    sample_sizes=[100, 500, 1000, 5000, 10000],
    n_trials=50
)
```

### 2. Importance Sampling (`variance_reduction.py`)

```python
from modules._import_helper import safe_import_from

ImportanceSampler = safe_import_from(
    '05_simulation_monte_carlo.src.variance_reduction',
    'ImportanceSampler'
)

def target_func(x):
    return np.exp(-x**2)

def proposal_sampler(n, rng):
    return rng.normal(0, 0.5, n)  # Narrower than standard

def weight_func(x):
    # p(x) / q(x)
    return np.exp(x**2 / 2)  # Ratio of N(0,1) / N(0,0.5)

sampler = ImportanceSampler(
    target_func=target_func,
    proposal_sampler=proposal_sampler,
    weight_func=weight_func,
    seed=42
)

result = sampler.estimate(n_samples=10000)
print(f"VRF: {result.variance_reduction_factor:.2f}x")
```

### 3. Control Variates (`variance_reduction.py`)

```python
from modules._import_helper import safe_import_from

ControlVariates = safe_import_from(
    '05_simulation_monte_carlo.src.variance_reduction',
    'ControlVariates'
)

# Use X^2 as control (E[X^2] = 1 for X ~ N(0,1))
cv = ControlVariates(
    target_func=lambda x: np.exp(-x**2),
    control_func=lambda x: x**2,
    control_mean=1.0,
    seed=42
)

result = cv.estimate(n_samples=10000)
print(f"VRF: {result.variance_reduction_factor:.2f}x")
print(f"Optimal c: {result.control_coefficient:.4f}")
```

### 4. Rare Event Estimation (`rare_events.py`)

```python
from modules._import_helper import safe_import_from

RareEventEstimator = safe_import_from(
    '05_simulation_monte_carlo.src.rare_events',
    'RareEventEstimator'
)

estimator = RareEventEstimator(seed=42)

# Estimate P(X > 4) for X ~ N(0,1)
result = estimator.estimate_tail_probability(
    distribution='normal',
    threshold=4.0,
    n_samples=50000,
    method='importance_sampling'
)

print(f"P(X > 4): {result.probability:.6e}")
print(f"Rel. Error: {result.relative_error:.2%}")
print(f"VRF: {result.variance_reduction_factor:.2f}x")

# Adaptive sampling (learns optimal proposal)
adaptive_result = estimator.adaptive_sampling(
    distribution='normal',
    threshold=4.0,
    n_pilot=1000,
    n_main=9000,
    target_relative_error=0.05
)
```

### 5. Physics Datasets (`synthetic_generators.py`)

```python
from modules._import_helper import safe_import_from

PhysicsDataGenerator = safe_import_from(
    '05_simulation_monte_carlo.src.synthetic_generators',
    'PhysicsDataGenerator'
)

generator = PhysicsDataGenerator(seed=42)

# Damped harmonic oscillator: ground truth E[x(t)] = A*e^(-γt)*cos(ωt)
dataset = generator.damped_harmonic_oscillator(
    n_samples=1000,
    noise_level=0.1
)

print(dataset.features.shape)  # (1000, 2): time, velocity
print(dataset.targets.shape)   # (1000,): positions

# Use ground truth for evaluation
predictions = dataset.ground_truth_func(dataset.features[:, 0])
```

**Available datasets:**
- `damped_harmonic_oscillator`: $m\ddot{x} + c\dot{x} + kx = 0$
- `projectile_motion`: Trajectory with air resistance
- `heat_diffusion_1d`: Temperature evolution $\partial_t u = \alpha \partial_{xx} u$
- `pendulum_energy`: Total energy $E = \frac{1}{2}ml^2\dot{\theta}^2 + mgl(1-\cos\theta)$

---

## 📓 Notebooks

### 1. Variance Reduction Effectiveness (`01_variance_reduction_effectiveness.ipynb`)

**Topics:**
- Naive MC baseline
- Importance sampling with optimal proposal
- Control variates with correlated variables
- Antithetic sampling for symmetric functions
- Error vs N comparison (VRF quantification)
- Exercise: tail expectation $\mathbb{E}[X | X > 2]$

**Key Figure:** RMSE vs sample size showing VR methods shift curve down (same $1/\sqrt{N}$ slope, lower intercept)

### 2. Rare Events Pitfalls (`02_rare_events_pitfalls.ipynb`)

**Topics:**
- Pitfall #1: Zero estimate bias (infinite relative error)
- Pitfall #2: Invalid CIs when $NP < 5$
- Importance sampling with exponential tilting
- Adaptive sampling (learn optimal proposal)
- Comparison across thresholds (rarer events)
- Exercise: Heavy-tailed distribution (Student-t)

**Key Figure:** Relative error vs threshold showing IS maintains low error while naive MC explodes

---

## 🎯 Key Results

### Variance Reduction Factors (VRF)

Typical VRFs for common problems:

| Method | Application | Typical VRF |
|--------|-------------|-------------|
| Importance Sampling | Tail probabilities | 10-1000x |
| Control Variates | Correlated systems | 2-20x |
| Antithetic | Monotonic functions | 1.5-4x |

### Sample Size Requirements

For 10% relative error in tail probability $P(X > c)$:

| Method | $c=3$ ($P \sim 10^{-3}$) | $c=4$ ($P \sim 10^{-5}$) | $c=5$ ($P \sim 10^{-7}$) |
|--------|--------------------------|--------------------------|--------------------------|
| Naive MC | $10^5$ samples | $10^7$ samples | $10^9$ samples |
| Importance Sampling | $10^3$ samples | $10^4$ samples | $10^5$ samples |

**Speedup:** 100x-10,000x for rare events!

---

## 🧪 Testing Strategy

**Unit tests:** (`tests/test_*.py`)
- MC integration accuracy on known integrals (Gaussian, exponential, polynomial)
- Variance reduction factor verification (VRF > 1 for appropriate cases)
- Rare event estimates within 20% relative error
- Reproducibility with seeds

**Sanity checks:**
- Confidence interval coverage (95% CI should cover true value ~95% of time)
- Effective sample size $N_{\text{eff}} > 0.5N$ for well-designed IS
- Convergence rate $O(N^{-1/2})$ observed in practice

**Integration tests:**
- End-to-end MC integration pipeline
- Full rare event estimation workflow (naive → IS → adaptive)
- Physics dataset generation with ground truth evaluation

---
## ✅ Module Completion Checklist

**After finishing all 5 notebooks, you should be able to:**
- [ ] Estimate integrals with MC and report 95% confidence intervals
- [ ] Diagnose convergence: plot error vs N on log-log, verify O(N^{-1/2})
- [ ] Implement importance sampling and achieve 10-50x variance reduction
- [ ] Choose good proposals (heavier tails, overlap with target)
- [ ] Apply control variates when correlated controls exist (ρ > 0.5)
- [ ] Use antithetic sampling for symmetric functions
- [ ] Estimate rare events (P < 10^-3) reliably
- [ ] Generate physics-inspired synthetic datasets with ground truth
- [ ] Separate model error from noise in ML experiments

**Next steps:**
- Apply to your research: financial risk, particle physics, system reliability
- Explore advanced topics: MCMC (Module 06), multilevel MC, quasi-MC
- Read *Monte Carlo Statistical Methods* by Robert & Casella

---
## 📖 References

### Core Texts
1. **Owen (2013):** *Monte Carlo Theory, Methods and Examples* - Comprehensive modern treatment
2. **Robert & Casella (2004):** *Monte Carlo Statistical Methods* - Bayesian perspective
3. **Glasserman (2003):** *Monte Carlo Methods in Financial Engineering* - Applications focus

### Rare Events
4. **Bucklew (2004):** *Introduction to Rare Event Simulation* - Specialized treatment
5. **Asmussen & Glynn (2007):** *Stochastic Simulation* - Advanced algorithms

### Variance Reduction
6. **Hammersley & Handscomb (1964):** *Monte Carlo Methods* - Classic techniques
7. **Fishman (1996):** *Monte Carlo: Concepts, Algorithms, and Applications* - Practical guide

### Physics Applications
8. **Landau & Binder (2014):** *A Guide to Monte Carlo Simulations in Statistical Physics*
9. **Newman & Barkema (1999):** *Monte Carlo Methods in Statistical Physics*

---

## 🔧 Configuration

Example YAML config (`configs/mc_integration.yaml`):

```yaml
name: "mc_integration_convergence"
seed: 42
output_dir: "reports/mc_integration"

# MC parameters
n_samples: 10000
confidence_level: 0.95

# Convergence analysis
min_samples: 100
max_samples: 50000
n_trials: 50

# Integration bounds
lower_bound: 0.0
upper_bound: 1.0
```

Load with:
```python
from modules._import_helper import safe_import_from

MCIntegrationConfig, load_config = safe_import_from(
    '05_simulation_monte_carlo.src',
    'MCIntegrationConfig', 'load_config'
)

config = load_config(
    Path("configs/mc_integration.yaml"),
    MCIntegrationConfig
)
```

---

## 📝 Completion Checklist

- [x] Core implementations
  - [x] `mc_integration.py` - MCIntegrator with CIs
  - [x] `variance_reduction.py` - IS, control variates, antithetic
  - [x] `rare_events.py` - Rare event estimators
  - [x] `synthetic_generators.py` - Physics datasets
- [x] Tests (>80% coverage target)
  - [x] `test_mc_integration.py`
  - [x] `test_variance_reduction.py`
  - [x] `test_rare_events.py`
  - [x] `test_synthetic_generators.py`
- [x] Notebooks
  - [x] `01_variance_reduction_effectiveness.ipynb`
  - [x] `02_rare_events_pitfalls.ipynb`
- [x] CLI
  - [x] `main.py` with `run_mc_demo`, `generate_physics_data`, `generate_stochastic_process`
- [x] Configuration
  - [x] `config.py` with dataclass configs
- [x] Documentation
  - [x] README.md with theory, examples, references

---

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

**When adding new MC methods:**
1. Add implementation to appropriate source file
2. Add dataclass result type (include VRF when applicable)
3. Add tests verifying correctness on known problems
4. Add notebook demonstrating effectiveness
5. Update README with theory and example

---

## 📄 License

MIT License - See [LICENSE](../../LICENSE) for details.

---

**Module Maintainer:** Computational ML Lab  
**Last Updated:** 2024  
**Status:** ✅ Complete & Tested
