# Module 04: Time Series & State Space - Complete! 🎉

## What Was Built

A comprehensive module on **physics-style state estimation** and **time series forecasting** with proper temporal validation.

### Core Implementations

#### 1. **Kalman Filter** (`kalman.py`)
- Classic linear Kalman filter for Gaussian systems
- Optimal state estimation via predict-update cycle
- Constant velocity tracking model included
- Joseph-form covariance update for stability

#### 2. **Extended Kalman Filter** (`ekf.py`)
- Handles nonlinear dynamics via linearization
- Jacobian-based covariance propagation
- Pendulum dynamics example included
- Good for weakly nonlinear systems

#### 3. **Particle Filter** (`particle_filter.py`)
- Bootstrap (SIR) particle filter
- Systematic resampling (low-variance)
- Handles arbitrary nonlinear/non-Gaussian systems
- Effective sample size monitoring

#### 4. **Time Series Forecasting** (`forecasting.py`)
- **Rolling-window backtesting** framework (prevents leakage!)
- Baseline methods: naive, MA, seasonal naive, exponential smoothing
- **LeakageGuard** utilities for temporal validation
- Comprehensive forecast metrics

#### 5. **CLI & Configs** (`main.py`, `config.py`)
- Typer-based CLI with two commands:
  - `run-tracking-demo` - State estimation experiments
  - `run-forecasting-demo` - Time series backtesting
- Pydantic configs extending `ExperimentConfig`
- YAML configuration files for reproducibility

### Tests (31 total)

- ✅ `test_kalman.py` - 8 tests (shapes, stability, tracking)
- ✅ `test_ekf.py` - 5 tests (nonlinear dynamics, Jacobians)
- ✅ `test_particle_filter.py` - 6 tests (resampling, reproducibility)
- ✅ `test_forecasting.py` - 12 tests (splits, leakage, methods, integration)

All tests verify:
- Correct shapes and dimensions
- Reproducibility with seeds
- Numerical stability (positive definite covariances)
- Sanity checks (filters reduce error vs raw observations)
- Temporal ordering enforcement

### Documentation

- **README.md** - Complete learning objectives, theory, experiments, failure modes
- **IMPLEMENTATION_SUMMARY.md** - Technical details and validation
- **Notebook** - Interactive demo of KF and EKF tracking
- **Config files** - Ready-to-run experiments

## Key Features

### 🎯 Physics Identity
- Implements full Bayesian filtering framework
- Optimal information fusion (model + measurements)
- Proper uncertainty quantification (covariance matrices)
- Handles linear, weakly nonlinear, and highly nonlinear systems

### 🔒 Temporal Data Best Practices
- **Rolling-window backtesting** (no future leakage)
- Time-aware train/test splits
- Leakage detection utilities
- Never shuffles temporal data

### 🔬 Engineering Quality
- Type hints on all functions
- Docstrings with mathematical equations
- Deterministic with seeds
- >80% test coverage goal
- Clean src/ vs tests/ vs notebooks/ separation

## Running the Module

### Quick Start
```bash
# Make script executable
chmod +x modules/04_time_series_state_space/quick_start.sh

# Run all demos
./modules/04_time_series_state_space/quick_start.sh
```

### Individual Commands
```bash
# Constant velocity tracking (Kalman Filter)
python -m modules.04_time_series_state_space.src.main run-tracking-demo \
    --config-path modules/04_time_series_state_space/configs/tracking_default.yaml \
    --seed 42

# Pendulum tracking (Extended Kalman Filter)
python -m modules.04_time_series_state_space.src.main run-tracking-demo \
    --config-path modules/04_time_series_state_space/configs/tracking_pendulum.yaml \
    --seed 42

# Time series forecasting
python -m modules.04_time_series_state_space.src.main run-forecasting-demo \
    --config-path modules/04_time_series_state_space/configs/forecasting_default.yaml \
    --seed 42
```

### Running Tests
```bash
# All tests
pytest modules/04_time_series_state_space/tests/ -v

# With coverage
pytest modules/04_time_series_state_space/tests/ \
    --cov=modules.04_time_series_state_space.src \
    --cov-report=term-missing
```

### Using in Code
```python
from modules._import_helper import safe_import_from

# Import filters
KalmanFilter, constant_velocity_model, position_observation_model = safe_import_from(
    '04_time_series_state_space.src.kalman',
    'KalmanFilter', 'constant_velocity_model', 'position_observation_model'
)

# Setup constant velocity model
dt = 0.1
F, Q = constant_velocity_model(dt, process_noise=0.01)
H, R = position_observation_model(obs_noise=0.5)

# Create and initialize filter
kf = KalmanFilter(F, H, Q, R)
kf.initialize(x0=np.array([0, 1]), P0=np.eye(2))

# Filtering loop
for measurement in measurements:
    kf.predict()
    kf.update(np.array([measurement]))
    state_estimate, covariance = kf.get_state()
```

## Learning Outcomes

Students who complete this module will:

1. ✅ Understand state-space representation of dynamical systems
2. ✅ Implement optimal Kalman filtering from scratch
3. ✅ Handle nonlinearity with EKF and particle filters
4. ✅ Build proper time-series pipelines without data leakage
5. ✅ Choose appropriate filter based on system characteristics
6. ✅ Quantify estimation uncertainty properly

## File Structure

```
modules/04_time_series_state_space/
├── README.md                         # Learning guide
├── IMPLEMENTATION_SUMMARY.md         # Technical details
├── quick_start.sh                    # Demo runner script
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── kalman.py                    # Kalman Filter (215 lines)
│   ├── ekf.py                       # Extended Kalman Filter (213 lines)
│   ├── particle_filter.py          # Particle Filter (222 lines)
│   ├── forecasting.py               # Time series methods (362 lines)
│   ├── config.py                    # Pydantic configs (58 lines)
│   └── main.py                      # CLI entry point (374 lines)
│
├── configs/                          # YAML configurations
│   ├── tracking_default.yaml        # Constant velocity + KF
│   ├── tracking_pendulum.yaml       # Pendulum + EKF
│   └── forecasting_default.yaml     # Time series backtesting
│
├── tests/                            # 31 comprehensive tests
│   ├── test_kalman.py               # 8 tests (142 lines)
│   ├── test_ekf.py                  # 5 tests (90 lines)
│   ├── test_particle_filter.py      # 6 tests (169 lines)
│   └── test_forecasting.py          # 12 tests (213 lines)
│
├── notebooks/                        # Educational materials
│   └── 01_state_estimation_demo.ipynb  # Interactive KF/EKF demo
│
└── reports/                          # Generated outputs (gitignored)
    ├── tracking_cv/
    ├── tracking_pendulum/
    └── forecasting/
```

**Total Lines of Code**: ~1,444 lines (src/) + ~614 lines (tests) = **2,058 lines**

## Next Steps

### For Learning
1. Run `quick_start.sh` to see all demos
2. Open and execute `notebooks/01_state_estimation_demo.ipynb`
3. Read through implementation in `src/` files
4. Modify configs to experiment with different parameters
5. Run tests to understand validation

### For Portfolio
This module demonstrates:
- ✅ Deep understanding of state estimation theory
- ✅ From-scratch implementation of classical algorithms
- ✅ Proper handling of temporal data (no leakage!)
- ✅ Production-quality code with tests
- ✅ Clear documentation and reproducibility

## Integration with Other Modules

- **Module 00**: Uses `ExperimentConfig`, `set_seed`, `safe_import_from`
- **Module 01**: Could extend with numerical optimization for parameter tuning
- **Module 02**: Complements Bayesian inference (filters are sequential Bayesian updates)
- **Module 05**: Foundation for advanced Monte Carlo methods (MCMC, SMC)

## References & Further Reading

- Kalman (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Anderson & Moore (1979). "Optimal Filtering"
- Arulampalam et al. (2002). "Tutorial on Particle Filters"
- Särkkä (2013). "Bayesian Filtering and Smoothing"
- Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice"

---

## Summary

**Module 04 is complete and production-ready!** 🚀

- ✅ 4 filters implemented (KF, EKF, PF, + forecasting methods)
- ✅ 31 comprehensive tests
- ✅ Full documentation
- ✅ CLI + configs + notebook
- ✅ Proper temporal validation (no leakage)
- ✅ Physics/math identity (state-space models, uncertainty)
- ✅ Engineering quality (types, tests, reproducibility)

This module serves as both a **learning resource** (theory → implementation → experiments) and a **portfolio piece** (production-quality state estimation library).
