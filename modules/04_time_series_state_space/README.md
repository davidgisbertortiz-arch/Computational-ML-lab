# Module 04: Time Series & State Space

**Status**: ✅ Complete

## 📚 What You'll Learn

- Kalman filtering for linear Gaussian state-space models
- Extended Kalman Filter (EKF) for nonlinear systems
- Particle filtering (Sequential Monte Carlo) for non-Gaussian/nonlinear problems
- Time series forecasting with proper backtesting methodology
- Data leakage prevention in temporal data

## 🎯 Learning Objectives

By the end of this module, you will:
1. Understand state-space representation of dynamical systems
2. Implement and apply Kalman filters for tracking and estimation
3. Handle nonlinear dynamics with EKF and particle filters
4. Build proper time-series forecasting pipelines with rolling-window validation
5. Recognize and prevent temporal data leakage

## 📖 Theory

### State-Space Models

A state-space model describes a system's evolution over time:

```
State equation:    x_t = f(x_{t-1}, u_t, w_t)
Observation equation: y_t = h(x_t, v_t)
```

Where:
- `x_t` = hidden state at time t
- `y_t` = observation at time t
- `w_t, v_t` = process and observation noise
- `f, h` = state transition and observation functions

### Kalman Filter (Linear Case)

For linear Gaussian systems, the Kalman filter provides **optimal** state estimation:

```
Prediction:
  x̂_{t|t-1} = F x̂_{t-1|t-1} + B u_t
  P_{t|t-1} = F P_{t-1|t-1} F^T + Q

Update (when observation y_t arrives):
  K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}  # Kalman gain
  x̂_{t|t} = x̂_{t|t-1} + K_t (y_t - H x̂_{t|t-1})
  P_{t|t} = (I - K_t H) P_{t|t-1}
```

**Key insight**: Optimally combines prediction (based on model) with measurement (data).

### Extended Kalman Filter (Nonlinear)

For nonlinear systems, EKF linearizes around current estimate:
1. Use nonlinear functions `f, h` directly in prediction/update
2. Compute Jacobians F = ∂f/∂x, H = ∂h/∂x for covariance propagation

**Limitation**: Only accurate if nonlinearity is weak or noise is small.

### Particle Filter

For highly nonlinear/non-Gaussian systems:
1. Represent belief by N particles (samples) instead of Gaussian
2. Propagate particles through nonlinear dynamics
3. Reweight by observation likelihood
4. Resample to avoid degeneracy

**Trade-off**: More flexible but computationally expensive (requires many particles).

### Time Series Forecasting

**Critical**: Temporal data requires special validation to prevent leakage.

**Rolling-window backtesting**:
```
Train: [------train------]|
Test:                      [test]
                           
Train: [------train------][test]|
Test:                             [test]

... continue rolling forward
```

**Never** shuffle time-series data or use future information in features.

## � Educational Notebooks

The `notebooks/` directory contains end-to-end tutorials covering the module content:

1. **[01_noise_models_and_simulation.ipynb](notebooks/01_noise_models_and_simulation.ipynb)**
   - **Topic**: Process vs measurement noise, SNR, observability
   - **What you'll learn**: Simulate tracking systems, understand noise impact, SNR analysis
   - **Exercises**: Prediction tasks, SNR sweeps, non-Gaussian noise, correlated noise

2. **[02_kalman_filter_linear_state_estimation.ipynb](notebooks/02_kalman_filter_linear_state_estimation.ipynb)**
   - **Topic**: Kalman filter derivation and implementation
   - **What you'll learn**: KF recursion, covariance evolution, noise sensitivity, common pitfalls
   - **Exercises**: Manual prediction step, Q/R tuning, innovation autocorrelation

3. **[03_nonlinear_estimation_ekf_vs_particle.ipynb](notebooks/03_nonlinear_estimation_ekf_vs_particle.ipynb)**
   - **Topic**: Extended Kalman Filter vs Particle Filter comparison
   - **What you'll learn**: Nonlinear pendulum tracking, particle degeneracy, computational trade-offs
   - **Exercises**: Tune particle count, test without resampling, nonlinear measurements

4. **[04_time_series_forecasting_backtesting.ipynb](notebooks/04_time_series_forecasting_backtesting.ipynb)**
   - **Topic**: Proper time series evaluation (no data leakage!)
   - **What you'll learn**: Rolling-window backtesting, leakage demonstration, metrics, uncertainty
   - **Exercises**: Exponential smoothing, expanding window, residual analysis

5. **[05_neural_time_series_state_space_decoding.ipynb](notebooks/05_neural_time_series_state_space_decoding.ipynb)** *(Optional: Neuroscience)*
   - **Topic**: Decoding latent neural states from spike trains
   - **What you'll learn**: Poisson observations, particle filter for non-Gaussian data, binning trade-offs
   - **Exercises**: Particle count study, multi-neuron decoding, oscillatory states

**How to run**: 
```bash
jupyter notebook modules/04_time_series_state_space/notebooks/
# Or use VS Code's notebook interface
```

Each notebook is self-contained (~2-3 min runtime), saves outputs to `reports/nb0X_*/`, and includes exercises with solutions.

---

## �🛠️ Implementation Plan

### Part 1: State Estimation
- [x] Linear Kalman Filter (constant velocity tracking)
- [x] Extended Kalman Filter (pendulum dynamics)
- [x] Bootstrap Particle Filter
- [x] Simulation utilities (ground truth + noisy observations)

### Part 2: Forecasting
- [x] Rolling-window backtesting framework
- [x] Baseline models (naive, moving average, ARIMA-style)
- [x] Leakage guards (time-aware splits)
- [x] Evaluation metrics (MAE, RMSE, directional accuracy)

### Part 3: Experiments
- [x] Tracking demo: compare KF/EKF/PF under different noise levels
- [x] Forecasting demo: proper backtesting on synthetic data
- [x] Failure mode analysis

## 🧪 Experiments & Metrics

### Experiment 1: Object Tracking
- **Goal**: Compare filter performance under varying noise
- **Metrics**: RMSE (position), computational time
- **Baseline**: Raw noisy observations
- **Setup**: Constant velocity model with position measurements

### Experiment 2: Nonlinear Tracking
- **Goal**: When does EKF fail vs. Particle Filter succeed?
- **Metrics**: RMSE, estimation uncertainty calibration
- **System**: Pendulum with angle measurements

### Experiment 3: Time Series Forecasting
- **Goal**: Validate proper backtesting prevents overfitting
- **Metrics**: Rolling MAE, RMSE, forecast vs actual plots
- **Baseline**: Naive (last value), moving average

## ⚠️ Failure Modes

1. **Kalman Filter Divergence**
   - Cause: Process noise Q too small → filter trusts model too much
   - Fix: Increase Q or add model mismatch terms

2. **EKF Linearization Errors**
   - Cause: Nonlinearity too strong for first-order approximation
   - Fix: Use particle filter or Unscented Kalman Filter

3. **Particle Degeneracy**
   - Cause: Too few particles or poor proposal distribution
   - Fix: Increase N, adaptive resampling, better proposals

4. **Time Series Leakage**
   - Cause: Using future information (e.g., scaling on full dataset)
   - Fix: Rolling-window splits, fit scalers only on train window

5. **Covariance Matrix Issues**
   - Cause: Numerical instability (P not positive definite)
   - Fix: Joseph form update, add small regularization

## 📊 Results Summary

See `reports/` for experiment outputs:
- `tracking_comparison.png` - KF vs EKF vs PF trajectories
- `noise_sensitivity.png` - RMSE vs noise level
- `backtesting_results.csv` - Rolling forecast metrics

## ✅ Definition of Done

- [x] All algorithms implemented with type hints
- [x] Docstrings with equations and references
- [x] Tests: prediction/update shapes, reproducibility with seed
- [x] Notebooks: tracking demo + forecasting demo
- [x] CLI: `run_tracking_demo` with configurable parameters
- [x] Failure modes documented
- [x] Coverage > 80%

## 📚 References

- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Julier & Uhlmann (1997). "Unscented Kalman Filter"
- Arulampalam et al. (2002). "Tutorial on Particle Filters"
- Hyndman & Athanasopoulos (2021). "Forecasting: Principles and Practice"
