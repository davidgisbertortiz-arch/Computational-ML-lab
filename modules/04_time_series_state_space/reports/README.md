# Reports Directory

This directory contains generated outputs from experiments:

- **tracking_cv/** - Constant velocity tracking results (Kalman Filter)
  - `tracking_results.png` - State estimates vs ground truth
  - `tracking_errors.png` - Estimation errors over time
  - `metrics.json` - RMSE and other metrics

- **tracking_pendulum/** - Pendulum tracking results (EKF/PF)
  - `tracking_results.png` - Angle and velocity estimates
  - `tracking_errors.png` - Tracking errors
  - `metrics.json` - Performance metrics

- **forecasting/** - Time series forecasting results
  - `forecasting_results.png` - Forecasts vs actuals
  - `metrics.json` - Backtesting metrics (MAE, RMSE)

All files in this directory are gitignored (generated artifacts).

## Regenerating Results

```bash
# Run tracking demos
python -m modules.04_time_series_state_space.src.main run-tracking-demo \
    --config-path configs/tracking_default.yaml --seed 42

# Run forecasting demo
python -m modules.04_time_series_state_space.src.main run-forecasting-demo \
    --config-path configs/forecasting_default.yaml --seed 42
```
