#!/bin/bash
# Quick start script for Module 04: Time Series & State Space

set -e

echo "=========================================="
echo "Module 04: Time Series & State Space"
echo "Quick Start Demo"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "modules/04_time_series_state_space" ]; then
    echo "❌ Error: Please run this script from the repository root"
    exit 1
fi

echo "📦 Setting up module..."
echo ""

# Run constant velocity tracking demo
echo "🎯 Demo 1: Constant Velocity Tracking (Kalman Filter)"
echo "----------------------------------------------"
python -m modules.04_time_series_state_space.src.main run-tracking-demo \
    --config-path modules/04_time_series_state_space/configs/tracking_default.yaml \
    --seed 42
echo ""

# Run pendulum tracking demo
echo "🎯 Demo 2: Pendulum Tracking (Extended Kalman Filter)"
echo "----------------------------------------------"
python -m modules.04_time_series_state_space.src.main run-tracking-demo \
    --config-path modules/04_time_series_state_space/configs/tracking_pendulum.yaml \
    --seed 42
echo ""

# Run forecasting demo
echo "🎯 Demo 3: Time Series Forecasting (Backtesting)"
echo "----------------------------------------------"
python -m modules.04_time_series_state_space.src.main run-forecasting-demo \
    --config-path modules/04_time_series_state_space/configs/forecasting_default.yaml \
    --seed 42
echo ""

echo "=========================================="
echo "✅ All demos complete!"
echo ""
echo "📊 Check results in:"
echo "   - modules/04_time_series_state_space/reports/tracking_cv/"
echo "   - modules/04_time_series_state_space/reports/tracking_pendulum/"
echo "   - modules/04_time_series_state_space/reports/forecasting/"
echo ""
echo "🧪 Run tests with:"
echo "   pytest modules/04_time_series_state_space/tests/ -v"
echo ""
echo "📓 Open notebooks:"
echo "   jupyter notebook modules/04_time_series_state_space/notebooks/"
echo "=========================================="
