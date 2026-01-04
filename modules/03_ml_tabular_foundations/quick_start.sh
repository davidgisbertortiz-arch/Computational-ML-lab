#!/bin/bash
# Quick start guide for Module 03

set -e

echo "🚀 Module 03: ML Tabular Foundations - Quick Start"
echo ""

# Step 1: Generate data
echo "Step 1: Generating synthetic particle collision dataset..."
python -m modules.03_ml_tabular_foundations.src.data
echo "✅ Data generated: data/particle_collisions.csv"
echo ""

# Step 2: Train baseline
echo "Step 2: Training baseline (Logistic Regression)..."
python -m modules.03_ml_tabular_foundations.src.train train \
  --config modules/03_ml_tabular_foundations/configs/baseline.yaml
echo "✅ Baseline trained"
echo ""

# Step 3: Train production model
echo "Step 3: Training production model (LightGBM)..."
python -m modules.03_ml_tabular_foundations.src.train train \
  --config modules/03_ml_tabular_foundations/configs/lightgbm.yaml
echo "✅ Production model trained"
echo ""

# Step 4: Run tests
echo "Step 4: Running tests..."
pytest modules/03_ml_tabular_foundations/tests/ -v
echo "✅ All tests passed"
echo ""

echo "🎉 Module 03 complete!"
echo ""
echo "📊 Results:"
echo "   - Models: reports/*/models/"
echo "   - Metrics: reports/*/metadata.json"
echo "   - Plots: reports/*/plots/"
echo ""
echo "Next steps:"
echo "  1. Open notebooks/01_eda.ipynb for data exploration"
echo "  2. Compare baseline vs production metrics"
echo "  3. Analyze feature importance"
