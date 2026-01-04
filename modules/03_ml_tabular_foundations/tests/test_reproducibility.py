"""Tests for reproducibility with fixed seeds."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
generate_particle_collision_data, split_data, get_feature_columns = safe_import_from(
    '03_ml_tabular_foundations.src.data',
    'generate_particle_collision_data', 'split_data', 'get_feature_columns'
)
create_logistic_pipeline, create_lightgbm_pipeline = safe_import_from(
    '03_ml_tabular_foundations.src.models',
    'create_logistic_pipeline', 'create_lightgbm_pipeline'
)


class TestReproducibility:
    """Test deterministic behavior with fixed seeds."""
    
    def test_data_generation_reproducible(self):
        """Verify same seed produces identical data."""
        df1 = generate_particle_collision_data(n_samples=100, random_state=42)
        df2 = generate_particle_collision_data(n_samples=100, random_state=42)
        
        pd.testing.assert_frame_equal(df1, df2)
        print("✅ Data generation is reproducible")
    
    def test_logistic_training_reproducible(self):
        """Verify logistic regression produces identical models with same seed."""
        set_seed(42)
        df = generate_particle_collision_data(n_samples=500, random_state=42)
        train_df, _, _ = split_data(df, random_state=42)
        
        feature_cols = get_feature_columns(df)
        X = train_df[feature_cols].values
        y = train_df['is_signal'].values
        
        # Train twice with same seed
        model1 = create_logistic_pipeline(random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)
        
        model2 = create_logistic_pipeline(random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
        print("✅ Logistic regression training is reproducible")
    
    def test_lightgbm_training_reproducible(self):
        """Verify LightGBM produces identical models with same seed."""
        set_seed(42)
        df = generate_particle_collision_data(n_samples=500, random_state=42)
        train_df, _, _ = split_data(df, random_state=42)
        
        feature_cols = get_feature_columns(df)
        X = train_df[feature_cols].values
        y = train_df['is_signal'].values
        
        # Train twice with same seed (small n_estimators for speed)
        model1 = create_lightgbm_pipeline(n_estimators=50, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)
        
        model2 = create_lightgbm_pipeline(n_estimators=50, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
        print("✅ LightGBM training is reproducible")
    
    def test_train_test_split_reproducible(self):
        """Verify splits are reproducible with same seed."""
        df = generate_particle_collision_data(n_samples=100, random_state=42)
        
        train1, val1, test1 = split_data(df, random_state=42)
        train2, val2, test2 = split_data(df, random_state=42)
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)
        
        print("✅ Train/test splits are reproducible")


if __name__ == "__main__":
    import pandas as pd
    pytest.main([__file__, "-v", "-s"])
