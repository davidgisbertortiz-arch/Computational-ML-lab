"""Tests to verify no data leakage in pipeline."""

import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from modules._import_helper import safe_import_from

generate_particle_collision_data, split_data, get_feature_columns = safe_import_from(
    '03_ml_tabular_foundations.src.data',
    'generate_particle_collision_data', 'split_data', 'get_feature_columns'
)
create_logistic_pipeline, create_lightgbm_pipeline = safe_import_from(
    '03_ml_tabular_foundations.src.models',
    'create_logistic_pipeline', 'create_lightgbm_pipeline'
)


class TestNoLeakage:
    """Test suite to prevent common data leakage bugs."""
    
    def test_scaler_fit_only_on_train(self):
        """Verify scaler is fit only on training data, not validation/test."""
        # Generate data
        df = generate_particle_collision_data(n_samples=1000, random_state=42)
        train_df, val_df, test_df = split_data(df, random_state=42)
        
        feature_cols = get_feature_columns(df)
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        
        # Manual scaler (what should happen inside pipeline)
        scaler = StandardScaler()
        scaler.fit(X_train)  # Fit only on train
        
        # Get statistics
        train_mean = scaler.mean_
        train_std = scaler.scale_
        
        # If we incorrectly fit on train+val
        scaler_leaky = StandardScaler()
        scaler_leaky.fit(np.vstack([X_train, X_val]))  # WRONG!
        leaky_mean = scaler_leaky.mean_
        leaky_std = scaler_leaky.scale_
        
        # They should be different (leakage changes statistics)
        assert not np.allclose(train_mean, leaky_mean), \
            "Scaler statistics should differ when fit on different data"
        
        # Pipeline should use correct (train-only) statistics
        pipeline = create_logistic_pipeline(random_state=42)
        pipeline.fit(X_train, train_df['is_signal'].values)
        
        # Extract scaler from pipeline
        pipeline_scaler = pipeline.named_steps['scaler']
        
        # Verify pipeline scaler matches train-only scaler
        np.testing.assert_allclose(pipeline_scaler.mean_, train_mean)
        np.testing.assert_allclose(pipeline_scaler.scale_, train_std)
        
        print("✅ No leakage: Scaler fit only on training data")
    
    def test_no_data_overlap_in_splits(self):
        """Verify train/val/test splits have no overlapping samples."""
        df = generate_particle_collision_data(n_samples=1000, random_state=42)
        train_df, val_df, test_df = split_data(df, random_state=42)
        
        # Check indices don't overlap
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)
        
        assert len(train_idx & val_idx) == 0, "Train and val sets overlap!"
        assert len(train_idx & test_idx) == 0, "Train and test sets overlap!"
        assert len(val_idx & test_idx) == 0, "Val and test sets overlap!"
        
        # Check all indices accounted for
        all_idx = train_idx | val_idx | test_idx
        assert len(all_idx) == len(df), "Some samples missing from splits"
        
        print("✅ No overlap: Splits are disjoint")
    
    def test_stratified_split_preserves_distribution(self):
        """Verify stratified split maintains class balance across splits."""
        df = generate_particle_collision_data(n_samples=1000, random_state=42)
        train_df, val_df, test_df = split_data(df, random_state=42, stratify=True)
        
        overall_signal_rate = df['is_signal'].mean()
        train_signal_rate = train_df['is_signal'].mean()
        val_signal_rate = val_df['is_signal'].mean()
        test_signal_rate = test_df['is_signal'].mean()
        
        # All splits should have similar class balance (within 2%)
        assert abs(train_signal_rate - overall_signal_rate) < 0.02
        assert abs(val_signal_rate - overall_signal_rate) < 0.02
        assert abs(test_signal_rate - overall_signal_rate) < 0.02
        
        print(f"✅ Stratification preserved:")
        print(f"   Overall: {overall_signal_rate:.1%}")
        print(f"   Train: {train_signal_rate:.1%}")
        print(f"   Val: {val_signal_rate:.1%}")
        print(f"   Test: {test_signal_rate:.1%}")
    
    def test_pipeline_transform_consistency(self):
        """Verify pipeline applies same transform to train and val data."""
        df = generate_particle_collision_data(n_samples=1000, random_state=42)
        train_df, val_df, _ = split_data(df, random_state=42)
        
        feature_cols = get_feature_columns(df)
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values
        
        # Fit pipeline on train
        pipeline = create_logistic_pipeline(random_state=42)
        pipeline.fit(X_train, train_df['is_signal'].values)
        
        # Transform both
        X_train_scaled = pipeline.named_steps['scaler'].transform(X_train)
        X_val_scaled = pipeline.named_steps['scaler'].transform(X_val)
        
        # Train data should be standardized (mean~0, std~1)
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-6)
        
        # Val data should use train statistics (mean NOT ~0)
        val_mean = X_val_scaled.mean(axis=0)
        assert not np.allclose(val_mean, 0, atol=0.1), \
            "Val data should not be centered at 0 (uses train statistics)"
        
        print("✅ Pipeline consistency: Same transform applied to train and val")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
