"""Config-driven training CLI for tabular ML experiments.

Commands:
- train: Train model with cross-validation and save artifacts
- evaluate: Evaluate trained model on test set
"""

from pathlib import Path
from typing import Optional
import pickle
import json
import typer
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

app = typer.Typer(help="Tabular ML training and evaluation CLI")


class TabularConfig(BaseModel):
    """Configuration for tabular ML experiment."""
    name: str = Field(..., description="Experiment name")
    seed: int = Field(42, description="Random seed")
    
    # Model config
    model_type: str = Field("lightgbm", description="Model type: logistic, lightgbm")
    model_params: dict = Field(default_factory=dict, description="Model hyperparameters")
    
    # Data config
    n_samples: int = Field(100_000, description="Number of samples to generate")
    signal_fraction: float = Field(0.1, description="Fraction of signal events")
    
    # Training config
    cv_folds: int = Field(5, description="Number of CV folds")
    calibrate: bool = Field(True, description="Apply Platt scaling")
    optimize_threshold: bool = Field(True, description="Optimize decision threshold")
    target_precision: float = Field(0.95, description="Target precision for threshold selection")
    
    # Output config
    output_dir: Path = Field(Path("reports"), description="Output directory")
    save_plots: bool = Field(True, description="Save evaluation plots")


@app.command()
def train(
    config_path: Path = typer.Option(..., "--config", help="Path to YAML config"),
    seed: Optional[int] = typer.Option(None, help="Override seed from config"),
):
    """Train model with config-driven experiment."""
    from modules._import_helper import safe_import_from
    import yaml
    
    set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
    
    # Load config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    config = TabularConfig(**config_dict)
    
    if seed is not None:
        config.seed = seed
    
    set_seed(config.seed)
    
    # Import locally to avoid circular imports
    from .data import load_data, split_data, get_feature_columns
    from .models import ModelRegistry, calibrate_model, select_optimal_threshold
    from .eval import (
        compute_metrics, plot_roc_curve, plot_precision_recall_curve,
        plot_calibration_curve, compute_permutation_importance, plot_feature_importance
    )
    
    print(f"🚀 Starting experiment: {config.name}")
    print(f"   Model: {config.model_type}")
    print(f"   Seed: {config.seed}")
    
    # Load and split data
    print("\n📊 Loading data...")
    df = load_data()
    train_df, val_df, test_df = split_data(df, random_state=config.seed)
    
    feature_cols = get_feature_columns(df)
    X_train = train_df[feature_cols].values
    y_train = train_df['is_signal'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['is_signal'].values
    
    print(f"   Train: {len(train_df):,} samples ({y_train.mean():.1%} signal)")
    print(f"   Val: {len(val_df):,} samples ({y_val.mean():.1%} signal)")
    print(f"   Test: {len(test_df):,} samples ({test_df['is_signal'].mean():.1%} signal)")
    
    # Create and train model
    print(f"\n🔧 Training {config.model_type} model...")
    model = ModelRegistry.create(config.model_type, random_state=config.seed, **config.model_params)
    model.fit(X_train, y_train)
    print("   ✅ Model trained")
    
    # Evaluate on validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_proba)
    
    print(f"\n📈 Validation metrics:")
    print(f"   AUC-ROC: {val_metrics['auc_roc']:.4f}")
    print(f"   Average Precision: {val_metrics['average_precision']:.4f}")
    print(f"   Recall@P95: {val_metrics['recall_at_p95']:.4f}")
    print(f"   ECE: {val_metrics['ece']:.4f}")
    print(f"   Brier Score: {val_metrics['brier_score']:.4f}")
    
    # Calibration
    if config.calibrate:
        print(f"\n🎯 Calibrating model...")
        model_calibrated = calibrate_model(model, X_val, y_val)
        y_val_proba_cal = model_calibrated.predict_proba(X_val)[:, 1]
        val_metrics_cal = compute_metrics(y_val, y_val_proba_cal)
        print(f"   ECE improvement: {val_metrics['ece']:.4f} → {val_metrics_cal['ece']:.4f}")
        model = model_calibrated
        y_val_proba = y_val_proba_cal
    
    # Threshold optimization
    optimal_threshold = 0.5
    if config.optimize_threshold:
        print(f"\n🎲 Optimizing decision threshold...")
        optimal_threshold = select_optimal_threshold(y_val, y_val_proba, config.target_precision)
        print(f"   Optimal threshold: {optimal_threshold:.4f} (target precision: {config.target_precision})")
        val_metrics_opt = compute_metrics(y_val, y_val_proba, threshold=optimal_threshold)
        print(f"   Recall@P95: {val_metrics['recall_at_p95']:.4f} → {val_metrics_opt['recall_at_p95']:.4f}")
    
    # Save artifacts
    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n💾 Saving artifacts to {output_dir}...")
    
    # Save model
    with open(model_dir / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    # Save metadata
    metadata = {
        'model_type': config.model_type,
        'seed': config.seed,
        'optimal_threshold': optimal_threshold,
        'calibrated': config.calibrate,
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
    }
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save plots
    if config.save_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        plot_roc_curve(y_val, y_val_proba, save_path=plots_dir / "roc_curve.png")
        plot_precision_recall_curve(y_val, y_val_proba, save_path=plots_dir / "pr_curve.png")
        plot_calibration_curve(y_val, y_val_proba, save_path=plots_dir / "calibration.png")
        
        # Feature importance
        print(f"\n🔍 Computing feature importance...")
        importance_df = compute_permutation_importance(
            model, X_val, y_val, feature_cols, n_repeats=5, random_state=config.seed
        )
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        plot_feature_importance(importance_df, save_path=plots_dir / "feature_importance.png")
        
        print(f"   ✅ Plots saved to {plots_dir}")
    
    print(f"\n✅ Training complete! Artifacts saved to {output_dir}")


@app.command()
def evaluate(
    model_path: Path = typer.Option(..., help="Path to trained model pickle"),
    output_path: Path = typer.Option("reports/test_metrics.json", help="Output path for metrics"),
):
    """Evaluate trained model on test set."""
    from modules._import_helper import safe_import_from
    import pickle
    
    set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')
    set_seed(42)
    
    from .data import load_data, split_data, get_feature_columns
    from .eval import compute_metrics
    
    print(f"🔍 Evaluating model from {model_path}")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load test data
    df = load_data()
    _, _, test_df = split_data(df, random_state=42)
    
    feature_cols = get_feature_columns(df)
    X_test = test_df[feature_cols].values
    y_test = test_df['is_signal'].values
    
    # Predict
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_test_proba)
    
    print(f"\n📊 Test Set Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # Save metrics
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    print(f"\n✅ Metrics saved to {output_path}")


if __name__ == "__main__":
    app()
