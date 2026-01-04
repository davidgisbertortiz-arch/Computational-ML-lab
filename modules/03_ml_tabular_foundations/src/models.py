"""Model definitions with sklearn pipelines.

Implements:
- Baseline: Logistic Regression
- Production: LightGBM Gradient Boosting
- Calibration wrapper: Platt scaling
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb


def create_logistic_pipeline(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    """Create logistic regression pipeline with standard scaling.
    
    Args:
        C: Inverse of regularization strength (smaller = stronger regularization)
        max_iter: Maximum iterations for optimization
        random_state: Random seed
        **kwargs: Additional LogisticRegression parameters
        
    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        ))
    ])


def create_lightgbm_pipeline(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 7,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    """Create LightGBM pipeline with standard scaling.
    
    Args:
        n_estimators: Number of boosting rounds
        learning_rate: Step size shrinkage
        num_leaves: Max number of leaves in one tree
        max_depth: Max tree depth (limits model complexity)
        min_child_samples: Minimum samples per leaf (prevents overfitting)
        subsample: Fraction of data to sample for each tree
        colsample_bytree: Fraction of features to sample for each tree
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        random_state: Random seed
        **kwargs: Additional LGBMClassifier parameters
        
    Returns:
        sklearn Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **kwargs
        ))
    ])


def calibrate_model(
    pipeline: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = 'sigmoid',
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Calibrate model probabilities using Platt scaling or isotonic regression.
    
    Args:
        pipeline: Trained sklearn pipeline
        X_cal: Calibration features
        y_cal: Calibration labels
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        cv: Number of CV folds for calibration
        
    Returns:
        Calibrated classifier
    """
    calibrated = CalibratedClassifierCV(
        pipeline,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def select_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_precision: float = 0.95,
) -> float:
    """Select decision threshold to achieve target precision.
    
    For imbalanced classification, default threshold=0.5 is often suboptimal.
    This finds the threshold that maximizes recall while maintaining desired precision.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        target_precision: Minimum acceptable precision
        
    Returns:
        Optimal threshold (or 1.0 if target precision unachievable)
    """
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find thresholds where precision >= target
    valid_idx = np.where(precisions >= target_precision)[0]
    
    if len(valid_idx) == 0:
        # Target precision not achievable
        return 1.0
    
    # Among valid thresholds, select one with highest recall
    best_idx = valid_idx[np.argmax(recalls[valid_idx])]
    
    # precision_recall_curve returns n+1 thresholds
    if best_idx < len(thresholds):
        return thresholds[best_idx]
    else:
        # Edge case: best threshold is at boundary
        return thresholds[-1]


class ModelRegistry:
    """Registry for creating models from config."""
    
    _MODELS = {
        'logistic': create_logistic_pipeline,
        'lightgbm': create_lightgbm_pipeline,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs) -> Pipeline:
        """Create model from type string and parameters.
        
        Args:
            model_type: One of 'logistic', 'lightgbm'
            **kwargs: Parameters for model constructor
            
        Returns:
            Model pipeline
            
        Raises:
            ValueError: If model_type unknown
        """
        if model_type not in cls._MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls._MODELS.keys())}"
            )
        
        return cls._MODELS[model_type](**kwargs)


if __name__ == "__main__":
    """Test model creation."""
    print("Testing model creation...")
    
    # Test logistic
    log_pipe = create_logistic_pipeline()
    print(f"✅ Logistic pipeline: {log_pipe}")
    
    # Test LightGBM
    lgb_pipe = create_lightgbm_pipeline()
    print(f"✅ LightGBM pipeline: {lgb_pipe}")
    
    # Test registry
    model = ModelRegistry.create('logistic', C=0.5)
    print(f"✅ Registry creation: {model}")
