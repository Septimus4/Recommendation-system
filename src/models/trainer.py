"""
Model training utilities for crop yield prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from src.config.constants import RANDOM_STATE, MODELS_DIR


def get_baseline_model() -> DummyRegressor:
    """Get baseline model (predicts mean)."""
    return DummyRegressor(strategy="mean")


def get_ridge_model(alpha: float = 1.0) -> Ridge:
    """Get Ridge regression model."""
    return Ridge(alpha=alpha, random_state=RANDOM_STATE)


def get_elasticnet_model(alpha: float = 1.0, l1_ratio: float = 0.5) -> ElasticNet:
    """Get ElasticNet model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)


def get_random_forest_model(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2
) -> RandomForestRegressor:
    """Get Random Forest model."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def get_gradient_boosting_model(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1
) -> GradientBoostingRegressor:
    """Get Gradient Boosting model."""
    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE
    )


def get_hist_gradient_boosting_model(
    max_iter: int = 100,
    max_depth: Optional[int] = None,
    learning_rate: float = 0.1
) -> HistGradientBoostingRegressor:
    """Get Histogram-based Gradient Boosting model."""
    return HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=RANDOM_STATE
    )


def create_model_pipeline(preprocessor, model) -> Pipeline:
    """
    Create a complete pipeline with preprocessing and model.
    
    Args:
        preprocessor: Fitted or unfitted preprocessor.
        model: Scikit-learn model.
        
    Returns:
        Pipeline object.
    """
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Train a model pipeline.
    
    Args:
        pipeline: Model pipeline.
        X_train: Training features.
        y_train: Training target.
        
    Returns:
        Fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        pipeline: Fitted model pipeline.
        X: Features.
        y: True target values.
        
    Returns:
        Dictionary of metrics.
    """
    y_pred = pipeline.predict(X)
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
        "mape": np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100  # Avoid division by zero
    }
    
    return metrics


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    groups: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Perform cross-validation.
    
    Args:
        pipeline: Model pipeline.
        X: Features.
        y: Target.
        cv: Number of folds.
        groups: Groups for GroupKFold (e.g., years).
        
    Returns:
        Dictionary with CV scores.
    """
    scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
    
    if groups is not None:
        cv_splitter = GroupKFold(n_splits=cv)
        cv_iter = cv_splitter.split(X, y, groups)
    else:
        cv_iter = cv
    
    results = {}
    
    for score_name in scoring:
        scores = cross_val_score(pipeline, X, y, cv=cv_iter, scoring=score_name)
        metric_name = score_name.replace("neg_", "").replace("_", " ")
        results[f"cv_{metric_name}_mean"] = -scores.mean() if "neg" in score_name else scores.mean()
        results[f"cv_{metric_name}_std"] = scores.std()
    
    return results


def save_model(pipeline: Pipeline, path: Optional[Path] = None) -> Path:
    """
    Save trained model pipeline.
    
    Args:
        pipeline: Fitted pipeline.
        path: Output path.
        
    Returns:
        Path where model was saved.
    """
    path = path or (MODELS_DIR / "model_pipeline.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")
    return path


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str]
) -> Optional[pd.DataFrame]:
    """
    Get feature importance from the model.
    
    Args:
        pipeline: Fitted pipeline with tree-based model.
        feature_names: List of feature names.
        
    Returns:
        DataFrame with feature importances or None if not available.
    """
    model = pipeline.named_steps.get("model")
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        
        # Handle case where feature names don't match
        if len(importances) != len(feature_names):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    return None
