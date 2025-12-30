"""
Feature engineering utilities for crop yield prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features for modeling.
    
    Args:
        df: Input DataFrame with base features.
        
    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()
    
    # Temperature categories
    if "avg_temp" in df.columns:
        df["temp_category"] = pd.cut(
            df["avg_temp"],
            bins=[-float("inf"), 10, 20, 30, float("inf")],
            labels=["cold", "temperate", "warm", "hot"]
        )
    
    # Rainfall categories
    if "rainfall_mm" in df.columns:
        df["rainfall_category"] = pd.cut(
            df["rainfall_mm"],
            bins=[-float("inf"), 500, 1000, 2000, float("inf")],
            labels=["arid", "semi_arid", "moderate", "wet"]
        )
    
    # Pesticide intensity (log-transformed)
    if "pesticides_tonnes" in df.columns:
        df["log_pesticides"] = np.log1p(df["pesticides_tonnes"])
    
    # Interaction features
    if "avg_temp" in df.columns and "rainfall_mm" in df.columns:
        df["temp_rain_interaction"] = df["avg_temp"] * df["rainfall_mm"] / 1000
    
    return df


def get_feature_names(include_engineered: bool = False) -> Dict[str, List[str]]:
    """
    Get lists of feature names by type.
    
    Args:
        include_engineered: Whether to include engineered features.
        
    Returns:
        Dictionary with feature lists by type.
    """
    base_numeric = ["rainfall_mm", "pesticides_tonnes", "avg_temp"]
    base_categorical = ["crop", "country"]
    
    engineered_numeric = ["log_pesticides", "temp_rain_interaction"]
    engineered_categorical = ["temp_category", "rainfall_category"]
    
    if include_engineered:
        return {
            "numeric": base_numeric + engineered_numeric,
            "categorical": base_categorical + engineered_categorical,
        }
    else:
        return {
            "numeric": base_numeric,
            "categorical": base_categorical,
        }


def select_features(
    df: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    target: str = "yield"
) -> tuple:
    """
    Select features and target from DataFrame.
    
    Args:
        df: Input DataFrame.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
        target: Target column name.
        
    Returns:
        Tuple of (X DataFrame, y Series).
    """
    feature_names = get_feature_names()
    numeric_features = numeric_features or feature_names["numeric"]
    categorical_features = categorical_features or feature_names["categorical"]
    
    all_features = numeric_features + categorical_features
    available_features = [f for f in all_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df[target].copy() if target in df.columns else None
    
    return X, y


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Correlation matrix DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get descriptive statistics for all features.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Statistics DataFrame.
    """
    stats = df.describe(include="all").T
    stats["missing"] = df.isna().sum()
    stats["missing_pct"] = (df.isna().sum() / len(df) * 100).round(2)
    stats["dtype"] = df.dtypes
    return stats
