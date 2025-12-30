"""
Preprocessing utilities for machine learning pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    handle_unknown: str = "ignore"
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        handle_unknown: How to handle unknown categories ('ignore' or 'error').
        
    Returns:
        ColumnTransformer preprocessor.
    """
    # Numeric preprocessing: impute missing + scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical preprocessing: impute missing + one-hot encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )
    
    return preprocessor


def fit_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame
) -> ColumnTransformer:
    """
    Fit the preprocessor on training data.
    
    Args:
        preprocessor: ColumnTransformer to fit.
        X: Training feature DataFrame.
        
    Returns:
        Fitted preprocessor.
    """
    return preprocessor.fit(X)


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    numeric_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Get feature names after preprocessing (including one-hot encoded).
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        numeric_features: Original numeric feature names.
        categorical_features: Original categorical feature names.
        
    Returns:
        List of all feature names after transformation.
    """
    feature_names = list(numeric_features)
    
    # Get one-hot encoded feature names
    if hasattr(preprocessor, "transformers_"):
        for name, transformer, columns in preprocessor.transformers_:
            if name == "categorical" and hasattr(transformer, "named_steps"):
                encoder = transformer.named_steps.get("encoder")
                if encoder and hasattr(encoder, "get_feature_names_out"):
                    cat_names = encoder.get_feature_names_out(categorical_features)
                    feature_names.extend(cat_names)
    
    return feature_names


def prepare_inference_input(
    crop: str,
    country: str,
    rainfall_mm: float,
    pesticides_tonnes: float,
    avg_temp: float
) -> pd.DataFrame:
    """
    Prepare a single input row for model inference.
    
    Args:
        crop: Crop name.
        country: Country name.
        rainfall_mm: Average rainfall in mm.
        pesticides_tonnes: Pesticides usage in tonnes.
        avg_temp: Average temperature in Celsius.
        
    Returns:
        DataFrame with single row ready for prediction.
    """
    return pd.DataFrame([{
        "crop": crop,
        "country": country,
        "rainfall_mm": rainfall_mm,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp
    }])


def prepare_recommendation_inputs(
    crops: List[str],
    country: str,
    rainfall_mm: float,
    pesticides_tonnes: float,
    avg_temp: float
) -> pd.DataFrame:
    """
    Prepare input rows for all crops (for recommendation).
    
    Args:
        crops: List of crop names to evaluate.
        country: Country name.
        rainfall_mm: Average rainfall in mm.
        pesticides_tonnes: Pesticides usage in tonnes.
        avg_temp: Average temperature in Celsius.
        
    Returns:
        DataFrame with one row per crop.
    """
    rows = []
    for crop in crops:
        rows.append({
            "crop": crop,
            "country": country,
            "rainfall_mm": rainfall_mm,
            "pesticides_tonnes": pesticides_tonnes,
            "avg_temp": avg_temp
        })
    return pd.DataFrame(rows)
