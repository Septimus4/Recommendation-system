"""
Model inference utilities for crop yield prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import joblib

from src.config.constants import MODEL_ARTIFACT_PATH, SUPPORTED_CROPS


def load_model(path: Optional[Path] = None):
    """
    Load a trained model pipeline.
    
    Args:
        path: Path to model file.
        
    Returns:
        Loaded pipeline.
    """
    path = path or MODEL_ARTIFACT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def predict_yield(
    model,
    crop: str,
    country: str,
    rainfall_mm: float,
    pesticides_tonnes: float,
    avg_temp: float
) -> float:
    """
    Predict yield for a single crop and context.
    
    Args:
        model: Trained model pipeline.
        crop: Crop name.
        country: Country name.
        rainfall_mm: Average rainfall in mm.
        pesticides_tonnes: Pesticides usage in tonnes.
        avg_temp: Average temperature.
        
    Returns:
        Predicted yield value.
    """
    input_df = pd.DataFrame([{
        "crop": crop,
        "country": country,
        "rainfall_mm": rainfall_mm,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp
    }])
    
    prediction = model.predict(input_df)[0]
    return float(prediction)


def recommend_crops(
    model,
    country: str,
    rainfall_mm: float,
    pesticides_tonnes: float,
    avg_temp: float,
    crops: Optional[List[str]] = None,
    top_n: Optional[int] = None
) -> List[Dict]:
    """
    Recommend crops based on predicted yields.
    
    Args:
        model: Trained model pipeline.
        country: Country name.
        rainfall_mm: Average rainfall in mm.
        pesticides_tonnes: Pesticides usage in tonnes.
        avg_temp: Average temperature.
        crops: List of crops to consider. Defaults to SUPPORTED_CROPS.
        top_n: Number of top recommendations to return. None returns all.
        
    Returns:
        List of dictionaries with crop recommendations sorted by yield.
    """
    crops = crops or SUPPORTED_CROPS
    
    # Create input DataFrame with all crops
    input_data = []
    for crop in crops:
        input_data.append({
            "crop": crop,
            "country": country,
            "rainfall_mm": rainfall_mm,
            "pesticides_tonnes": pesticides_tonnes,
            "avg_temp": avg_temp
        })
    
    input_df = pd.DataFrame(input_data)
    
    # Get predictions
    predictions = model.predict(input_df)
    
    # Create results
    results = []
    for i, crop in enumerate(crops):
        results.append({
            "rank": 0,  # Will be set after sorting
            "crop": crop,
            "predicted_yield": float(predictions[i]),
            "yield_unit": "hg/ha"
        })
    
    # Sort by predicted yield descending
    results.sort(key=lambda x: x["predicted_yield"], reverse=True)
    
    # Assign ranks
    for i, result in enumerate(results):
        result["rank"] = i + 1
    
    # Return top N if specified
    if top_n is not None:
        results = results[:top_n]
    
    return results


def validate_crop(crop: str, supported_crops: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate if a crop is supported.
    
    Args:
        crop: Crop name to validate.
        supported_crops: List of supported crops.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    supported_crops = supported_crops or SUPPORTED_CROPS
    
    if crop in supported_crops:
        return True, ""
    
    # Try case-insensitive match
    crop_lower = crop.lower()
    for supported in supported_crops:
        if supported.lower() == crop_lower:
            return True, f"Did you mean '{supported}'?"
    
    return False, f"Crop '{crop}' is not supported. Supported crops: {supported_crops}"


def get_supported_crops() -> List[str]:
    """Get list of supported crops."""
    return SUPPORTED_CROPS.copy()
