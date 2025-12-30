"""
Model loader for inference in the API.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and inference with the trained model pipeline."""
    
    def __init__(self, model_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to the model joblib file.
            metadata_path: Path to the model metadata JSON file.
        """
        # Default paths
        base_path = Path(__file__).parent.parent
        self.model_path = model_path or base_path / "models" / "model_pipeline.joblib"
        self.metadata_path = metadata_path or base_path / "models" / "model_metadata.json"
        
        self.model = None
        self.metadata = None
        self.is_loaded = False
        
        # Default values if metadata is not available
        self._default_crops = [
            "Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", 
            "Wheat", "Cassava", "Sweet potatoes", "Plantains and others", "Yams"
        ]
        self._default_features = ["rainfall_mm", "pesticides_tonnes", "avg_temp", "crop", "country"]
    
    def load_model(self) -> None:
        """Load the model and metadata from disk."""
        try:
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded from {self.metadata_path}")
            else:
                logger.warning("Metadata file not found, using defaults")
                self.metadata = {}
            
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def model_version(self) -> str:
        """Get model version."""
        return self.metadata.get("model_name", "unknown") if self.metadata else "unknown"
    
    @property
    def supported_crops(self) -> List[str]:
        """Get list of supported crops."""
        if self.metadata and "supported_crops" in self.metadata:
            return self.metadata["supported_crops"]
        return self._default_crops
    
    @property
    def supported_countries(self) -> List[str]:
        """Get list of supported countries."""
        if self.metadata and "supported_countries" in self.metadata:
            return self.metadata["supported_countries"]
        return []
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        if self.metadata:
            numeric = self.metadata.get("numeric_features", [])
            categorical = self.metadata.get("categorical_features", [])
            return numeric + categorical
        return self._default_features
    
    def predict(
        self,
        crop: str,
        country: str,
        rainfall_mm: float,
        pesticides_tonnes: float,
        avg_temp: float
    ) -> float:
        """
        Make a yield prediction for a single input.
        
        Args:
            crop: Crop name.
            country: Country name.
            rainfall_mm: Average rainfall in mm.
            pesticides_tonnes: Pesticides usage in tonnes.
            avg_temp: Average temperature in Celsius.
            
        Returns:
            Predicted yield value.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Create input DataFrame
        input_df = pd.DataFrame([{
            "rainfall_mm": rainfall_mm,
            "pesticides_tonnes": pesticides_tonnes,
            "avg_temp": avg_temp,
            "crop": crop,
            "country": country
        }])
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        # Ensure non-negative yield
        return max(0, float(prediction))
    
    def recommend(
        self,
        country: str,
        rainfall_mm: float,
        pesticides_tonnes: float,
        avg_temp: float,
        top_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Recommend crops based on predicted yields.
        
        Args:
            country: Country name.
            rainfall_mm: Average rainfall in mm.
            pesticides_tonnes: Pesticides usage in tonnes.
            avg_temp: Average temperature in Celsius.
            top_n: Number of top recommendations to return.
            
        Returns:
            List of crop recommendations sorted by yield.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        crops = self.supported_crops
        
        # Create input DataFrame for all crops
        input_data = []
        for crop in crops:
            input_data.append({
                "rainfall_mm": rainfall_mm,
                "pesticides_tonnes": pesticides_tonnes,
                "avg_temp": avg_temp,
                "crop": crop,
                "country": country
            })
        
        input_df = pd.DataFrame(input_data)
        
        # Make predictions for all crops
        predictions = self.model.predict(input_df)
        
        # Create results
        results = []
        for i, crop in enumerate(crops):
            results.append({
                "rank": 0,
                "crop": crop,
                "predicted_yield": max(0, float(predictions[i])),
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


# Singleton instance for the API
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
        _model_loader.load_model()
    return _model_loader
