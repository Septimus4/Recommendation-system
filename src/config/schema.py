"""
Schema definitions for input validation and data contracts.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class PredictionInput:
    """Schema for yield prediction request."""
    crop: str
    country: str
    rainfall_mm: float
    pesticides_tonnes: float
    avg_temp: float
    
    def validate(self) -> List[str]:
        """Validate input values and return list of errors."""
        errors = []
        
        if self.rainfall_mm < 0:
            errors.append("rainfall_mm must be non-negative")
        if self.rainfall_mm > 5000:
            errors.append("rainfall_mm exceeds realistic maximum (5000 mm)")
            
        if self.pesticides_tonnes < 0:
            errors.append("pesticides_tonnes must be non-negative")
            
        if self.avg_temp < -10 or self.avg_temp > 50:
            errors.append("avg_temp must be between -10 and 50 degrees Celsius")
            
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model input."""
        return {
            "crop": self.crop,
            "country": self.country,
            "rainfall_mm": self.rainfall_mm,
            "pesticides_tonnes": self.pesticides_tonnes,
            "avg_temp": self.avg_temp,
        }


@dataclass
class RecommendationInput:
    """Schema for crop recommendation request."""
    country: str
    rainfall_mm: float
    pesticides_tonnes: float
    avg_temp: float
    
    def validate(self) -> List[str]:
        """Validate input values and return list of errors."""
        errors = []
        
        if self.rainfall_mm < 0:
            errors.append("rainfall_mm must be non-negative")
        if self.rainfall_mm > 5000:
            errors.append("rainfall_mm exceeds realistic maximum (5000 mm)")
            
        if self.pesticides_tonnes < 0:
            errors.append("pesticides_tonnes must be non-negative")
            
        if self.avg_temp < -10 or self.avg_temp > 50:
            errors.append("avg_temp must be between -10 and 50 degrees Celsius")
            
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "country": self.country,
            "rainfall_mm": self.rainfall_mm,
            "pesticides_tonnes": self.pesticides_tonnes,
            "avg_temp": self.avg_temp,
        }


@dataclass
class PredictionOutput:
    """Schema for yield prediction response."""
    crop: str
    predicted_yield: float
    yield_unit: str = "hg/ha"
    model_version: str = "1.0.0"


@dataclass
class CropRecommendation:
    """Single crop recommendation entry."""
    rank: int
    crop: str
    predicted_yield: float
    yield_unit: str = "hg/ha"


@dataclass
class RecommendationOutput:
    """Schema for crop recommendation response."""
    recommendations: List[CropRecommendation]
    context: Dict[str, Any]
    model_version: str = "1.0.0"
