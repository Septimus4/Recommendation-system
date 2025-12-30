"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class PredictionRequest(BaseModel):
    """Request schema for yield prediction."""
    crop: str = Field(..., description="Name of the crop", example="Wheat")
    country: str = Field(..., description="Country name", example="India")
    rainfall_mm: float = Field(..., ge=0, le=10000, description="Average annual rainfall in mm", example=1000)
    pesticides_tonnes: float = Field(..., ge=0, description="Pesticides usage in tonnes", example=5000)
    avg_temp: float = Field(..., ge=-50, le=60, description="Average temperature in Celsius", example=20)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "crop": "Wheat",
                    "country": "India",
                    "rainfall_mm": 1000,
                    "pesticides_tonnes": 5000,
                    "avg_temp": 20
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for yield prediction."""
    crop: str = Field(..., description="Name of the crop")
    predicted_yield: float = Field(..., description="Predicted yield value")
    yield_unit: str = Field(default="hg/ha", description="Unit of yield measurement")
    model_version: str = Field(default="1.0.0", description="Model version used for prediction")


class RecommendationRequest(BaseModel):
    """Request schema for crop recommendation."""
    country: str = Field(..., description="Country name", example="India")
    rainfall_mm: float = Field(..., ge=0, le=10000, description="Average annual rainfall in mm", example=1000)
    pesticides_tonnes: float = Field(..., ge=0, description="Pesticides usage in tonnes", example=5000)
    avg_temp: float = Field(..., ge=-50, le=60, description="Average temperature in Celsius", example=20)
    top_n: Optional[int] = Field(default=None, ge=1, le=20, description="Number of top recommendations")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "country": "India",
                    "rainfall_mm": 1000,
                    "pesticides_tonnes": 5000,
                    "avg_temp": 20,
                    "top_n": 5
                }
            ]
        }
    }


class CropRecommendation(BaseModel):
    """Single crop recommendation entry."""
    rank: int = Field(..., description="Rank position (1 = best)")
    crop: str = Field(..., description="Name of the crop")
    predicted_yield: float = Field(..., description="Predicted yield value")
    yield_unit: str = Field(default="hg/ha", description="Unit of yield measurement")


class RecommendationResponse(BaseModel):
    """Response schema for crop recommendation."""
    recommendations: List[CropRecommendation] = Field(..., description="List of crop recommendations")
    context: Dict[str, Any] = Field(..., description="Input context used for recommendation")
    model_version: str = Field(default="1.0.0", description="Model version used")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_version: str = Field(..., description="Model version")
    supported_crops: List[str] = Field(..., description="List of supported crops")
    supported_countries: List[str] = Field(..., description="List of supported countries (sample)")
    features: List[str] = Field(..., description="Input feature names")


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    detail: str = Field(..., description="Error detail message")
