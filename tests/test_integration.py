"""
Integration tests for API endpoints.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from schemas import (
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    CropRecommendation
)


class TestPredictionSchema:
    """Tests for prediction request/response schemas."""
    
    def test_valid_request(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            crop="Wheat",
            country="India",
            rainfall_mm=1000,
            pesticides_tonnes=5000,
            avg_temp=20
        )
        assert request.crop == "Wheat"
        assert request.rainfall_mm == 1000
    
    def test_rejects_negative_rainfall(self):
        """Test negative rainfall is rejected."""
        with pytest.raises(ValueError):
            PredictionRequest(
                crop="Wheat",
                country="India",
                rainfall_mm=-100,
                pesticides_tonnes=5000,
                avg_temp=20
            )
    
    def test_rejects_extreme_temperature(self):
        """Test extreme temperature is rejected."""
        with pytest.raises(ValueError):
            PredictionRequest(
                crop="Wheat",
                country="India",
                rainfall_mm=1000,
                pesticides_tonnes=5000,
                avg_temp=100  # Too high
            )
    
    def test_response_format(self):
        """Test prediction response format."""
        response = PredictionResponse(
            crop="Wheat",
            predicted_yield=25000.5,
            yield_unit="hg/ha",
            model_version="1.0.0"
        )
        assert response.crop == "Wheat"
        assert response.predicted_yield == 25000.5
        assert response.yield_unit == "hg/ha"


class TestRecommendationSchema:
    """Tests for recommendation request/response schemas."""
    
    def test_valid_request(self):
        """Test valid recommendation request."""
        request = RecommendationRequest(
            country="India",
            rainfall_mm=1000,
            pesticides_tonnes=5000,
            avg_temp=20,
            top_n=5
        )
        assert request.country == "India"
        assert request.top_n == 5
    
    def test_optional_top_n(self):
        """Test top_n is optional."""
        request = RecommendationRequest(
            country="India",
            rainfall_mm=1000,
            pesticides_tonnes=5000,
            avg_temp=20
        )
        assert request.top_n is None
    
    def test_response_format(self):
        """Test recommendation response format."""
        recommendations = [
            CropRecommendation(rank=1, crop="Potatoes", predicted_yield=100000),
            CropRecommendation(rank=2, crop="Wheat", predicted_yield=25000)
        ]
        
        response = RecommendationResponse(
            recommendations=recommendations,
            context={"country": "India"},
            model_version="1.0.0"
        )
        
        assert len(response.recommendations) == 2
        assert response.recommendations[0].rank == 1
        assert response.recommendations[0].crop == "Potatoes"


class TestCropRecommendation:
    """Tests for crop recommendation model."""
    
    def test_basic_creation(self):
        """Test basic recommendation creation."""
        rec = CropRecommendation(
            rank=1,
            crop="Wheat",
            predicted_yield=25000
        )
        assert rec.rank == 1
        assert rec.crop == "Wheat"
        assert rec.yield_unit == "hg/ha"  # Default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
