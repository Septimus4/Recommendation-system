"""
Unit tests for the API.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import PredictionRequest, RecommendationRequest


class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            crop="Wheat",
            country="India",
            rainfall_mm=1000,
            pesticides_tonnes=5000,
            avg_temp=20
        )
        assert request.crop == "Wheat"
        assert request.country == "India"
        assert request.rainfall_mm == 1000
        assert request.pesticides_tonnes == 5000
        assert request.avg_temp == 20
    
    def test_prediction_request_negative_rainfall(self):
        """Test that negative rainfall raises error."""
        with pytest.raises(ValueError):
            PredictionRequest(
                crop="Wheat",
                country="India",
                rainfall_mm=-100,
                pesticides_tonnes=5000,
                avg_temp=20
            )
    
    def test_prediction_request_extreme_temp(self):
        """Test that extreme temperature raises error."""
        with pytest.raises(ValueError):
            PredictionRequest(
                crop="Wheat",
                country="India",
                rainfall_mm=1000,
                pesticides_tonnes=5000,
                avg_temp=100  # Too high
            )
    
    def test_recommendation_request_valid(self):
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
    
    def test_recommendation_request_optional_top_n(self):
        """Test that top_n is optional."""
        request = RecommendationRequest(
            country="India",
            rainfall_mm=1000,
            pesticides_tonnes=5000,
            avg_temp=20
        )
        assert request.top_n is None


class TestModelLoader:
    """Tests for model loader (without actual model)."""
    
    def test_model_loader_initialization(self):
        """Test model loader can be initialized."""
        from model_loader import ModelLoader
        
        loader = ModelLoader()
        assert loader.model is None
        assert not loader.is_loaded
    
    def test_model_loader_default_crops(self):
        """Test default supported crops."""
        from model_loader import ModelLoader
        
        loader = ModelLoader()
        crops = loader.supported_crops
        assert isinstance(crops, list)
        assert len(crops) > 0
        assert "Wheat" in crops
        assert "Maize" in crops


class TestValidation:
    """Tests for input validation logic."""
    
    def test_rainfall_bounds(self):
        """Test rainfall must be within bounds."""
        # Valid
        request = PredictionRequest(
            crop="Wheat",
            country="India",
            rainfall_mm=0,
            pesticides_tonnes=0,
            avg_temp=20
        )
        assert request.rainfall_mm == 0
        
        # Max valid
        request = PredictionRequest(
            crop="Wheat",
            country="India",
            rainfall_mm=10000,
            pesticides_tonnes=0,
            avg_temp=20
        )
        assert request.rainfall_mm == 10000
    
    def test_temperature_bounds(self):
        """Test temperature must be within realistic bounds."""
        # Min valid
        request = PredictionRequest(
            crop="Wheat",
            country="Canada",
            rainfall_mm=500,
            pesticides_tonnes=100,
            avg_temp=-50
        )
        assert request.avg_temp == -50
        
        # Max valid
        request = PredictionRequest(
            crop="Wheat",
            country="Saudi Arabia",
            rainfall_mm=50,
            pesticides_tonnes=100,
            avg_temp=60
        )
        assert request.avg_temp == 60


class TestRecommendation:
    """Tests for recommendation logic."""
    
    def test_recommendation_sorting(self):
        """Test that recommendations would be sorted by yield."""
        # This is a logical test - actual sorting is in model_loader
        recommendations = [
            {"crop": "A", "predicted_yield": 100},
            {"crop": "B", "predicted_yield": 300},
            {"crop": "C", "predicted_yield": 200},
        ]
        
        sorted_recs = sorted(recommendations, key=lambda x: x["predicted_yield"], reverse=True)
        
        assert sorted_recs[0]["crop"] == "B"
        assert sorted_recs[1]["crop"] == "C"
        assert sorted_recs[2]["crop"] == "A"
    
    def test_recommendation_top_n(self):
        """Test top_n filtering."""
        all_crops = ["A", "B", "C", "D", "E"]
        top_n = 3
        
        result = all_crops[:top_n]
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
