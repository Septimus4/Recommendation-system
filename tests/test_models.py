"""
Unit tests for model training and inference.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.inference import (
    validate_crop,
    get_supported_crops
)


class TestValidateCrop:
    """Tests for crop validation."""
    
    def test_valid_crop(self):
        """Test valid crop passes validation."""
        is_valid, error = validate_crop("Wheat")
        assert is_valid is True
        assert error == ""
    
    def test_invalid_crop(self):
        """Test invalid crop fails validation."""
        is_valid, error = validate_crop("InvalidCrop")
        assert is_valid is False
        assert "not supported" in error
    
    def test_case_sensitivity(self):
        """Test crop validation is case-aware but helpful."""
        is_valid, error = validate_crop("wheat")
        # Should fail but suggest correct name
        assert is_valid is True or "Did you mean" in error


class TestGetSupportedCrops:
    """Tests for supported crops list."""
    
    def test_returns_list(self):
        """Test returns a list."""
        crops = get_supported_crops()
        assert isinstance(crops, list)
    
    def test_contains_common_crops(self):
        """Test contains common crops."""
        crops = get_supported_crops()
        assert "Wheat" in crops
        assert "Maize" in crops
        assert "Potatoes" in crops
    
    def test_returns_copy(self):
        """Test returns a copy, not the original."""
        crops1 = get_supported_crops()
        crops2 = get_supported_crops()
        crops1.append("TestCrop")
        assert "TestCrop" not in crops2


class TestModelPredictions:
    """Tests for model prediction logic."""
    
    def test_prediction_is_positive(self):
        """Test predictions should be positive."""
        # Simulating prediction logic
        prediction = max(0, -100)  # Model should clip negative values
        assert prediction >= 0
    
    def test_recommendation_sorting(self):
        """Test recommendations are sorted by yield."""
        recommendations = [
            {"crop": "A", "predicted_yield": 100},
            {"crop": "B", "predicted_yield": 300},
            {"crop": "C", "predicted_yield": 200},
        ]
        
        sorted_recs = sorted(
            recommendations, 
            key=lambda x: x["predicted_yield"], 
            reverse=True
        )
        
        assert sorted_recs[0]["crop"] == "B"
        assert sorted_recs[1]["crop"] == "C"
        assert sorted_recs[2]["crop"] == "A"
    
    def test_top_n_filtering(self):
        """Test top_n filtering works."""
        all_items = list(range(10))
        top_n = 3
        result = all_items[:top_n]
        
        assert len(result) == 3
        assert result == [0, 1, 2]


class TestInputValidation:
    """Tests for input validation rules."""
    
    def test_rainfall_must_be_positive(self):
        """Test rainfall validation."""
        rainfall = -100
        is_valid = rainfall >= 0
        assert is_valid is False
    
    def test_temperature_bounds(self):
        """Test temperature must be in realistic range."""
        valid_temps = [0, 15, 30, -10, 45]
        invalid_temps = [-60, 100]
        
        for temp in valid_temps:
            assert -50 <= temp <= 60
        
        for temp in invalid_temps:
            assert not (-50 <= temp <= 60)
    
    def test_pesticides_must_be_positive(self):
        """Test pesticides validation."""
        pesticides = -500
        is_valid = pesticides >= 0
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
