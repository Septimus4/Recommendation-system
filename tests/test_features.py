"""
Unit tests for feature engineering and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.engineering import (
    create_features,
    get_feature_names,
    select_features,
    compute_correlation_matrix
)


class TestCreateFeatures:
    """Tests for feature creation."""
    
    def test_creates_temp_category(self):
        """Test temperature category creation."""
        df = pd.DataFrame({
            'avg_temp': [5, 15, 25, 35],
            'rainfall_mm': [500, 1000, 1500, 2000]
        })
        result = create_features(df)
        
        assert 'temp_category' in result.columns
        assert result['temp_category'].iloc[0] == 'cold'
        assert result['temp_category'].iloc[1] == 'temperate'
        assert result['temp_category'].iloc[2] == 'warm'
        assert result['temp_category'].iloc[3] == 'hot'
    
    def test_creates_rainfall_category(self):
        """Test rainfall category creation."""
        df = pd.DataFrame({
            'rainfall_mm': [300, 700, 1500, 3000],
            'avg_temp': [20, 20, 20, 20]
        })
        result = create_features(df)
        
        assert 'rainfall_category' in result.columns
        assert result['rainfall_category'].iloc[0] == 'arid'
        assert result['rainfall_category'].iloc[1] == 'semi_arid'
        assert result['rainfall_category'].iloc[2] == 'moderate'
        assert result['rainfall_category'].iloc[3] == 'wet'
    
    def test_creates_log_pesticides(self):
        """Test log pesticides feature."""
        df = pd.DataFrame({
            'pesticides_tonnes': [0, 100, 1000]
        })
        result = create_features(df)
        
        assert 'log_pesticides' in result.columns
        assert result['log_pesticides'].iloc[0] == np.log1p(0)
        assert result['log_pesticides'].iloc[1] == np.log1p(100)


class TestGetFeatureNames:
    """Tests for feature name retrieval."""
    
    def test_base_features(self):
        """Test base feature names."""
        features = get_feature_names(include_engineered=False)
        
        assert 'numeric' in features
        assert 'categorical' in features
        assert 'rainfall_mm' in features['numeric']
        assert 'crop' in features['categorical']
    
    def test_engineered_features(self):
        """Test engineered feature names."""
        features = get_feature_names(include_engineered=True)
        
        assert 'log_pesticides' in features['numeric']
        assert 'temp_category' in features['categorical']


class TestSelectFeatures:
    """Tests for feature selection."""
    
    def test_selects_correct_columns(self):
        """Test correct column selection."""
        df = pd.DataFrame({
            'rainfall_mm': [1000],
            'pesticides_tonnes': [5000],
            'avg_temp': [20],
            'crop': ['Wheat'],
            'country': ['India'],
            'yield': [25000],
            'other_col': ['ignored']
        })
        
        X, y = select_features(df)
        
        assert 'yield' not in X.columns
        assert 'other_col' not in X.columns
        assert len(y) == 1
        assert y.iloc[0] == 25000


class TestCorrelationMatrix:
    """Tests for correlation computation."""
    
    def test_computes_correlation(self):
        """Test correlation matrix computation."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],  # Perfectly correlated with a
            'c': [5, 4, 3, 2, 1]   # Negatively correlated with a
        })
        
        corr = compute_correlation_matrix(df)
        
        assert corr.loc['a', 'b'] == pytest.approx(1.0)
        assert corr.loc['a', 'c'] == pytest.approx(-1.0)
    
    def test_excludes_non_numeric(self):
        """Test that non-numeric columns are excluded."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'text': ['x', 'y', 'z']
        })
        
        corr = compute_correlation_matrix(df)
        
        assert 'text' not in corr.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
