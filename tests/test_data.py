"""
Unit tests for data loading and cleaning functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.cleaner import (
    standardize_column_names,
    clean_text_column,
    impute_missing_values
)


class TestStandardizeColumnNames:
    """Tests for column name standardization."""
    
    def test_basic_rename(self):
        """Test basic column renaming."""
        df = pd.DataFrame({'Area': ['USA'], 'Item': ['Wheat']})
        mapping = {'Area': 'country', 'Item': 'crop'}
        result = standardize_column_names(df, mapping)
        
        assert 'country' in result.columns
        assert 'crop' in result.columns
        assert 'Area' not in result.columns
    
    def test_strips_whitespace(self):
        """Test that column names are stripped."""
        df = pd.DataFrame({' Area ': ['USA'], 'Item ': ['Wheat']})
        result = standardize_column_names(df, {})
        
        # Columns should be stripped of whitespace
        assert 'Area' in result.columns or 'country' in result.columns
        assert 'Item' in result.columns or 'crop' in result.columns
    
    def test_preserves_unmapped_columns(self):
        """Test unmapped columns are preserved."""
        df = pd.DataFrame({'Area': ['USA'], 'Other': [123]})
        mapping = {'Area': 'country'}
        result = standardize_column_names(df, mapping)
        
        assert 'Other' in result.columns


class TestCleanTextColumn:
    """Tests for text column cleaning."""
    
    def test_strips_whitespace(self):
        """Test whitespace stripping."""
        series = pd.Series(['  USA  ', 'India ', ' Brazil'])
        result = clean_text_column(series)
        
        assert result.iloc[0] == 'USA'
        assert result.iloc[1] == 'India'
        assert result.iloc[2] == 'Brazil'
    
    def test_handles_empty_strings(self):
        """Test handling of empty strings."""
        series = pd.Series(['', '  ', 'USA'])
        result = clean_text_column(series)
        
        assert result.iloc[0] == ''
        assert result.iloc[1] == ''
        assert result.iloc[2] == 'USA'


class TestImputeMissingValues:
    """Tests for missing value imputation."""
    
    def test_median_imputation(self):
        """Test median imputation strategy."""
        df = pd.DataFrame({
            'value': [1, 2, 3, np.nan, 5],
            'yield': [100, 200, 300, 400, 500]  # Target shouldn't be imputed
        })
        result = impute_missing_values(df, strategy='median')
        
        assert result['value'].isna().sum() == 0
        assert result['value'].iloc[3] == 2.5  # Median of 1,2,3,5
    
    def test_mean_imputation(self):
        """Test mean imputation strategy."""
        df = pd.DataFrame({
            'value': [10, 20, 30, np.nan],
            'yield': [100, 200, 300, 400]
        })
        result = impute_missing_values(df, strategy='mean')
        
        assert result['value'].isna().sum() == 0
        assert result['value'].iloc[3] == 20.0  # Mean of 10,20,30
    
    def test_does_not_impute_target(self):
        """Test that yield column is not imputed."""
        df = pd.DataFrame({
            'value': [1, 2, 3],
            'yield': [100, np.nan, 300]
        })
        result = impute_missing_values(df, strategy='median')
        
        # yield should still have NaN
        assert result['yield'].isna().sum() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
