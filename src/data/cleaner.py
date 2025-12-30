"""
Data cleaning utilities for crop yield prediction system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from src.config.constants import COLUMN_MAPPING, COUNTRY_MAPPING


def standardize_column_names(df: pd.DataFrame, mapping: Optional[Dict] = None) -> pd.DataFrame:
    """
    Standardize column names using mapping.
    
    Args:
        df: Input DataFrame.
        mapping: Column name mapping dictionary.
        
    Returns:
        DataFrame with standardized column names.
    """
    mapping = mapping or COLUMN_MAPPING
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)
    return df


def standardize_country_names(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """
    Standardize country names using mapping.
    
    Args:
        df: Input DataFrame.
        country_col: Name of country column.
        
    Returns:
        DataFrame with standardized country names.
    """
    df = df.copy()
    if country_col in df.columns:
        df[country_col] = df[country_col].str.strip()
        df[country_col] = df[country_col].replace(COUNTRY_MAPPING)
    return df


def clean_text_column(series: pd.Series) -> pd.Series:
    """
    Clean text column by stripping whitespace and standardizing case.
    
    Args:
        series: Input Series.
        
    Returns:
        Cleaned Series.
    """
    return series.str.strip()


def clean_yield_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the pre-merged yield dataframe.
    
    Args:
        df: Raw yield_df DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Standardize country names
    df = standardize_country_names(df)
    
    # Clean crop names
    if "crop" in df.columns:
        df["crop"] = clean_text_column(df["crop"])
    
    # Ensure year is integer
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    
    # Handle missing values in numeric columns
    numeric_cols = ["yield", "rainfall_mm", "pesticides_tonnes", "avg_temp"]
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Remove rows where target is missing
    if "yield" in df.columns:
        df = df.dropna(subset=["yield"])
    
    # Remove duplicates
    key_cols = ["country", "crop", "year"]
    key_cols = [c for c in key_cols if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    
    return df


def clean_context_data(df: pd.DataFrame, context_type: str) -> pd.DataFrame:
    """
    Clean context data (pesticides, rainfall, temperature).
    
    Args:
        df: Raw context DataFrame.
        context_type: Type of context data ('pesticides', 'rainfall', 'temperature').
        
    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    if context_type == "pesticides":
        # Rename columns
        df = df.rename(columns={
            "Area": "country",
            "Year": "year",
            "Value": "pesticides_tonnes"
        })
        df = df[["country", "year", "pesticides_tonnes"]]
        df["pesticides_tonnes"] = pd.to_numeric(df["pesticides_tonnes"], errors="coerce")
        
    elif context_type == "rainfall":
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            "Area": "country",
            "Year": "year",
            "average_rain_fall_mm_per_year": "rainfall_mm"
        })
        df = df[["country", "year", "rainfall_mm"]]
        df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        
    elif context_type == "temperature":
        df = df.rename(columns={
            "country": "country",
            "year": "year",
            "avg_temp": "avg_temp"
        })
        df = df[["country", "year", "avg_temp"]]
        df["avg_temp"] = pd.to_numeric(df["avg_temp"], errors="coerce")
    
    # Standardize country names
    df = standardize_country_names(df)
    
    # Ensure year is integer
    df["year"] = df["year"].astype(int)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["country", "year"], keep="first")
    
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values in numeric columns.
    
    Args:
        df: Input DataFrame.
        strategy: Imputation strategy ('median', 'mean', 'country_median').
        
    Returns:
        DataFrame with imputed values.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col == "yield":  # Don't impute target
            continue
            
        if df[col].isna().any():
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "country_median":
                # Impute with country-specific median, then global median
                df[col] = df.groupby("country")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[col] = df[col].fillna(df[col].median())
    
    return df


def remove_outliers(df: pd.DataFrame, column: str, n_std: float = 3) -> pd.DataFrame:
    """
    Remove outliers based on standard deviation.
    
    Args:
        df: Input DataFrame.
        column: Column to check for outliers.
        n_std: Number of standard deviations for outlier threshold.
        
    Returns:
        DataFrame with outliers removed.
    """
    df = df.copy()
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - n_std * std
    upper_bound = mean + n_std * std
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df
