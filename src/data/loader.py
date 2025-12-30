"""
Data loading utilities for crop yield prediction system.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from src.config.constants import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    YIELD_RAW_FILE,
    YIELD_DF_RAW_FILE,
    PESTICIDES_RAW_FILE,
    RAINFALL_RAW_FILE,
    TEMP_RAW_FILE,
    CONSOLIDATED_FILE,
)


def load_yield_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw yield data from FAO format.
    
    Args:
        file_path: Path to yield CSV file. Defaults to YIELD_RAW_FILE.
        
    Returns:
        DataFrame with yield data.
    """
    file_path = file_path or YIELD_RAW_FILE
    df = pd.read_csv(file_path)
    return df


def load_yield_df_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load pre-merged yield dataframe (yield + context features).
    
    Args:
        file_path: Path to yield_df CSV file. Defaults to YIELD_DF_RAW_FILE.
        
    Returns:
        DataFrame with yield and context data.
    """
    file_path = file_path or YIELD_DF_RAW_FILE
    df = pd.read_csv(file_path, index_col=0)
    return df


def load_pesticides_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load pesticides usage data.
    
    Args:
        file_path: Path to pesticides CSV file.
        
    Returns:
        DataFrame with pesticides data.
    """
    file_path = file_path or PESTICIDES_RAW_FILE
    df = pd.read_csv(file_path)
    return df


def load_rainfall_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load rainfall data.
    
    Args:
        file_path: Path to rainfall CSV file.
        
    Returns:
        DataFrame with rainfall data.
    """
    file_path = file_path or RAINFALL_RAW_FILE
    df = pd.read_csv(file_path)
    return df


def load_temperature_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load temperature data.
    
    Args:
        file_path: Path to temperature CSV file.
        
    Returns:
        DataFrame with temperature data.
    """
    file_path = file_path or TEMP_RAW_FILE
    df = pd.read_csv(file_path)
    return df


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load all raw datasets.
    
    Returns:
        Dictionary with all raw dataframes.
    """
    return {
        "yield": load_yield_data(),
        "yield_df": load_yield_df_data(),
        "pesticides": load_pesticides_data(),
        "rainfall": load_rainfall_data(),
        "temperature": load_temperature_data(),
    }


def load_consolidated_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the consolidated, cleaned dataset.
    
    Args:
        file_path: Path to consolidated CSV file.
        
    Returns:
        DataFrame with consolidated data ready for modeling.
    """
    file_path = file_path or CONSOLIDATED_FILE
    if not file_path.exists():
        raise FileNotFoundError(
            f"Consolidated data file not found at {file_path}. "
            "Run the data fusion pipeline first."
        )
    df = pd.read_csv(file_path)
    return df


def save_consolidated_data(df: pd.DataFrame, file_path: Optional[Path] = None) -> None:
    """
    Save consolidated dataset to CSV.
    
    Args:
        df: DataFrame to save.
        file_path: Output path. Defaults to CONSOLIDATED_FILE.
    """
    file_path = file_path or CONSOLIDATED_FILE
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Saved consolidated data to {file_path}")
