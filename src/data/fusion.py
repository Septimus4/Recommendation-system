"""
Data fusion utilities for combining yield and context datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.config.constants import PROCESSED_DATA_DIR


def merge_datasets(
    yield_df: pd.DataFrame,
    pesticides_df: Optional[pd.DataFrame] = None,
    rainfall_df: Optional[pd.DataFrame] = None,
    temperature_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge yield data with context data (pesticides, rainfall, temperature).
    
    Since yield_df already contains merged data, this function handles
    additional merging or re-merging from raw sources if needed.
    
    Args:
        yield_df: Yield DataFrame (possibly pre-merged).
        pesticides_df: Pesticides DataFrame.
        rainfall_df: Rainfall DataFrame.
        temperature_df: Temperature DataFrame.
        
    Returns:
        Merged DataFrame.
    """
    result = yield_df.copy()
    
    # Check if data is already merged
    has_context = all(col in result.columns for col in ["rainfall_mm", "pesticides_tonnes", "avg_temp"])
    
    if has_context:
        # Data is already merged, return as is
        return result
    
    # Merge pesticides data
    if pesticides_df is not None:
        result = result.merge(
            pesticides_df[["country", "year", "pesticides_tonnes"]],
            on=["country", "year"],
            how="left"
        )
    
    # Merge rainfall data
    if rainfall_df is not None:
        result = result.merge(
            rainfall_df[["country", "year", "rainfall_mm"]],
            on=["country", "year"],
            how="left"
        )
    
    # Merge temperature data
    if temperature_df is not None:
        result = result.merge(
            temperature_df[["country", "year", "avg_temp"]],
            on=["country", "year"],
            how="left"
        )
    
    return result


def validate_consolidated_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the consolidated dataset.
    
    Args:
        df: Consolidated DataFrame.
        
    Returns:
        Tuple of (is_valid, list of errors).
    """
    errors = []
    
    # Check required columns
    required_cols = ["country", "crop", "year", "yield", "rainfall_mm", "pesticides_tonnes", "avg_temp"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for duplicate rows
    key_cols = ["country", "crop", "year"]
    if all(col in df.columns for col in key_cols):
        duplicates = df.duplicated(subset=key_cols, keep=False)
        if duplicates.any():
            errors.append(f"Found {duplicates.sum()} duplicate rows for (country, crop, year)")
    
    # Check target column
    if "yield" in df.columns:
        if df["yield"].isna().any():
            errors.append(f"Target column 'yield' has {df['yield'].isna().sum()} missing values")
        if not pd.api.types.is_numeric_dtype(df["yield"]):
            errors.append("Target column 'yield' is not numeric")
    
    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_fusion_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of the consolidated dataset.
    
    Args:
        df: Consolidated DataFrame.
        
    Returns:
        Dictionary with summary statistics.
    """
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "year_range": (int(df["year"].min()), int(df["year"].max())) if "year" in df.columns else None,
        "unique_countries": df["country"].nunique() if "country" in df.columns else None,
        "unique_crops": df["crop"].nunique() if "crop" in df.columns else None,
        "crop_list": df["crop"].unique().tolist() if "crop" in df.columns else None,
        "missing_values": df.isna().sum().to_dict(),
        "missing_percentage": (df.isna().sum() / len(df) * 100).round(2).to_dict(),
    }
    return summary


def save_fusion_summary(summary: Dict, output_path: Optional[Path] = None) -> None:
    """
    Save fusion summary to a text file.
    
    Args:
        summary: Summary dictionary.
        output_path: Output file path.
    """
    output_path = output_path or (PROCESSED_DATA_DIR / "fusion_summary.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DATA FUSION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total rows: {summary['total_rows']}\n")
        f.write(f"Total columns: {summary['total_columns']}\n")
        f.write(f"Columns: {', '.join(summary['columns'])}\n\n")
        
        if summary['year_range']:
            f.write(f"Year range: {summary['year_range'][0]} - {summary['year_range'][1]}\n")
        if summary['unique_countries']:
            f.write(f"Unique countries: {summary['unique_countries']}\n")
        if summary['unique_crops']:
            f.write(f"Unique crops: {summary['unique_crops']}\n")
        
        f.write("\nCrops included:\n")
        if summary['crop_list']:
            for crop in sorted(summary['crop_list']):
                f.write(f"  - {crop}\n")
        
        f.write("\nMissing values:\n")
        for col, count in summary['missing_values'].items():
            pct = summary['missing_percentage'][col]
            f.write(f"  {col}: {count} ({pct}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Fusion strategy:\n")
        f.write("- Join keys: (country, year)\n")
        f.write("- Crop is retained from yield dataset\n")
        f.write("- Context features (rainfall, pesticides, temp) joined on country+year\n")
        f.write("- Missing values imputed with median\n")
        f.write("=" * 60 + "\n")
    
    print(f"Saved fusion summary to {output_path}")
