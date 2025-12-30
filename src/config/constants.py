"""
Constants and configuration values for the crop yield prediction system.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Data files
YIELD_RAW_FILE = RAW_DATA_DIR / "yield.csv"
YIELD_DF_RAW_FILE = RAW_DATA_DIR / "yield_df.csv"
PESTICIDES_RAW_FILE = RAW_DATA_DIR / "pesticides.csv"
RAINFALL_RAW_FILE = RAW_DATA_DIR / "rainfall.csv"
TEMP_RAW_FILE = RAW_DATA_DIR / "temp.csv"
CONSOLIDATED_FILE = PROCESSED_DATA_DIR / "consolidated.csv"

# Column mappings - standardized column names
COLUMN_MAPPING = {
    "Area": "country",
    "Item": "crop",
    "Year": "year",
    "hg/ha_yield": "yield",
    "average_rain_fall_mm_per_year": "rainfall_mm",
    "pesticides_tonnes": "pesticides_tonnes",
    "avg_temp": "avg_temp",
}

# Supported crops (after data cleaning)
SUPPORTED_CROPS = [
    "Maize",
    "Potatoes",
    "Rice, paddy",
    "Sorghum",
    "Soybeans",
    "Wheat",
    "Cassava",
    "Sweet potatoes",
    "Plantains and others",
    "Yams",
]

# Feature columns for modeling
NUMERIC_FEATURES = ["rainfall_mm", "pesticides_tonnes", "avg_temp"]
CATEGORICAL_FEATURES = ["crop", "country"]
TARGET_COLUMN = "yield"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Validation ranges for input features
FEATURE_RANGES = {
    "rainfall_mm": (0, 5000),
    "pesticides_tonnes": (0, 500000),
    "avg_temp": (-10, 50),
}

# Country name standardization mapping
COUNTRY_MAPPING = {
    "United States of America": "United States",
    "USA": "United States",
    "UK": "United Kingdom",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Russian Federation": "Russia",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "Viet Nam": "Vietnam",
    "Iran (Islamic Republic of)": "Iran",
    "Syrian Arab Republic": "Syria",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Bolivia (Plurinational State of)": "Bolivia",
    "United Republic of Tanzania": "Tanzania",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
}

# MLflow configuration
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")
MLFLOW_EXPERIMENT_NAME = "crop_yield_prediction"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_ARTIFACT_PATH = MODELS_DIR / "model_pipeline.joblib"
