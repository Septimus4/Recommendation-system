"""
Script to retrain the model without rainfall feature.

This script trains a Random Forest model using only:
- pesticides_tonnes
- avg_temp  
- crop (categorical)
- country (categorical)

Rainfall was removed because in the training data it's a country-level constant
(same value for all years), making it confounded with country identity.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "consolidated.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Feature configuration (without rainfall)
NUMERIC_FEATURES = ["pesticides_tonnes", "avg_temp"]
CATEGORICAL_FEATURES = ["crop", "country"]
TARGET = "yield"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load and prepare the dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    return df


def create_preprocessor():
    """Create preprocessing pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    
    return preprocessor


def train_model(X_train, y_train, preprocessor):
    """Train the Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "test_mae": mean_absolute_error(y_test, y_pred),
        "test_r2": r2_score(y_test, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"  RMSE: {metrics['test_rmse']:,.2f}")
    print(f"  MAE:  {metrics['test_mae']:,.2f}")
    print(f"  R²:   {metrics['test_r2']:.4f}")
    
    return metrics


def save_model(pipeline, metrics, X_train, X_test):
    """Save model and metadata."""
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save pipeline
    model_path = MODELS_DIR / "model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Get supported values from training data
    df = pd.read_csv(DATA_PATH)
    supported_crops = sorted(df["crop"].unique().tolist())
    supported_countries = sorted(df["country"].unique().tolist())
    
    # Create metadata
    metadata = {
        "model_name": "Random Forest",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "test_r2": metrics["test_r2"],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "supported_crops": supported_crops,
        "supported_countries": supported_countries,
        "trained_at": datetime.now().isoformat(),
        "note": "Rainfall removed - it was confounded with country identity in training data"
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    return model_path, metadata_path


def main():
    """Main training workflow."""
    print("=" * 60)
    print("RETRAINING MODEL WITHOUT RAINFALL FEATURE")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare features and target
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]
    
    print(f"\nFeatures used: {feature_cols}")
    print(f"Target: {TARGET}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    # Train model
    pipeline = train_model(X_train, y_train, preprocessor)
    
    # Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Save
    save_model(pipeline, metrics, X_train, X_test)
    
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE!")
    print("=" * 60)
    
    return pipeline, metrics


if __name__ == "__main__":
    main()
