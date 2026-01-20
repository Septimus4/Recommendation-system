"""
Script to retrain the model without rainfall_mm feature.

The rainfall data in the training set is a country-level constant (same value for all years),
making it confounded with country identity and useless for "what-if" scenarios.
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Updated feature definitions (NO RAINFALL)
NUMERIC_FEATURES = ['pesticides_tonnes', 'avg_temp']
CATEGORICAL_FEATURES = ['crop', 'country']
TARGET = 'yield'


def main():
    print("=" * 60)
    print("RETRAINING MODEL WITHOUT RAINFALL")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(DATA_PROCESSED / 'consolidated.csv')
    print(f"   Dataset shape: {df.shape}")
    
    # Prepare features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    print(f"\n2. Features used:")
    print(f"   Numeric: {NUMERIC_FEATURES}")
    print(f"   Categorical: {CATEGORICAL_FEATURES}")
    print(f"   Target: {TARGET}")
    
    # Train/test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create preprocessing pipeline
    print("\n4. Creating preprocessing pipeline...")
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
    
    # Create full pipeline with Random Forest
    print("\n5. Training Random Forest model...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    print("   Training complete!")
    
    # Evaluate
    print("\n6. Evaluating model...")
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n   Training Metrics:")
    print(f"     RMSE: {train_rmse:,.2f}")
    print(f"     MAE:  {train_mae:,.2f}")
    print(f"     R²:   {train_r2:.4f}")
    
    print(f"\n   Test Metrics:")
    print(f"     RMSE: {test_rmse:,.2f}")
    print(f"     MAE:  {test_mae:,.2f}")
    print(f"     R²:   {test_r2:.4f}")
    
    # Save model
    print("\n7. Saving model...")
    model_path = MODELS_DIR / 'model_pipeline.joblib'
    joblib.dump(pipeline, model_path)
    print(f"   Model saved to: {model_path}")
    
    # Save metadata
    print("\n8. Saving metadata...")
    supported_crops = sorted(df['crop'].unique().tolist())
    supported_countries = sorted(df['country'].unique().tolist())
    
    metadata = {
        "model_name": "Random Forest",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "supported_crops": supported_crops,
        "supported_countries": supported_countries,
        "trained_at": datetime.now().isoformat(),
        "note": "Model trained without rainfall_mm (removed due to data quality issues)"
    }
    
    metadata_path = MODELS_DIR / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE!")
    print("=" * 60)
    
    return test_r2, test_rmse


if __name__ == "__main__":
    main()
