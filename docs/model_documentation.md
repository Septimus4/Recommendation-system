# Model Documentation

Technical documentation for the machine learning model.

## Model Overview

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Task | Regression (yield prediction) |
| Target | Crop yield (hg/ha) |
| Framework | scikit-learn |

## Performance Metrics

| Metric | Value |
|--------|-------|
| Test R² | 0.8985 |
| Test RMSE | 27,847 hg/ha |
| Test MAE | 15,214 hg/ha |

## Model Comparison

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| **Random Forest** | **0.899** | **27,848** | **15,214** |
| HistGradientBoosting | 0.854 | 33,436 | 20,126 |
| Ridge Regression | 0.686 | 48,990 | 32,776 |

---

## Features

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| rainfall_mm | Numeric | Annual rainfall |
| pesticides_tonnes | Numeric | Pesticide usage |
| avg_temp | Numeric | Average temperature |
| crop | Categorical | Crop type (10 categories) |
| country | Categorical | Country (101 categories) |

### Feature Preprocessing

```
Numeric Features:
  ├── SimpleImputer (median)
  └── StandardScaler

Categorical Features:
  ├── SimpleImputer (constant="unknown")
  └── OneHotEncoder (handle_unknown="ignore")
```

---

## Training Details

### Data Split

- **Method**: Time-based split
- **Training**: Years ≤ 2010 (82%)
- **Testing**: Years > 2010 (18%)
- **Training samples**: 10,775
- **Test samples**: 2,355

### Hyperparameters

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
```

---

## Pipeline Structure

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', numeric_pipeline, ['rainfall_mm', 'pesticides_tonnes', 'avg_temp']),
        ('cat', categorical_pipeline, ['crop', 'country'])
    ])),
    ('model', RandomForestRegressor())
])
```

---

## Feature Importance

Top features by importance:

1. **crop_Potatoes** - High-yield crop indicator
2. **crop_Cassava** - High-yield crop indicator
3. **country_Netherlands** - High-productivity region
4. **pesticides_tonnes** - Input intensity proxy
5. **avg_temp** - Climate factor

---

## Model Files

| File | Description |
|------|-------------|
| `models/model_pipeline.joblib` | Serialized sklearn pipeline |
| `models/model_metadata.json` | Model metadata and supported values |

### Loading the Model

```python
import joblib

# Load model
model = joblib.load('models/model_pipeline.joblib')

# Make prediction
import pandas as pd
input_df = pd.DataFrame([{
    'rainfall_mm': 1000,
    'pesticides_tonnes': 5000,
    'avg_temp': 20,
    'crop': 'Wheat',
    'country': 'India'
}])

prediction = model.predict(input_df)
print(f"Predicted yield: {prediction[0]:.0f} hg/ha")
```

---

## Inference Logic

### Prediction

```python
def predict(crop, country, rainfall_mm, pesticides_tonnes, avg_temp):
    input_df = pd.DataFrame([{
        'rainfall_mm': rainfall_mm,
        'pesticides_tonnes': pesticides_tonnes,
        'avg_temp': avg_temp,
        'crop': crop,
        'country': country
    }])
    return model.predict(input_df)[0]
```

### Recommendation

```python
def recommend(country, rainfall_mm, pesticides_tonnes, avg_temp):
    results = []
    for crop in SUPPORTED_CROPS:
        yield_pred = predict(crop, country, rainfall_mm, pesticides_tonnes, avg_temp)
        results.append({'crop': crop, 'yield': yield_pred})
    
    # Sort by yield descending
    return sorted(results, key=lambda x: x['yield'], reverse=True)
```

---

## MLflow Tracking

Experiments are logged to `mlruns/`:

```bash
# View experiments
mlflow ui --backend-store-uri mlruns
```

### Logged Information

- Model type and hyperparameters
- Train/test metrics (RMSE, MAE, R²)
- Model artifacts

---

## Limitations

1. **Country-level granularity**: Cannot predict for regions within countries
2. **Temporal cutoff**: Trained on data up to 2013
3. **Feature scope**: Missing soil, irrigation, variety data
4. **Pesticide proxy**: National totals, not crop-specific application

## Future Improvements

- Add soil type features
- Include irrigation data
- Use more recent data
- Regional-level predictions
- Time series forecasting
