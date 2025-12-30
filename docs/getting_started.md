# Getting Started

This guide will help you set up and run the Crop Yield Prediction system.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Septimus4/Recommendation-system.git
cd Recommendation-system
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install separately
pip install -r api/requirements.txt      # API only
pip install -r app/requirements.txt      # Streamlit only
```

## Quick Start

### Step 1: Prepare Data

The consolidated dataset should already exist at `data/processed/consolidated.csv`. If not, run:

```bash
jupyter notebook notebooks/01_eda_and_fusion.ipynb
```

### Step 2: Train Model (Optional)

A pre-trained model exists at `models/model_pipeline.joblib`. To retrain:

```bash
jupyter notebook notebooks/02_modeling_mlflow.ipynb
```

### Step 3: Start the API

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Step 4: Start the Streamlit App

In a new terminal:

```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Verify Installation

### Test API Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "message": "API is healthy and model is loaded"}
```

### Test Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"crop": "Wheat", "country": "India", "rainfall_mm": 1000, "pesticides_tonnes": 5000, "avg_temp": 20}'
```

## Run Tests

```bash
pytest tests/ -v
```

## Common Issues

### Model Not Found

If you see "Model not found" error:
1. Ensure `models/model_pipeline.joblib` exists
2. Run the modeling notebook to train a new model

### API Connection Error

If Streamlit can't connect to API:
1. Ensure API is running on port 8000
2. Check the API URL in the Streamlit sidebar

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

- Read the [API Reference](api_reference.md) for endpoint details
- See the [User Guide](user_guide.md) for application usage
- Check [Model Documentation](model_documentation.md) for ML details
