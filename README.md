# Crop Yield Prediction & Recommendation System

[![CI/CD Pipeline](https://github.com/Septimus4/Recommendation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Septimus4/Recommendation-system/actions/workflows/ci.yml)

A machine learning-powered web application that helps farmers predict crop yields and get recommendations for optimal crops based on environmental conditions.

## 🌾 Overview

This system provides:
- **Yield Prediction**: Predict expected yield for a specific crop given environmental conditions
- **Crop Recommendation**: Get ranked recommendations for the best crops to grow based on your conditions

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│   ML Model      │
│   Frontend      │     │    Backend      │     │   (scikit-learn)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
Recommendation-system/
├── data/
│   ├── raw/              # Original datasets
│   ├── interim/          # Temporary processing files
│   └── processed/        # Cleaned, consolidated data
├── notebooks/
│   ├── 01_eda_and_fusion.ipynb    # Data exploration & cleaning
│   └── 02_modeling_mlflow.ipynb   # Model training & evaluation
├── src/
│   ├── config/           # Configuration and constants
│   ├── data/             # Data loading and cleaning
│   ├── features/         # Feature engineering
│   └── models/           # Model training and inference
├── api/
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Request/response schemas
│   ├── model_loader.py   # Model loading utilities
│   ├── Dockerfile        # API containerization
│   └── tests/            # API unit tests
├── app/
│   ├── app.py            # Streamlit application
│   └── requirements.txt  # Frontend dependencies
├── models/               # Saved model artifacts
├── reports/
│   ├── figures/          # Generated plots
│   └── screenshots/      # UI screenshots
├── mlruns/               # MLflow tracking data
└── .github/workflows/    # CI/CD pipeline
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Septimus4/Recommendation-system.git
cd Recommendation-system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
# For API
pip install -r api/requirements.txt

# For Streamlit app
pip install -r app/requirements.txt

# For notebooks and development
pip install pandas numpy scikit-learn matplotlib seaborn mlflow jupyter
```

### Data Preparation

1. **Run EDA and data fusion notebook**
```bash
cd notebooks
jupyter notebook 01_eda_and_fusion.ipynb
```
This will:
- Explore the raw datasets
- Clean and standardize the data
- Generate `data/processed/consolidated.csv`

### Model Training

2. **Run modeling notebook**
```bash
jupyter notebook 02_modeling_mlflow.ipynb
```
This will:
- Perform PCA analysis
- Train multiple models
- Log experiments to MLflow
- Save the best model to `models/model_pipeline.joblib`

### Running the Application

3. **Start the API**
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
API will be available at `http://localhost:8000`

4. **Start the Streamlit app** (in a new terminal)
```bash
cd app
streamlit run app.py
```
App will be available at `http://localhost:8501`

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Yield Prediction
```bash
POST /predict
{
    "crop": "Wheat",
    "country": "India",
    "rainfall_mm": 1000,
    "pesticides_tonnes": 5000,
    "avg_temp": 20
}
```

### Crop Recommendation
```bash
POST /recommend
{
    "country": "India",
    "rainfall_mm": 1000,
    "pesticides_tonnes": 5000,
    "avg_temp": 20,
    "top_n": 5
}
```

## 🔬 MLflow Tracking

View experiment results:
```bash
mlflow ui --backend-store-uri mlruns
```
Open `http://localhost:5000` to see:
- Model comparison
- Hyperparameters
- Metrics (RMSE, MAE, R²)
- Artifacts

## 🐳 Docker

Build and run the API:
```bash
cd api
docker build -t crop-yield-api .
docker run -p 8000:8000 crop-yield-api
```

## 🧪 Testing

Run API tests:
```bash
cd api
pytest tests/ -v
```

## 📈 Model Performance

| Model | Test RMSE | Test MAE | Test R² |
|-------|-----------|----------|---------|
| Baseline (Mean) | - | - | - |
| Ridge Regression | - | - | - |
| Random Forest | - | - | - |
| Gradient Boosting | - | - | - |

*Run the modeling notebook to see actual metrics*

## 📋 Features Used

- **Numeric Features**:
  - `rainfall_mm`: Average annual rainfall
  - `pesticides_tonnes`: Pesticides usage
  - `avg_temp`: Average temperature

- **Categorical Features**:
  - `crop`: Type of crop
  - `country`: Country/region

## 🌍 Supported Crops

- Maize
- Potatoes
- Rice, paddy
- Sorghum
- Soybeans
- Wheat
- Cassava
- Sweet potatoes
- Plantains and others
- Yams

## ⚠️ Limitations

- Predictions are based on historical data and may not account for:
  - Soil quality variations
  - Extreme weather events
  - Local farming practices
  - Market conditions
- Model is trained on country-level data, not local/regional
- Pesticide values are proxies for agricultural input intensity

## 🔮 Future Improvements

- Add soil type as a feature
- Include real-time weather data
- Add economic/price data for profit optimization
- Regional/local level predictions
- Time series forecasting

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) (when running)
- [Data Fusion Summary](data/processed/fusion_summary.txt)
- [Business Report](reports/business_report.pdf)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- School Project Team

## 🙏 Acknowledgments

- FAO for crop yield data
- Climate data providers
- scikit-learn and MLflow communities
