# Development Guide

Guide for developers contributing to the project.

## Project Structure

```
Recommendation-system/
├── api/                    # FastAPI backend
│   ├── main.py            # API endpoints
│   ├── schemas.py         # Pydantic models
│   ├── model_loader.py    # Model loading
│   └── tests/             # API tests
├── app/                    # Streamlit frontend
│   └── app.py             # UI application
├── src/                    # Source code modules
│   ├── config/            # Configuration
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   └── models/            # ML utilities
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── data/                   # Data files
├── models/                 # Model artifacts
└── docs/                   # Documentation
```

---

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/Septimus4/Recommendation-system.git
cd Recommendation-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Dev Dependencies

```bash
pip install pytest black isort flake8
```

---

## Running Tests

### All Tests

```bash
pytest tests/ api/tests/ -v
```

### Specific Test File

```bash
pytest tests/test_data.py -v
```

### With Coverage

```bash
pytest --cov=src --cov=api tests/ -v
```

---

## Code Style

### Formatting

```bash
# Format code
black src/ api/ app/ tests/

# Sort imports
isort src/ api/ app/ tests/
```

### Linting

```bash
flake8 src/ api/ app/ tests/ --max-line-length=120
```

---

## Adding New Features

### Adding a New Endpoint

1. Define schema in `api/schemas.py`:
```python
class NewRequest(BaseModel):
    field: str = Field(..., description="Description")
```

2. Add endpoint in `api/main.py`:
```python
@app.post("/new-endpoint")
async def new_endpoint(request: NewRequest):
    # Implementation
    return {"result": "value"}
```

3. Add tests in `api/tests/test_api.py`

### Adding a New Feature

1. Add feature logic in `src/features/engineering.py`
2. Update preprocessor if needed
3. Retrain model
4. Update API schemas if needed
5. Add tests

---

## Testing Guidelines

### Test Structure

```python
class TestFeatureName:
    """Tests for feature name."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = function_under_test(input)
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Test Naming

- `test_<functionality>_<scenario>`
- Example: `test_prediction_valid_input`

---

## Git Workflow

### Branches

- `main`: Production-ready code
- `feature/*`: New features
- `fix/*`: Bug fixes

### Commit Messages

```
type: short description

- Detail 1
- Detail 2
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`

### Pull Request

1. Create feature branch
2. Make changes
3. Run tests
4. Push and create PR
5. Request review

---

## API Development

### Running Locally

```bash
cd api
uvicorn main:app --reload --port 8000
```

### API Documentation

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Testing Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"crop": "Wheat", "country": "India", "rainfall_mm": 1000, "pesticides_tonnes": 5000, "avg_temp": 20}'
```

---

## Model Development

### Retraining

```bash
jupyter notebook notebooks/02_modeling_mlflow.ipynb
```

### Updating Model

1. Train new model
2. Save to `models/model_pipeline.joblib`
3. Update `models/model_metadata.json`
4. Test with API

### MLflow

```bash
mlflow ui --backend-store-uri mlruns
```

---

## Streamlit Development

### Running Locally

```bash
cd app
streamlit run app.py
```

### Hot Reload

Streamlit auto-reloads on file changes.

---

## Docker

### Build API Image

```bash
cd api
docker build -t crop-yield-api .
```

### Run Container

```bash
docker run -p 8000:8000 crop-yield-api
```

---

## CI/CD

### GitHub Actions

The CI pipeline (`.github/workflows/ci.yml`) runs:
1. Code quality checks (flake8, black)
2. Unit tests
3. Docker build

### Running CI Locally

```bash
# Lint
flake8 api/ app/ src/ --count --select=E9,F63,F7,F82 --show-source

# Tests
pytest tests/ api/tests/ -v

# Docker build
cd api && docker build -t crop-yield-api .
```

---

## Common Tasks

### Add New Crop

1. Add to `src/config/constants.py`:
```python
SUPPORTED_CROPS = [..., "NewCrop"]
```

2. Retrain model with new crop data
3. Update metadata

### Add New Feature

1. Add to preprocessing pipeline
2. Update `NUMERIC_FEATURES` or `CATEGORICAL_FEATURES`
3. Retrain model
4. Update API schemas

### Update Documentation

1. Edit files in `docs/`
2. Update README.md if needed
3. Commit changes

---

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [scikit-learn Docs](https://scikit-learn.org/stable/)
- [pytest Docs](https://docs.pytest.org/)
