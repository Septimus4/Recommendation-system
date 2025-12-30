# Documentation Index

Welcome to the Crop Yield Prediction & Recommendation System documentation.

## 📚 Contents

| Document | Description |
|----------|-------------|
| [Getting Started](getting_started.md) | Installation and quick start guide |
| [API Reference](api_reference.md) | REST API endpoints documentation |
| [Data Guide](data_guide.md) | Dataset information and processing |
| [Model Documentation](model_documentation.md) | ML model details and performance |
| [User Guide](user_guide.md) | How to use the Streamlit application |
| [Development Guide](development_guide.md) | Contributing and development setup |

## 🌾 Project Overview

This system helps farmers:
- **Predict** expected crop yields based on environmental conditions
- **Recommend** optimal crops for their specific conditions

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI   │────▶│  ML Model   │
│  Frontend   │     │   Backend   │     │  (sklearn)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 🔗 Quick Links

- [GitHub Repository](https://github.com/Septimus4/Recommendation-system)
- [API Documentation](http://localhost:8000/docs) (when running)
- [Streamlit App](http://localhost:8501) (when running)
