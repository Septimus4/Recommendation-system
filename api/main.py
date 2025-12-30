"""
FastAPI application for Crop Yield Prediction and Recommendation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Optional

from schemas import (
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
    CropRecommendation,
    HealthResponse,
    ModelInfoResponse
)
from model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model loader
model_loader: Optional[ModelLoader] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    global model_loader
    
    # Startup
    logger.info("Loading model...")
    try:
        model_loader = ModelLoader()
        model_loader.load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Crop Yield Prediction API",
    description="API for predicting crop yields and recommending optimal crops based on environmental conditions.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint returning API health status."""
    return HealthResponse(
        status="healthy",
        message="Crop Yield Prediction API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model_loader is None or not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return HealthResponse(
        status="healthy",
        message="API is healthy and model is loaded"
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model."""
    if model_loader is None or not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_version=model_loader.model_version,
        supported_crops=model_loader.supported_crops,
        supported_countries=model_loader.supported_countries[:50],  # Limit for response size
        features=model_loader.feature_names
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_yield(request: PredictionRequest):
    """
    Predict crop yield for a specific crop and environmental conditions.
    
    - **crop**: Name of the crop (e.g., "Wheat", "Maize", "Rice, paddy")
    - **country**: Country name (e.g., "India", "United States")
    - **rainfall_mm**: Average annual rainfall in millimeters
    - **pesticides_tonnes**: Pesticides usage in tonnes
    - **avg_temp**: Average temperature in Celsius
    """
    if model_loader is None or not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate crop
    if request.crop not in model_loader.supported_crops:
        raise HTTPException(
            status_code=400,
            detail=f"Crop '{request.crop}' is not supported. Supported crops: {model_loader.supported_crops}"
        )
    
    # Validate input ranges
    if request.rainfall_mm < 0:
        raise HTTPException(status_code=400, detail="rainfall_mm must be non-negative")
    if request.pesticides_tonnes < 0:
        raise HTTPException(status_code=400, detail="pesticides_tonnes must be non-negative")
    if request.avg_temp < -50 or request.avg_temp > 60:
        raise HTTPException(status_code=400, detail="avg_temp must be between -50 and 60")
    
    try:
        predicted_yield = model_loader.predict(
            crop=request.crop,
            country=request.country,
            rainfall_mm=request.rainfall_mm,
            pesticides_tonnes=request.pesticides_tonnes,
            avg_temp=request.avg_temp
        )
        
        return PredictionResponse(
            crop=request.crop,
            predicted_yield=round(predicted_yield, 2),
            yield_unit="hg/ha",
            model_version=model_loader.model_version
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_crops(request: RecommendationRequest):
    """
    Recommend crops based on environmental conditions.
    
    Returns a ranked list of crops sorted by predicted yield.
    
    - **country**: Country name (e.g., "India", "United States")
    - **rainfall_mm**: Average annual rainfall in millimeters
    - **pesticides_tonnes**: Pesticides usage in tonnes
    - **avg_temp**: Average temperature in Celsius
    - **top_n**: Number of top recommendations to return (optional, default: all)
    """
    if model_loader is None or not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input ranges
    if request.rainfall_mm < 0:
        raise HTTPException(status_code=400, detail="rainfall_mm must be non-negative")
    if request.pesticides_tonnes < 0:
        raise HTTPException(status_code=400, detail="pesticides_tonnes must be non-negative")
    if request.avg_temp < -50 or request.avg_temp > 60:
        raise HTTPException(status_code=400, detail="avg_temp must be between -50 and 60")
    
    try:
        recommendations = model_loader.recommend(
            country=request.country,
            rainfall_mm=request.rainfall_mm,
            pesticides_tonnes=request.pesticides_tonnes,
            avg_temp=request.avg_temp,
            top_n=request.top_n
        )
        
        # Convert to response format
        crop_recommendations = [
            CropRecommendation(
                rank=rec["rank"],
                crop=rec["crop"],
                predicted_yield=round(rec["predicted_yield"], 2),
                yield_unit=rec["yield_unit"]
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            recommendations=crop_recommendations,
            context={
                "country": request.country,
                "rainfall_mm": request.rainfall_mm,
                "pesticides_tonnes": request.pesticides_tonnes,
                "avg_temp": request.avg_temp
            },
            model_version=model_loader.model_version
        )
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
