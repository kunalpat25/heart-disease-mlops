"""
Heart Disease Prediction API
FastAPI application for serving the ML model.
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['result']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction requests',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

# Feature names (must match training order)
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Global model and scaler
model = None
scaler = None


def load_model_and_scaler():
    """Load the model and scaler from disk."""
    global model, scaler
    
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    # Try multiple possible paths
    possible_model_paths = [
        os.path.join(models_dir, 'best_model_random_forest.joblib'),
        os.path.join(base_dir, '..', 'models', 'best_model_random_forest.joblib'),
        '/app/models/best_model_random_forest.joblib'
    ]
    
    possible_scaler_paths = [
        os.path.join(models_dir, 'scaler.joblib'),
        os.path.join(base_dir, '..', 'models', 'scaler.joblib'),
        '/app/models/scaler.joblib'
    ]
    
    model_path = None
    scaler_path = None
    
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    for path in possible_scaler_paths:
        if os.path.exists(path):
            scaler_path = path
            break
    
    if model_path is None or scaler_path is None:
        logger.error("Model or scaler file not found!")
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    logger.info("Model and scaler loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Heart Disease Prediction API...")
    try:
        load_model_and_scaler()
    except FileNotFoundError as e:
        logger.warning(f"Model not loaded: {e}")
    yield
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered API for predicting heart disease risk based on patient health data.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class HeartDiseaseInput(BaseModel):
    """Input schema for heart disease prediction."""
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction."""
    prediction: int = Field(..., description="Predicted class (0 = No disease, 1 = Disease)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability_no_disease: float = Field(..., description="Probability of no heart disease")
    probability_disease: float = Field(..., description="Probability of heart disease")
    confidence: float = Field(..., description="Confidence of the prediction")
    model_version: str = Field(default="1.0.0", description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    timestamp: str


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time:.4f}s"
    )
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_heart_disease(input_data: HeartDiseaseInput):
    """
    Predict heart disease risk based on patient health data.
    
    Returns prediction with probability scores and confidence.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    start_time = time.time()
    
    try:
        # Extract features in correct order
        features = np.array([[
            input_data.age,
            input_data.sex,
            input_data.cp,
            input_data.trestbps,
            input_data.chol,
            input_data.fbs,
            input_data.restecg,
            input_data.thalach,
            input_data.exang,
            input_data.oldpeak,
            input_data.slope,
            input_data.ca,
            input_data.thal
        ]])
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Log prediction
        prediction_label = "Heart Disease" if prediction == 1 else "No Heart Disease"
        logger.info(f"Prediction made: {prediction_label} (confidence: {max(probabilities):.4f})")
        
        # Update metrics
        PREDICTIONS_TOTAL.labels(result=prediction_label).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            probability_no_disease=float(probabilities[0]),
            probability_disease=float(probabilities[1]),
            confidence=float(max(probabilities)),
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
