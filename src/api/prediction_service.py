"""
ML Model Prediction Service
FastAPI endpoint for serving predictions using TorchScript
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(
    title="Bearish Alpha Bot ML API",
    version="2.0.0",
    description="Production ML inference using TorchScript"
)

# Global model storage
MODEL = None
MODEL_METADATA = None


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    features: List[float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0.0] * 82  # 82 features
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    prediction: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    model_loaded: bool
    model_accuracy: float
    export_format: str
    timestamp: str


def load_model():
    """
    ✅ Load TorchScript model on startup
    
    No class definition needed!
    TorchScript is self-contained.
    """
    global MODEL, MODEL_METADATA
    
    # ✅ Load TorchScript model (.ptc)
    model_path = Path('data/models/final/lstm_final_latest.ptc')
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run training workflow first."
        )
    
    print(f"🔄 Loading TorchScript model from: {model_path}")
    
    try:
        # ✅ Load compiled model - NO IMPORT NEEDED!
        MODEL = torch.jit.load(str(model_path))
        MODEL.eval()  # Always set to eval mode
        
        print("✅ TorchScript model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Load metadata separately
    metadata_path = Path('logs/final_training/metadata_latest.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            MODEL_METADATA = data['metadata']
        
        print(f"✅ Metadata loaded!")
        print(f"   Test Accuracy: {MODEL_METADATA['final_test_accuracy']:.4f}")
        print(f"   Export Format: {MODEL_METADATA.get('export_format', 'unknown')}")
    else:
        print("⚠️  Metadata not found, using defaults")
        MODEL_METADATA = {
            'final_test_accuracy': 0.0,
            'timestamp': datetime.utcnow().isoformat(),
            'export_format': 'torchscript'
        }


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_accuracy": MODEL_METADATA.get('final_test_accuracy', 0.0) if MODEL_METADATA else 0.0,
        "export_format": MODEL_METADATA.get('export_format', 'unknown') if MODEL_METADATA else 'unknown',
        "timestamp": MODEL_METADATA.get('timestamp', '') if MODEL_METADATA else ''
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_accuracy": MODEL_METADATA.get('final_test_accuracy', 0.0),
        "export_format": MODEL_METADATA.get('export_format', 'torchscript'),
        "timestamp": MODEL_METADATA.get('timestamp', '')
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction from features
    
    ✅ Uses TorchScript model - fast and reliable
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate features
    expected_features = MODEL_METADATA.get('num_features', 82)
    if len(request.features) != expected_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_features} features, got {len(request.features)}"
        )
    
    # Make prediction
    try:
        with torch.no_grad():
            # Convert to tensor: shape (1, features)
            features = torch.FloatTensor([request.features])
            
            # Forward pass
            outputs = MODEL(features)
            
            # Get probabilities and prediction
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Class names
        class_names = ['Bullish', 'Neutral', 'Bearish']
        
        return {
            "prediction": prediction,
            "class_name": class_names[prediction],
            "confidence": confidence,
            "probabilities": {
                class_names[i]: float(probabilities[0, i])
                for i in range(len(class_names))
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get model metrics"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not MODEL_METADATA:
        raise HTTPException(status_code=500, detail="Metadata not available")
    
    return {
        "test_accuracy": MODEL_METADATA.get('final_test_accuracy', 0.0),
        "tuning_cv_score": MODEL_METADATA.get('tuning_cv_score', 0.0),
        "tuning_holdout_score": MODEL_METADATA.get('tuning_holdout_score', 0.0),
        "training_samples": MODEL_METADATA.get('training_samples', 0),
        "test_samples": MODEL_METADATA.get('test_samples', 0),
        "num_features": MODEL_METADATA.get('num_features', 0),
        "export_format": MODEL_METADATA.get('export_format', 'unknown'),
        "timestamp": MODEL_METADATA.get('timestamp', '')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
