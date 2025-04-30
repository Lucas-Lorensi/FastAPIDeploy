"""
API endpoints for salary predictions.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.models.prediction import PredictionFeatures, SalaryPrediction
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["predictions"],
    responses={404: {"description": "Not found"}},
)

@router.post("/predict", response_model=SalaryPrediction)
async def predict_salary(features: PredictionFeatures):
    """
    Predict salary based on job features.
    
    Args:
        features: Input features for prediction
        
    Returns:
        SalaryPrediction: Predicted salary and metadata
    """
    try:
        # Get model service instance
        model_service = ModelService.get_instance()
        
        # Make prediction
        prediction_result = model_service.predict(features)
        
        return SalaryPrediction(
            salary_usd=prediction_result["salary_usd"],
            prediction_metadata={
                "model_version": model_service.model_version,
                "features_used": model_service.features_used
            }
        )
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making prediction")