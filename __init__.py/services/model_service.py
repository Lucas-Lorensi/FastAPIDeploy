"""
Service for loading and using the trained model.
"""
import os
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List
from app.models.prediction import PredictionFeatures

logger = logging.getLogger(__name__)

class ModelService:
    """
    Service for managing the ML model for salary prediction.
    
    This is implemented as a singleton to ensure the model is loaded only once.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def initialize(cls):
        """Initialize the model service"""
        cls.get_instance()
    
    def __init__(self):
        """Initialize the model service"""
        self.model = None
        self.model_path = os.getenv("MODEL_PATH", "model/xgboost.sav")
        self.model_version = "1.0.0"
        self.features_used = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            # For our specific model, let's define the features we expect
            # These should match the column names expected by the model
            self.features_used = [
                "work_year",
                "experience_level_encoded",
                "company_size_encoded",
                "remote_ratio",
                "employment_type_PT",
                "job_title_Data Engineer",
                "job_title_Data Manager",
                "job_title_Data Scientist",
                "job_title_Machine Learning Engineer"
            ]
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _preprocess_features(self, features: PredictionFeatures) -> pd.DataFrame:
        """
        Preprocess raw features into the format expected by the model.
        
        Args:
            features: Raw prediction features
            
        Returns:
            DataFrame: Preprocessed features ready for model prediction
        """
        # Create a DataFrame with default values (all 0)
        input_data = pd.DataFrame(np.zeros((1, len(self.features_used))), 
                                 columns=self.features_used)
        
        # Fill in the base features
        input_data["work_year"] = features.work_year
        input_data["remote_ratio"] = features.remote_ratio
        
        # Encode experience level
        exp_level_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
        input_data["experience_level_encoded"] = exp_level_map.get(features.experience_level, 0)
        
        # Encode company size
        company_size_map = {'S': 0, 'M': 1, 'L': 2}
        input_data["company_size_encoded"] = company_size_map.get(features.company_size, 0)
        
        # Handle employment type (one-hot encoded)
        if features.employment_type == "PT":
            input_data["employment_type_PT"] = 1
        
        # Handle job title (one-hot encoded)
        job_title_column = f"job_title_{features.job_title.replace(' ', '_')}"
        if job_title_column in input_data.columns:
            input_data[job_title_column] = 1
        
        return input_data
    
    def predict(self, features: PredictionFeatures) -> Dict[str, Any]:
        """
        Make a salary prediction based on input features.
        
        Args:
            features: Input features for prediction
            
        Returns:
            Dict: Prediction result with salary in USD
        """
        try:
            # Preprocess features
            input_data = self._preprocess_features(features)
            
            # Make prediction
            prediction = self.model.predict(input_data)[0]
            
            # Round to nearest integer for cleaner display
            rounded_prediction = round(prediction)
            
            return {"salary_usd": rounded_prediction}
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")