"""
Main FastAPI application file for salary prediction API.
"""
import logging
from fastapi import FastAPI, Depends
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.model_service import ModelService
from app.routers import predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Data Science Salary Predictor API",
    description="API for predicting data science salaries based on job features",
    version="1.0.0",
)

# Include routers
app.include_router(predict.router)

# Add health check
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up the application...")
    # Initialize the model service
    ModelService.initialize()
    logger.info("Model loaded successfully")