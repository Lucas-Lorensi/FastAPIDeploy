"""
Tests for the prediction endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    """Test the prediction endpoint with valid data"""
    test_data = {
        "work_year": 2022,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "remote_ratio": 100,
        "company_size": "M"
    }
    
    response = client.post("/api/v1/predict", json=test_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "salary_usd" in result
    assert isinstance(result["salary_usd"], (int, float))
    
    assert "prediction_metadata" in result
    assert "model_version" in result["prediction_metadata"]
    assert "features_used" in result["prediction_metadata"]

def test_predict_endpoint_invalid_data():
    """Test the prediction endpoint with invalid data"""
    # Missing required fields
    test_data = {
        "experience_level": "SE",
        "employment_type": "FT"
    }
    
    response = client.post("/api/v1/predict", json=test_data)
    assert response.status_code == 422  # Validation error