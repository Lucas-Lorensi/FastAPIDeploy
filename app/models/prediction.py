"""
Pydantic models for the salary prediction API.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class PredictionFeatures(BaseModel):
    """
    Input features for salary prediction.
    
    Attributes:
        work_year: The year of the job data
        experience_level: Experience level code (EN=Entry, MI=Mid, SE=Senior, EX=Expert)
        employment_type: Employment type (FT=Full-time, PT=Part-time, CT=Contract, FL=Freelance)
        job_title: Job title for the data science role
        remote_ratio: Percentage of remote work (0, 50, 100)
        company_size: Company size code (S=Small, M=Medium, L=Large)
    """
    work_year: int = Field(..., description="Year of the job data (e.g., 2020, 2021)")
    experience_level: str = Field(..., description="Experience level (EN, MI, SE, EX)")
    employment_type: str = Field(..., description="Employment type (FT, PT, CT, FL)")
    job_title: str = Field(..., description="Job title")
    remote_ratio: int = Field(..., description="Remote work percentage (0, 50, 100)")
    company_size: str = Field(..., description="Company size (S, M, L)")

    class Config:
        schema_extra = {
            "example": {
                "work_year": 2022,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_title": "Data Scientist",
                "remote_ratio": 100,
                "company_size": "M"
            }
        }

class SalaryPrediction(BaseModel):
    """
    Salary prediction response.
    
    Attributes:
        salary_usd: Predicted salary in USD
        prediction_metadata: Additional metadata about the prediction
    """
    salary_usd: float = Field(..., description="Predicted salary in USD")
    prediction_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the prediction"
    )

    class Config:
        schema_extra = {
            "example": {
                "salary_usd": 120000.0,
                "prediction_metadata": {
                    "model_version": "1.0.0",
                    "confidence_score": 0.85
                }
            }
        }