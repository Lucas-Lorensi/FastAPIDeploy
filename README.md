# Data Science Salary Predictor

A machine learning application that predicts data science salaries based on various job features such as experience level, job title, company size, and work arrangement.

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/lucas-lorensi/FastAPIDeploy.git
   cd salary-predictor
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the training notebook to generate the model (if not already done):
   ```
   jupyter notebook notebooks/model_training.ipynb
   ```

### Running the API

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation will be available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Using Docker

1. Build the Docker image:
   ```
   docker build -t salary-predictor .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 salary-predictor
   ```

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /api/v1/predict`: Predict salary based on job features

Example request:
```json
{
  "work_year": 2022,
  "experience_level": "SE",
  "employment_type": "FT",
  "job_title": "Data Scientist",
  "remote_ratio": 100,
  "company_size": "M"
}
```

Example response:
```json
{
  "salary_usd": 120000,
  "prediction_metadata": {
    "model_version": "1.0.0",
    "features_used": [...]
  }
}
```

## Testing

Run tests with:

```
pytest
```

## Model Training

The model was trained using XGBoost on the [Data Science Job Salaries dataset](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) from Kaggle.

See `notebooks/model_training.ipynb` for the training process.