"""
Simple script to run the FastAPI application.
Place this file in your project root directory.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)