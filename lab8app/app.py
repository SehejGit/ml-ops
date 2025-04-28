# app.py
from fastapi import FastAPI, HTTPException
import uvicorn
import joblib
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'This is a model for classifying Reddit comments'}

class request_body(BaseModel):
    reddit_comment: str

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    try:
        logger.info("Loading model from joblib file...")
        # Load the model from the joblib file
        model_pipeline = joblib.load("reddit_model_pipeline.joblib")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data: request_body):
    try:
        X = [data.reddit_comment]
        predictions = model_pipeline.predict_proba(X)
        # Convert numpy array to list for JSON serialization
        predictions_list = predictions.tolist()
        return {'Predictions': predictions_list}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    # Change port to 8001 to avoid conflicts
    uvicorn.run(app, host="0.0.0.0", port=8001)