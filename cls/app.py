from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from naive_bayes_classifier import NaiveBayesClassifier
import requests

import logging

logger = logging.getLogger("uvicorn.error")

MODEL_TRAINER_URL = "http://model-trainer:8510"


app = FastAPI()

model_handler = NaiveBayesClassifier()

class PredictRequest(BaseModel) :
    features: Dict

@app.on_event("startup")
def fetch_model_on_startup():
    """
     On startup, this service actively fetches the latest trained model
     from the trainer service to be ready for predictions.
     """
    try:
        response = requests.get(f"{MODEL_TRAINER_URL}/export_model/")
        if response.status_code != 200:
            raise Exception(f"Failed to fetch model: {response.text}")

        model_dict = response.json()
        model_handler.load_model(model_dict)
        print("Model loaded successfully on startup")
    except Exception as e:
        print(f"Error loading model on startup: {e}")
        raise RuntimeError(f"Could not load model on startup: {e}")


@app.get("/model_columns/", tags=["Prediction"])
def get_model_columns():
    """
       This allows the frontend to dynamically build its prediction form
       with the correct feature names and their possible values.
       """
    try:
        if model_handler is None or model_handler.features is None:
            raise HTTPException(status_code=404, detail="Model not trained yet. Please train a model first.")

        if not isinstance(model_handler.features, dict) or len(model_handler.features) == 0:
            raise HTTPException(status_code=404, detail="Model features are empty. Please train a model first.")

        return model_handler.features

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/predict/")
def predict(request: PredictRequest) :
    """
       Receives feature data in a POST request and returns a prediction
       using the loaded Naive Bayes model.
       """
    try :
        result = model_handler.predict(request.features)
        return {"prediction" : result}
    except Exception as e :
        raise HTTPException(status_code=400, detail=str(e))


