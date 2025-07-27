
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from naive_bayes_classifier import NaiveBayesClassifier
import requests

import logging

logger = logging.getLogger("uvicorn.error")

MODEL_TRAINER_URL = "http://model-trainer:8509"


app = FastAPI()

model_handler = NaiveBayesClassifier()

class PredictRequest(BaseModel) :
    features: Dict

@app.on_event("startup")
def fetch_model_on_startup():
    try:
        response = requests.get(f"{MODEL_TRAINER_URL}/export_model/")
        if response.status_code != 200:
            raise Exception(f"Failed to fetch model: {response.text}")

        model_dict = response.json()
        model_handler.load_model(model_dict)
        print("Model loaded successfully on startup")  # <-- הדפסה להצלחת הטעינ
    except Exception as e:
        print(f"Error loading model on startup: {e}")  # <-- הדפסה במקרה של שגיאה
        raise RuntimeError(f"Could not load model on startup: {e}")


@app.get("/model_columns/", tags=["Prediction"])
def get_model_columns():
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
    try :
        result = model_handler.predict(request.features)
        return {"prediction" : result}
    except Exception as e :
        raise HTTPException(status_code=400, detail=str(e))


