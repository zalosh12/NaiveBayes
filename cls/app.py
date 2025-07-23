from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from naive_bayes_classifier import NaiveBayesClassifier
import requests

app = FastAPI()

model_handler = NaiveBayesClassifier()

class PredictRequest(BaseModel) :
    features: Dict

@app.on_event("startup")
def fetch_model_on_startup() :
    try :
        # כתובת הקונטיינר השני (לפי שם שירות ברשת docker או כתובת IP)
        response = requests.get("http://model-server:8000/export_model/")
        if response.status_code != 200 :
            raise Exception(f"Failed to fetch model: {response.text}")

        model_dict = response.json()
        model_handler.load_model(model_dict)

    except Exception as e :
        raise RuntimeError(f"Could not load model on startup: {e}")




@app.get("/model_columns/", tags=["Prediction"])
def get_model_columns() :
    try :
        if model_handler is None or not hasattr(model_handler.features, 'features') :
            raise HTTPException(status_code=404, detail="Model not trained yet. Please train a model first.")

        options = model_handler.features
        return options

    except HTTPException as e :
        raise e
    except Exception as e :
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/")
def predict(request: PredictRequest) :
    try :
        result = model_handler.predict(request.features)
        return {"prediction" : result}
    except Exception as e :
        raise HTTPException(status_code=400, detail=str(e))