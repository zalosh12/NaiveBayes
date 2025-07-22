from fastapi import FastAPI, HTTPException, Query
from fastapi import Request

from model_manager import ModelManager
from pydantic import BaseModel
from typing import List, Dict, Union, Any
import os


app = FastAPI()
manager = ModelManager()


class TrainRequest(BaseModel):
    url: str

class PredictRequest(BaseModel):
    model_name: str
    features: dict


@app.post("/train/")
def train_model(request: TrainRequest):
    try:
        name = os.path.basename(request.url)
        accuracy =  manager.create_model_by_df(name,request.url)
        return {"message":"Model trained successfully","accuracy":accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models_list/", response_model=List[str])
def get_models():
    return manager.list_models()

@app.get("/model_columns/{model_name}")
def get_model_columns(model_name: str):
    try :
        options = manager.get_feature_options(model_name)
        print(f"Options for model {model_name}:", options)
        return options
    except Exception as e :
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/")
async def predict(request: Request):
    try:
        data = await request.json()
        print("Received JSON:", data)
        pred_req = PredictRequest(**data)
        prediction = manager.predict(pred_req.model_name, pred_req.features)
        return {"prediction": prediction}
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

