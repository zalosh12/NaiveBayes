import numpy as np
import logging
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from manager import Manager
import json

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger("uvicorn.error")

# URLs of other services in the Docker network
CLS_SERVER_URL = "http://model-classifier:8010"
CLIENT_URL = "http://streamlit-client:8011"


app = FastAPI()



manager = Manager() # Main model manager instance


class TrainFromUrlRequest(BaseModel) :
    url: str # URL to load CSV data for training


class PredictRequest(BaseModel) :
    features: dict # Feature data for prediction

@app.on_event("startup")
def default_data():
    # On server startup, train model on default data
    try:
        accuracy = manager.run(file_src=None)
        logger.info(f"Model initialized with accuracy: {accuracy}")
    except Exception as e:
        logger.error(f"Failed to initialize model with default data: {e}")

@app.post("/train_from_upload/", tags=["Training"])
async def train_from_upload(file: UploadFile = File(...)) :
    # Train model from uploaded CSV file
    if not file.filename.endswith('.csv') :
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try :
        accuracy = manager.run(file)
        return {"message" : "Model trained successfully from uploaded file.", "accuracy" : accuracy}

    except HTTPException as e :

        raise e
    except Exception as e :
        logger.error(f"An unexpected error occurred in /train_from_upload/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@app.post("/train_from_url/", tags=["Training"])
def train_from_url(request: TrainFromUrlRequest) :
    # Train model from CSV data downloaded from a given URL
    try :
        accuracy = manager.run(request.url)
        return {"message" : "Model trained successfully from URL.", "accuracy" : accuracy}

    except HTTPException as e :
        raise e
    except Exception as e :
        logger.error(f"An unexpected error occurred in /train_from_url/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process and train from URL: {str(e)}")

@app.get("/export_model/", tags=["Model"])
def export_model():
    # Export the trained model as JSON (handling numpy data types)
    try:
        model = manager.trained_model
        if model is None:
            raise Exception("No trained model found.")

        json_compatible_model = jsonable_encoder(model, custom_encoder={
            np.integer: int,
            np.floating: float,
            np.ndarray: lambda x: x.tolist()
        })

        return json_compatible_model

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))