import numpy as np
import logging
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from manager import Manager

# הגדרת לוגר לתיעוד שגיאות בטרמינל
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phishing Detection API",
    description="API for training a phishing detection model and making predictions.",
    version="1.0.0"
)

manager = Manager()


class TrainFromUrlRequest(BaseModel) :
    url: str


class PredictRequest(BaseModel) :
    features: dict


@app.post("/train_from_upload/", tags=["Training"])
async def train_from_upload(file: UploadFile = File(...)) :
    if not file.filename.endswith('.csv') :
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")

    try :
        accuracy = manager.run(file)
        return {"message" : "Model trained successfully from uploaded file.", "accuracy" : accuracy}

    except HTTPException as e :
        # אם זו שגיאת HTTP שהכנו (למשל, 400 על קובץ פגום), תן לה לעבור
        raise e
    except Exception as e :
        # אם זו שגיאה לא צפויה, תעד אותה והחזר שגיאת 500
        logger.error(f"An unexpected error occurred in /train_from_upload/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")


@app.post("/train_from_url/", tags=["Training"])
def train_from_url(request: TrainFromUrlRequest) :
    try :
        accuracy = manager.run(request.url)
        return {"message" : "Model trained successfully from URL.", "accuracy" : accuracy}

    except HTTPException as e :
        raise e
    except Exception as e :
        logger.error(f"An unexpected error occurred in /train_from_url/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process and train from URL: {str(e)}")


@app.get("/model_columns/", tags=["Prediction"])
def get_model_columns() :
    try :
        if manager.classifier is None or not hasattr(manager.classifier, 'features') :
            raise HTTPException(status_code=404, detail="Model not trained yet. Please train a model first.")

        options = manager.classifier.features
        return options

    except HTTPException as e :
        raise e
    except Exception as e :
        logger.error(f"An unexpected error occurred in /model_columns/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/", tags=["Prediction"])
async def predict(request: PredictRequest) :
    try :
        if manager.classifier is None :
            raise HTTPException(status_code=404, detail="Model not trained yet. Please train a model first.")

        # המרה אוטומטית על ידי Pydantic, אין צורך לקרוא את ה-JSON ידנית
        numpy_prediction = manager.classifier.predict(request.features)

        # המרה מסוגי numpy לסוגי פייתון רגילים
        if isinstance(numpy_prediction, np.integer) :
            python_prediction = int(numpy_prediction)
        elif isinstance(numpy_prediction, np.floating) :
            python_prediction = float(numpy_prediction)
        else :
            python_prediction = numpy_prediction

        return {"prediction" : python_prediction}

    except HTTPException as e :
        raise e
    except Exception as e :
        logger.error(f"An unexpected error occurred in /predict/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))