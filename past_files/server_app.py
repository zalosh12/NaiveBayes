from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from past_files.classifier_manager import ClassifierManager
from server.naive_bayes_classifier import NaiveBayesClassifier
from server.evaluator import Evaluator
from server.data_loader import DataLoader
from server.data_spliter import DataSplitter

app = FastAPI()

classifier = NaiveBayesClassifier()
evaluator = Evaluator(classifier)
data_loader = DataLoader
data_splitter = DataSplitter

manager = ClassifierManager(
    classifier=classifier,
    evaluator=evaluator,
    ui=None,
    data_loader=data_loader,
    data_splitter=data_splitter
)

class URLInput(BaseModel):
    url: str

@app.post("/upload_from_url/")
def upload_from_url(data: URLInput):
    print(f"we recieved an upload request from{data.url}")
    try:
        response = requests.get(data.url)
        response.raise_for_status()
        manager.load_data(data.url)
        return {"message": "Data loaded from URL"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optional_features/")
def get_features():
    try:
        features = manager.classifier.features
        if features is None:
            raise ValueError("Model is not trained or features are undefined.")
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get features: {str(e)}")

@app.post("/split/")
def split():
    try:
        manager.split_data()
        return {"message": "Data split successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/")
def train():
    try:
        manager.train()
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate/")
def evaluate():
    try:
        accuracy = manager.evaluate()
        return {"accuracy": float(accuracy)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

class SampleInput(BaseModel):
    features: dict

@app.post("/predict/")
def predict(sample: SampleInput):
    try:
        # for k, v in sample.features.items() :
        #     print(f"{k}: {v}")
        prediction = manager.predict_sample(sample.features)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

