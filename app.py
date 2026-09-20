import os
import sys

import certifi
import pymongo
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi import FastAPI, Request, UploadFile, File
from uvicorn import run as app_run

from networksecurity.constant.training_pipeline import (
    DATA_INTEGRATION_COLLECTION_NAME,
    DATA_INTEGRATION_DATABASE_NAME,
)
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.exception.exception import NetworkSecurityException

import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


# Load environment variables
load_dotenv()

# MongoDB connection
ca = certifi.where()
MONGO_DB_URI = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(
    MONGO_DB_URI,
    tlsCAFile=ca
)

database = client[DATA_INTEGRATION_DATABASE_NAME]
collection = database[DATA_INTEGRATION_COLLECTION_NAME]


# FastAPI application
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

        return Response(content="Training is successful")

    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        df = pd.read_csv(file.file)

        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=final_model
        )

        y_pred = network_model.predict(df)

        df["predicted_column"] = y_pred

        os.makedirs("predicted_output", exist_ok=True)
        df.to_csv("predicted_output/output.csv", index=False)

        return df.to_dict(orient="records")

    except Exception as e:
        raise NetworkSecurityException(e, sys)



if __name__ == "__main__":
    app_run(
        app,
        host="localhost",
        port=8000
    )