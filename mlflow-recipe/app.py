from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from typing import Optional
import pandas as pd
import joblib

app = FastAPI()

class ClassificationInput(BaseModel):
    Timestamp: int
    Gmtoffset: Optional[int] = 0
    Datetime: datetime
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

    @field_validator('Datetime')
    @classmethod
    def validate_datetime(cls, value):
         print(value)
         return pd.to_datetime(value) 
         

class RegressionInput(BaseModel):
    Open: float
    High: float
    Low: float
    Volume: int


classification_model = joblib.load('models/model_classification.pkl')
regression_model = joblib.load('models/model_regression.pkl')

@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/predict_classification")
async def predict(input_data: ClassificationInput):
    
        df = pd.DataFrame([input_data.model_dump().values()], 
                          columns=input_data.model_dump().keys())
        pred = classification_model.predict(df)
        return {"predicted_class": int(pred[0])}



@app.post("/predict_regression")
async def predict(input_data: RegressionInput):
    
        df = pd.DataFrame([input_data.model_dump().values()], 
                          columns=input_data.model_dump().keys())
        print(df)
        pred = regression_model.predict(df)
        return {"Regression_result": pred[0]}
