from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "MENTAL_HEALTH_MODEL.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Student Mental Health Score API", version="1.0.0")


class PredictionInput(BaseModel):
    Age: int = Field(..., ge=0)
    Gender: str
    Country: str
    Academic_Level: str
    Most_Used_Platform: str
    Purpose_Of_Use: str
    Avg_Daily_Usage_Hours: float = Field(..., ge=0)
    Daily_Unlocks: int = Field(..., ge=0)
    Study_Hours: float = Field(..., ge=0)
    Physical_Activity_Hours: float = Field(..., ge=0)
    Sleep_Hours_Per_Night: float = Field(..., ge=0)
    Stress_Level: str


@app.get("/")
def root():
    return {"message": "Student Mental Health Score API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.model_dump()])[0]
    return {"predicted_mental_health_score": float(prediction)}
