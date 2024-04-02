from fastapi import FastAPI
from pydantic import BaseModel

from model.model import algorithm
from model.modelSingle import algorithm as algorithmSingle
from model.explainable import explaination
from model.model import __version__ as model_version
from typing import List
import numpy as np

app = FastAPI()


class TextIn(BaseModel):
    company_persona: dict
    candidate_array: List[dict]
    w_soft_skills: float
    w_technical_skills: float
    w_education: float
    w_experience: float

class PredictionOut(BaseModel):
    prediction: List[dict]
    
class TextInSingle(BaseModel):
    company_persona: dict
    candidate_persona: dict
    w_soft_skills: float
    w_technical_skills: float
    w_education: float
    w_experience: float

class PredictionOutSingle(BaseModel):
    prediction: dict
    
    
class ExplainIn(BaseModel):
    company_persona: dict
    candidate_persona: dict

class ExplainOut(BaseModel):
    prediction: dict





@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    prediction = algorithm(payload.company_persona,payload.candidate_array,payload.w_technical_skills,payload.w_education,payload.w_soft_skills,payload.w_experience)
    return {"prediction": prediction}

@app.post("/predictSingle", response_model=PredictionOutSingle)
def predict(payload: TextInSingle):
    prediction = algorithmSingle(payload.company_persona,payload.candidate_persona,payload.w_technical_skills,payload.w_education,payload.w_soft_skills,payload.w_experience)
    return {"prediction": prediction}

@app.post("/explain", response_model=ExplainOut)
def predict(payload: ExplainIn):
    prediction = explaination(payload.company_persona,payload.candidate_persona)
    return {"prediction": prediction}
