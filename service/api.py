from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: List[float]


class PredictBatchRequest(BaseModel):
    rows: List[List[float]]


root = Path(__file__).resolve().parents[1]
model_path = root / "artifacts" / "model.joblib"
if not model_path.exists():
    raise RuntimeError("model artifact not found; run scripts/train.py first")

model = joblib.load(model_path)
app = FastAPI(title="AI Biotech ML Service")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict_one(req: PredictRequest) -> dict:
    x = np.array(req.features, dtype=float).reshape(1, -1)
    try:
        proba = float(model.predict_proba(x)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proba": proba}


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest) -> dict:
    X = np.array(req.rows, dtype=float)
    try:
        proba = model.predict_proba(X)[:, 1].astype(float).tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proba": proba}


