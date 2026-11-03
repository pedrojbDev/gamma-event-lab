"""HTTP interface for simulated feature vectors."""

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat

from gamma_lab.core import predict

app = FastAPI(title="Gamma Event Lab")
MODEL = Path(__file__).resolve().parent.parent / "model" / "model.json"


class Event(BaseModel):
    features: Annotated[list[FiniteFloat], Field(min_length=10, max_length=10)]


@app.post("/predict")
def classify(event: Event):
    if not MODEL.is_file():
        raise HTTPException(503, "Model unavailable; train or install the model artifact")
    model = json.loads(MODEL.read_text())
    score = float(predict(model, [event.features])[0])
    return {
        "gamma_score": score,
        "threshold": model["threshold"],
        "selected": score >= model["threshold"],
        "simulation_only": True,
    }
