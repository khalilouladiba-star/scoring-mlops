"""
Étape 5 — API REST de scoring avec FastAPI.

Charge le modèle "champion" (pipeline scaler + SMOTE + classifieur) produit
par src/train_models.py et expose :
  - GET  /health         : vérification que l'API et le modèle sont OK
  - POST /predict        : prédiction pour un client
  - GET  /model-info      : métadonnées du modèle en production

Lancement local :
    uvicorn api.main:app --reload --port 8000

Déploiement cloud : voir README.md (Heroku / AWS / GCP) et
.github/workflows/cicd.yml pour la pipeline CI/CD.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "champion_metadata.json"

app = FastAPI(
    title="API de Scoring Crédit",
    description="Prédit la probabilité de défaut de paiement d'un client.",
    version="1.0.0",
)

model = None
metadata = None


@app.on_event("startup")
def load_artifacts():
    global model, metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modèle introuvable: {MODEL_PATH}. Lancez d'abord src/train_models.py")
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)


class ClientData(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    monthly_income: float = Field(..., ge=0, example=3500)
    debt_ratio: float = Field(..., ge=0, le=2, example=0.3)
    revolving_utilization: float = Field(..., ge=0, le=2, example=0.4)
    num_credit_lines: int = Field(..., ge=0, example=5)
    num_late_30_59: int = Field(..., ge=0, example=0)
    num_late_60_89: int = Field(..., ge=0, example=0)
    num_late_90_plus: int = Field(..., ge=0, example=0)
    num_dependents: int = Field(..., ge=0, example=1)
    num_real_estate_loans: int = Field(..., ge=0, example=1)

    def to_features(self) -> pd.DataFrame:
        d = self.dict()
        d["total_late_payments"] = d["num_late_30_59"] + d["num_late_60_89"] + d["num_late_90_plus"]
        d["income_per_dependent"] = d["monthly_income"] / (d["num_dependents"] + 1)
        d["has_severe_late_payment"] = int(d["num_late_60_89"] > 0 or d["num_late_90_plus"] > 0)
        d["real_estate_ratio"] = d["num_real_estate_loans"] / (d["num_credit_lines"] + 1)
        return pd.DataFrame([d])


class PredictionResponse(BaseModel):
    default_probability: float
    prediction: int
    decision: str
    threshold_used: float
    model_name: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(client: ClientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    X = client.to_features()
    X = X[metadata["features"]]  # même ordre de colonnes que l'entraînement

    proba = float(model.predict_proba(X)[0, 1])
    threshold = metadata["threshold"]
    pred = int(proba >= threshold)

    return PredictionResponse(
        default_probability=round(proba, 4),
        prediction=pred,
        decision="Refusé (risque de défaut élevé)" if pred == 1 else "Accepté",
        threshold_used=threshold,
        model_name=metadata["name"],
    )
