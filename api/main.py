from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import boto3
import os
import json
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Heart Risk Predictor API")

# ---------------------------------
# Cargar modelo (pipeline sklearn)
# ---------------------------------
BASE_DIR = Path(__file__).resolve().parent       # carpeta api/
MODEL_PATH = BASE_DIR / "model_pipeline.pkl"

print("Cargando modelo desde:", MODEL_PATH)

pipe = joblib.load(MODEL_PATH)

# ---------------------------------
# Configuración S3 (opcional)
# ---------------------------------
S3_BUCKET = os.getenv("S3_BUCKET", "")  # deja vacío si no vas a usar S3
s3 = boto3.client("s3") if S3_BUCKET else None


class PatientFeatures(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.post("/predict")
def predict(data: PatientFeatures):
    X = [data.model_dump()]

    proba = pipe.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)

    result = {
        "prediction": pred,
        "probability": float(proba),
        "timestamp": datetime.utcnow().isoformat()
    }

    # Log en S3 si está configurado
    if s3 and S3_BUCKET:
        key = f"predictions/{result['timestamp']}.json"
        payload = {"input": X[0], "output": result}
        try:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=json.dumps(payload).encode("utf-8")
            )
        except Exception as e:
            print("No se pudo escribir en S3:", e)

    return result