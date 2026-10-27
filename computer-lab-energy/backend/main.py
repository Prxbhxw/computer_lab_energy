from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "historical_energy.csv"
MODEL_PATH = BASE / "energy_model.joblib"
METRICS_PATH = BASE / "model_metrics.json"

app = FastAPI(title="Computer Lab Energy API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    if not MODEL_PATH.exists():
        # Convenience fallback: train on first run if user skipped the training command.
        from train_model import train
        train()
    return joblib.load(MODEL_PATH)

def load_data():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=500, detail="Historical CSV dataset was not found.")
    return pd.read_csv(DATA_PATH)

model = load_model()

class PredictionInput(BaseModel):
    computers_used: int = Field(ge=1, le=500)
    operating_hours: float = Field(gt=0, le=24)
    avg_power_watts: float = Field(gt=0, le=2000)
    tariff_per_kwh: float = Field(default=8.0, ge=0, le=1000)

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/api/summary")
def summary():
    df = load_data()
    # Aggregate rows by month for a simple trend line.
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%b %Y")
    monthly = df.groupby("month", sort=False)["energy_kwh"].sum().reset_index()
    recent = df.tail(30)
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "total_kwh": round(float(df["energy_kwh"].sum()), 1),
        "average_daily_kwh": round(float(df["energy_kwh"].mean()), 2),
        "days_recorded": int(len(df)),
        "peak_daily_kwh": round(float(df["energy_kwh"].max()), 2),
        "monthly_trend": [
            {"month": row["month"], "energy_kwh": round(float(row["energy_kwh"]), 1)}
            for _, row in monthly.iterrows()
        ],
        "recent_days": [
            {"date": row["date"].strftime("%d %b"), "energy_kwh": round(float(row["energy_kwh"]), 2)}
            for _, row in recent.iterrows()
        ],
        "metrics": metrics,
        "dataset_note": "Synthetic demonstration data — replace with real historical readings."
    }

@app.post("/api/predict")
def predict(payload: PredictionInput):
    try:
        row = pd.DataFrame([{
            "computers_used": payload.computers_used,
            "operating_hours": payload.operating_hours,
            "avg_power_watts": payload.avg_power_watts,
        }])
        kwh = max(0.0, float(model.predict(row)[0]))
        return {
            "predicted_kwh": round(kwh, 2),
            "estimated_cost": round(kwh * payload.tariff_per_kwh, 2),
            "tariff_per_kwh": payload.tariff_per_kwh,
            "inputs": payload.model_dump()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
