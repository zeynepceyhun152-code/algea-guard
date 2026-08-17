"""
AlgaeGuard API — FastAPI backend.

Serves:
  GET  /api/waterbodies              list of monitored sites
  GET  /api/history/{waterbody}      daily sensor + risk history for charting
  GET  /api/metrics                  model evaluation metrics (honest numbers)
  POST /api/predict                  next-day phycocyanin + bloom-alert prediction
       given today's sensor readings (what-if simulator)
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
DATA = pd.read_csv(BASE / "data" / "features_daily.csv", parse_dates=["date"])
REG = joblib.load(BASE / "models" / "regressor.pkl")
CLF = joblib.load(BASE / "models" / "classifier.pkl")
METRICS = json.loads((BASE / "models" / "metrics.json").read_text())

FEATURE_COLS = [
    "water temperature", "ph", "specific conductivity",
    "dissolved oxygen conc.", "dissolved oxygen sat.", "turbidity",
    "chlorophyll", "phycocyanin", "air temperature", "barometric pressure",
    "wind speed", "no3-n",
    "water temperature_roll3", "ph_roll3", "specific conductivity_roll3",
    "dissolved oxygen conc._roll3", "dissolved oxygen sat._roll3",
    "turbidity_roll3", "chlorophyll_roll3", "phycocyanin_roll3",
    "water temperature_delta1", "ph_delta1", "phycocyanin_delta1",
]

app = FastAPI(title="AlgaeGuard API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class SensorReading(BaseModel):
    waterbody: str = Field(..., description="Name of the monitored waterbody")
    water_temperature: float
    ph: float
    specific_conductivity: float
    dissolved_oxygen_conc: float
    dissolved_oxygen_sat: float
    turbidity: float
    chlorophyll: float
    phycocyanin: float
    air_temperature: float = 20.0
    barometric_pressure: float = 1013.0
    wind_speed: float = 2.0
    no3_n: float = 0.1


@app.get("/api/waterbodies")
def waterbodies():
    return sorted(DATA["waterbody"].unique().tolist())


@app.get("/api/history/{waterbody}")
def history(waterbody: str):
    g = DATA[DATA["waterbody"] == waterbody].sort_values("date")
    if g.empty:
        raise HTTPException(404, f"Unknown waterbody '{waterbody}'")
    cols = ["date", "phycocyanin", "chlorophyll", "water temperature", "ph", "risk_tier"]
    g = g[cols].dropna(subset=["phycocyanin"])
    g["date"] = g["date"].dt.strftime("%Y-%m-%d")
    return g.to_dict(orient="records")


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(reading: SensorReading):
    row = {
        "water temperature": reading.water_temperature,
        "ph": reading.ph,
        "specific conductivity": reading.specific_conductivity,
        "dissolved oxygen conc.": reading.dissolved_oxygen_conc,
        "dissolved oxygen sat.": reading.dissolved_oxygen_sat,
        "turbidity": reading.turbidity,
        "chlorophyll": reading.chlorophyll,
        "phycocyanin": reading.phycocyanin,
        "air temperature": reading.air_temperature,
        "barometric pressure": reading.barometric_pressure,
        "wind speed": reading.wind_speed,
        "no3-n": reading.no3_n,
    }
    # For a live single-reading request we don't have real 3-day rolling history,
    # so we approximate roll3 with today's value and delta1 with 0 (no change known
    # yet). This is documented as a simplification for the what-if simulator; the
    # /api/history-backed dashboard uses true rolling features computed from data.
    for base in ["water temperature", "ph", "specific conductivity",
                 "dissolved oxygen conc.", "dissolved oxygen sat.", "turbidity",
                 "chlorophyll", "phycocyanin"]:
        row[f"{base}_roll3"] = row[base]
    row["water temperature_delta1"] = 0.0
    row["ph_delta1"] = 0.0
    row["phycocyanin_delta1"] = 0.0

    X = pd.DataFrame([row])[FEATURE_COLS]
    delta = float(REG.predict(X)[0])
    predicted_level = max(0.0, reading.phycocyanin + delta)
    alert_prob = float(CLF.predict_proba(X)[0][1])
    alert = bool(alert_prob >= 0.5)

    return {
        "waterbody": reading.waterbody,
        "today_phycocyanin_rfu": reading.phycocyanin,
        "predicted_next_day_phycocyanin_rfu": round(predicted_level, 3),
        "predicted_change_rfu": round(delta, 3),
        "bloom_alert_probability": round(alert_prob, 3),
        "bloom_alert": alert,
        "risk_label": "Alert" if alert_prob >= 0.5 else ("Elevated" if alert_prob >= 0.25 else "Watch"),
    }


# Serve the static frontend dashboard from the same origin (avoids CORS setup for demo)
app.mount("/", StaticFiles(directory=str(BASE / "frontend"), html=True), name="frontend")
