import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="EV Battery Safety API", version="2.0")

# Resolve model/metadata paths relative to THIS FILE's location, not the
# process's working directory -- avoids "file not found" depending on how
# the script gets launched (Code Runner, a terminal, Docker's WORKDIR, etc.)
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "ev_battery_model.json"
FEATURES_PATH = SCRIPT_DIR / "model_features.json"

model = xgb.XGBClassifier()
FEATURE_NAMES: List[str] = []
THRESHOLD = 0.15

try:
    model.load_model(str(MODEL_PATH))
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    FEATURE_NAMES = meta["feature_names"]
    THRESHOLD = meta.get("threshold", 0.15)
    print(f"Model + feature metadata loaded. Expecting features: {FEATURE_NAMES}")
except Exception as e:
    # Plain ASCII only here -- some Windows terminals default to cp1252,
    # which can't encode emoji and will crash this print statement itself,
    # masking the real error underneath it.
    print(f"ERROR loading model/metadata: {e}")
    print(f"Looked for model at: {MODEL_PATH}")
    print(f"Looked for feature metadata at: {FEATURES_PATH}")
    print("Run train_model.py first (it writes both files into this scripts/ folder).")


class TelemetryReading(BaseModel):
    # mileage is only used to order readings chronologically -- it is NOT fed
    # to the model. It correlated with the fault label strongly enough that
    # the original model was using it as a shortcut instead of learning the
    # real physical signal, so it's excluded from the trained feature set.
    mileage: float
    avg_cell_voltage: float
    max_cell_voltage: float
    min_cell_voltage: float
    avg_current: float
    avg_temp: float
    max_temp: float
    voltage_gap: float


class PredictRequest(BaseModel):
    car_id: str
    # The trained model uses a 10-reading rolling window (mean/std of voltage
    # gap, mean of max temp) to capture sustained stress rather than a single
    # noisy snapshot. Send up to the last 10 readings for this vehicle here,
    # in any order -- they get sorted by mileage before scoring. The reading
    # with the highest mileage is the one that gets evaluated; the earlier
    # ones just provide rolling-window context.
    # If you only have a single reading, send a list of length 1 -- the
    # rolling stats will simply equal that one value (std = 0), which is a
    # reasonable fallback but less accurate than having real history.
    readings: List[TelemetryReading] = Field(..., min_length=1, max_length=10)


def engineer_features(readings_df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors engineer_battery_features() from train_model.py / work3.ipynb,
    applied to a single vehicle's window of readings instead of the full
    training set."""
    df = readings_df.sort_values(by="mileage").reset_index(drop=True)

    df["rolling_max_temp_mean_10"] = df["max_temp"].rolling(window=10, min_periods=1).mean()
    df["rolling_voltage_gap_mean_10"] = df["voltage_gap"].rolling(window=10, min_periods=1).mean()
    df["rolling_voltage_gap_std_10"] = (
        df["voltage_gap"].rolling(window=10, min_periods=1).std().fillna(0)
    )
    df["thermal_load"] = df["max_temp"] * df["avg_current"].abs()
    df["voltage_temp_ratio"] = df["voltage_gap"] / (df["max_temp"] + 1e-5)

    return df


@app.post("/predict")
def predict_battery_health(request: PredictRequest):
    if not FEATURE_NAMES:
        raise HTTPException(status_code=503, detail="Model not loaded correctly on server startup.")

    try:
        raw = pd.DataFrame([r.dict() for r in request.readings])
        engineered = engineer_features(raw)

        # Evaluate the most recent reading (highest mileage after sorting)
        current_row = engineered.iloc[[-1]][FEATURE_NAMES]

        probability = float(model.predict_proba(current_row)[0][1])
        prediction = int(probability >= THRESHOLD)

        return {
            "status": "success",
            "car_id": request.car_id,
            "prediction_label": prediction,
            "fault_probability_percent": round(probability * 100, 2),
            "decision_threshold": THRESHOLD,
            "assessment": "FAULT DETECTED" if prediction == 1 else "HEALTHY",
            "readings_used_for_context": len(request.readings),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": bool(FEATURE_NAMES), "expected_features": FEATURE_NAMES}