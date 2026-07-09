from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI(title="EV Battery Safety API", version="1.0")
model = xgb.XGBClassifier()

try:
    model.load_model("ev_battery_model.json")
    print("✅ XGBoost Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class EVTelemetry(BaseModel):
    mileage: float
    avg_cell_voltage: float
    max_cell_voltage: float
    min_cell_voltage: float
    avg_current: float
    avg_temp: float
    max_temp: float
    voltage_gap: float

@app.post("/predict")
def predict_battery_health(data: EVTelemetry):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        return {
            "status": "success",
            "prediction_label": prediction,
            "fault_probability_percent": round(probability * 100, 2),
            "assessment": "FAULT DETECTED" if prediction == 1 else "HEALTHY"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))