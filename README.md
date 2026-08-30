# EV Battery Fault Prediction API

An end-to-end machine learning project that predicts electric vehicle battery
faults from telemetry data (cell voltages, current, temperature). Built with
XGBoost, FastAPI, and Docker.

## Project Story

The first version of this model looked great on paper — 96% accuracy. It was
wrong in an interesting way: it was **cheating**.

Digging into it, two real data leakage issues turned up:

1. **Row-level train/test split.** The original split randomly divided
   individual telemetry rows instead of whole vehicles, so a car's readings
   could appear in both the training set and the test set. The model was
   partly memorizing, not generalizing.
2. **Mileage as a shortcut.** Faulty batteries in this dataset skew toward
   higher-mileage vehicles. Including raw `mileage` as a feature let the
   model learn "older car → probably faulty" instead of learning anything
   about the actual physical signal.

The pipeline was rebuilt to close both gaps:

- **Vehicle-grouped train/test split** — every reading from a given car stays
  entirely in train or entirely in test, never split across both.
- **GroupKFold cross-validation** during hyperparameter search, so the same
  leakage can't creep back in during tuning.
- **Rolling-window and interaction features** (10-reading rolling mean/std of
  voltage gap, rolling mean of max temperature, thermal load, voltage/temp
  ratio) in place of the raw mileage shortcut.
- **A tuned decision threshold (0.15, not the default 0.5)** — since missing
  a real fault is far more costly than a false alarm, the threshold is set to
  favor recall.

**Result on held-out vehicles the model never saw during training:**

| Metric | Before (leaky) | After (fixed) |
|---|---|---|
| Accuracy | 96%* | 94% |
| Fault recall | ~50% | 97% |

\* The original 96% is not a valid comparison point — it was measured on a
leaky split and doesn't reflect real generalization. It's included here to
show what the fix corrected, not as a benchmark the new model "beat."

## Tech Stack

- **Machine Learning:** Python, XGBoost, Scikit-Learn, Pandas, NumPy
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **Deployment:** Docker

## API Design Note: Why the Request Takes a List of Readings

The fixed model relies on rolling-window features (e.g. "average voltage gap
over the last 10 readings"), which a single real-time telemetry snapshot
doesn't contain on its own. Rather than silently feed the model incomplete
context, the `/predict` endpoint accepts up to the last 10 readings for a
vehicle; the API computes the same rolling/interaction features used in
training and scores the most recent reading.

If only one reading is available, the API still works — it just falls back
to that single reading as its own rolling window (less accurate than having
real history, but not broken).

A production version of this would more likely maintain rolling state
per-vehicle server-side (e.g. in a small database) so callers could just
send the latest reading each time. That's a deliberate scope decision for
this project, not an oversight.

## Project Structure

```
ev-battery-safety-api/
├── data/                       # Raw/processed telemetry (git-ignored)
├── notebooks/                  # EDA, leakage diagnosis, and model iterations
│   ├── prework.ipynb           # Raw telemetry → feature extraction
│   ├── work.ipynb              # First model (leaky split, mileage included)
│   ├── work2.ipynb             # Vehicle-grouped split, class imbalance handling
│   └── work3.ipynb             # Rolling-window features, GroupKFold tuning,
│                                #   threshold tuning — the final pipeline
├── scripts/
│   ├── train_model.py          # Reproduces the work3.ipynb pipeline as a script
│   ├── main.py                 # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── ev_battery_model.json   # Trained XGBoost model
│   └── model_features.json     # Feature order + decision threshold
├── .gitignore
└── README.md
```

## How to Run

### Retrain the model (optional — a trained model is already included)

```bash
cd scripts
python train_model.py --data ../data/final_training_dataset.csv
```

This writes `ev_battery_model.json` and `model_features.json` into
`scripts/`, ready for the API or Docker build to pick up.

### Run locally

```bash
cd scripts
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### Run with Docker

```bash
cd scripts
docker build -t ev-battery-ai .
docker run -p 8000:8000 ev-battery-ai
```

## Example Request

```
POST /predict
```
```json
{
  "car_id": "test-car-1",
  "readings": [
    {"mileage": 90000, "avg_cell_voltage": 3.6, "max_cell_voltage": 3.9, "min_cell_voltage": 3.1,
     "avg_current": -300, "avg_temp": 42, "max_temp": 78, "voltage_gap": 0.6},
    {"mileage": 91000, "avg_cell_voltage": 3.5, "max_cell_voltage": 3.9, "min_cell_voltage": 2.8,
     "avg_current": -340, "avg_temp": 45, "max_temp": 88, "voltage_gap": 0.7}
  ]
}
```

## Example Response

```json
{
  "status": "success",
  "car_id": "test-car-1",
  "prediction_label": 1,
  "fault_probability_percent": 53.11,
  "decision_threshold": 0.15,
  "assessment": "FAULT DETECTED",
  "readings_used_for_context": 2
}
```

## Status

Actively being iterated on. The core leakage fix, retraining pipeline, and
API redesign are complete and validated end-to-end (local + Docker). Next
planned step: exploring server-side rolling state so the API can accept a
single reading per call instead of a manual history window.