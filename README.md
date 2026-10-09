# EV Battery Fault Prediction

Can charging telemetry alone tell a faulty EV battery pack from a healthy one, on a car the model has never seen?

This project answers that with XGBoost on 176,327 charging snippets from 49 vehicles, and ships the model behind a small FastAPI service in Docker. The short answer is **partly**: tested fairly on every car, the model catches 88% of faulty snippets but raises a false alarm on 38% of healthy ones. Judged per car instead of per snippet, it separates faulty from healthy cars with an AUC of 0.94.

The first version of the model scored 96%. Most of this repository is the story of why that number was wrong and what an honest test looks like.

## Results

| # | Model | Tested on | Accuracy | Fault recall | False alarms | Majority guess |
|---|---|---|---|---|---|---|
| 1 | All 8 features | Random 80/20 split of rows | 95.8% | 95.7% | 4.0% | 56.0% |
| 2 | Mileage removed | Random 80/20 split of rows | 89.0% | 90.2% | 12.6% | 56.0% |
| 3 | All 8 features | 9 held-out cars (set A) | 70.5% | 52.6% | 25.0% | 80.0% |
| 4 | Model 3 + class weight | 9 held-out cars (set A) | 72.0% | 49.7% | 22.4% | 80.0% |
| 5 | Rebuilt pipeline, cutoff 0.10 | 9 held-out cars (set B) | 82.3% | 97.5% | 49.8% | 68.0% |
| 6 | **Rebuilt pipeline, fair re-test** | **All 49 cars, each held out in turn** | **76.7%** | **88.2%** | **38.0%** | **56.2%** |

- *Fault recall* is the share of faulty snippets caught. *False alarms* is the share of healthy snippets wrongly flagged. *Majority guess* is the accuracy from always predicting the more common class in that test set.
- Rows 1 and 2 are inflated by a row-level split and are not valid for new cars.
- Rows 3 to 5 each rest on a single draw of 9 test cars, and sets A and B are different draws.
- **Row 6 is the number to trust.** It comes from repeated grouped cross-validation (7 folds of cars, 5 repeats), with the decision cutoff chosen on validation cars only. Snippet-level AUC is 0.86 pooled, and 0.84 on average per fold (range 0.66 to 0.92).

Per car, averaging each car's snippet probabilities gives a car-level AUC of 0.94 (0.92 to 0.97 across repeats). With a car cutoff chosen on training cars, about 12.6 of 16 faulty cars are caught and 6.0 of 33 healthy cars are flagged.

## What went wrong with 96%, and what I learned

1. **Mileage was a shortcut.** Mileage was the model's top feature and is the only one that is not a battery measurement. In this dataset, 99.9% of snippets beyond 110,000 km come from faulty cars. Removing mileage dropped accuracy from 95.8% to 89.0%.
2. **The split leaked the answer.** Labels are assigned per car, and a random split of rows puts every car in both train and test. A model that recognizes the car already knows the label. Splitting by car dropped accuracy to 70.5%, below the 80% majority guess for that test set.
3. **One test set of 9 cars is not reliable.** The rebuilt pipeline scored 97.5% recall on one draw of 9 cars and 48.9% accuracy on another. Which cars are drawn changed the result more than which model was used. The cutoff had also been tuned on the test cars.
4. **The rebuild itself added nothing measurable.** Under the fair test, AUC is 0.84 with the 7 original features, 0.84 with the 5 engineered features added, and 0.84 with the class weight. What moved the earlier numbers was the split and the cutoff.
5. **Mileage helps on average but is unstable.** Adding it raises average AUC to 0.86, but it ranges from 0.49 to 0.98 depending on which cars are held out. It is left out because it describes this fleet rather than the battery.

## Dataset

The data comes from the dataset released with:

> Zhang, J., Wang, Y., Jiang, B., He, H., et al. "Realistic fault detection of li-ion battery via dynamical deep learning." *Nature Communications* 14 (2023). https://doi.org/10.1038/s41467-023-41226-5
> Data: https://doi.org/10.6084/m9.figshare.23659323

The vehicle set used here matches "Battery Dataset 3" (49 vehicles, 16 faulty, 176,327 snippets) in He, H., et al., "EVBattery: A Large-Scale Electric Vehicle Dataset for Battery Health and Capacity Estimation" (arXiv:2201.12358).

- 49 vehicles: 33 healthy, 16 faulty
- 176,327 charging snippets, each 128 time steps of 8 signals, one `.pkl` file per snippet (about 1.75 GB)
- One label per car. Every snippet inherits its car's label.
- Cars are very uneven: 27 to 16,282 snippets each. The five largest cars are all faulty and supply 42% of the rows.

The raw data is not included in this repository (see `.gitignore`). Download it from the link above and place it under `data/`.

## Features

`prework.ipynb` reduces each snippet to one row using the mean, max and min over its 128 time steps.

| Feature (code name) | Computed from | What it measures |
|---|---|---|
| `mileage` | metadata | Odometer reading, km. In the final pipeline it only orders a car's snippets and is not a model input. |
| `avg_cell_voltage` | mean of column 0 | Average cell voltage in the snippet |
| `max_cell_voltage` | max of column 0 | Highest voltage reached in the snippet |
| `min_cell_voltage` | min of column 0 | Lowest voltage in the snippet |
| `voltage_gap` | max − min of column 0 | Voltage change during the snippet (not a gap between cells) |
| `avg_current` | mean of column 1 | Charging current, A (negative = charging) |
| `avg_temp` | mean of column 2 | **Charge level, %. Misnamed, see below.** |
| `max_temp` | max of column 2 | **Highest charge level in the snippet. Misnamed.** |

**A naming caveat.** Column 2 was labeled "temperature" when the pipeline was written. It behaves like state of charge instead: it never exceeds 100, is exactly 100 in 35% of snippets, rises only while the car charges, and tracks voltage at r = 0.97. The dataset documentation lists state of charge among its eight signals. The column order should still be confirmed against the dataset's `column.pkl`, and the features renamed. The same applies to `thermal_load` below, which is really charge level × current.

The rebuilt pipeline (`work3.ipynb`, `train_model.py`) adds five features computed per car: `rolling_max_temp_mean_10`, `rolling_voltage_gap_mean_10`, `rolling_voltage_gap_std_10`, `thermal_load` and `voltage_temp_ratio`. Only columns 0 to 2 of the raw data are used so far; columns 3 to 7 are not.

## Project structure

```
ev-battery-safety-api/
├── data/                            # Raw and processed telemetry (git-ignored)
├── notebooks/
│   ├── prework.ipynb                # Raw .pkl files → one training table
│   ├── work.ipynb                   # Exploration, models 1 and 2 (row-level split)
│   ├── work2.ipynb                  # Car-level split, models 3 and 4
│   ├── work3.ipynb                  # Engineered features, GroupKFold tuning, cutoff sweep
│   └── work4_fair_evaluation.ipynb  # Fair re-test on all 49 cars, ablation, per-car results, SHAP
├── scripts/
│   ├── train_model.py               # Rebuilds the work3 pipeline as a script
│   ├── main.py                      # FastAPI application
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── ev_battery_model.json        # Trained XGBoost model
│   └── model_features.json          # Feature order and decision cutoff
├── .gitignore
└── README.md
```

## How to run

### Reproduce the results

Run the notebooks in order from the `notebooks/` folder. `work4_fair_evaluation.ipynb` only needs `data/final_training_dataset.csv`, which `prework.ipynb` produces. With `REPEATS = 5` it takes about 20 minutes on a laptop CPU; set `REPEATS = 1` for a quick run.

Results can differ by a point or so between machines (GPU or CPU, XGBoost version, row order).

### Retrain the served model

```bash
cd scripts
python train_model.py --data ../data/final_training_dataset.csv
```

This writes `ev_battery_model.json` and `model_features.json` into `scripts/`.

### Run the API locally

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

## API

`POST /predict` takes a car ID and up to the last 10 readings for that car. The model uses 10-reading rolling features, which a single snapshot does not contain, so the service computes the same features used in training and scores the most recent reading. With only one reading it still works, using that reading as its own window.

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

`GET /health` reports whether the model loaded and which features it expects.

A production version would keep rolling state per vehicle on the server, so callers could send one reading at a time.

## Known issues and limitations

**In this repository**

- The served cutoff is 0.15 (`model_features.json`). `work3.ipynb` selected 0.10 on its test cars, and the fair re-test selected cutoffs between 0.16 and 0.44 (0.29 on average). The service has not been updated to a validated cutoff.
- `train_model.py` splits 39 training cars and 10 test cars and scores its grid search by recall. `work3.ipynb` uses 40 and 9 and scores by F1. The two should be aligned.
- Feature names containing `temp` refer to charge level (see the naming caveat above).

**In the results**

- Everything rests on 16 faulty cars. Fold-level AUC ranges from 0.66 to 0.92 depending on which cars are held out.
- A 38% false-alarm rate is too high to act on automatically. The model is a first-pass screen at best.
- Labels are per car, so snippets from a faulty car's early life are labeled faulty too. The project cannot yet say how early a fault becomes visible.
- The faulty cars in this dataset are the high-mileage, heavily recorded ones. The model may partly be learning this fleet rather than battery faults. Nothing has been tested on a second dataset.
- In the fair re-test, hyperparameters were fixed at the `work3` grid-search optimum and not re-tuned inside each fold.

## Next steps

1. Confirm what each raw column measures, rename the features, and bring in the five unused signals.
2. Drop the engineered features and class weight, and try features that describe voltage relative to charge level.
3. Reduce false alarms with a per-car decision rule, including a minimum number of snippets before a car can be flagged.
4. Balance the cars so a few large ones do not dominate training.
5. Score each faulty car's snippets in mileage order to test for early warning.
6. Align the API with a validated cutoff and test on a second fleet.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, FastAPI, Uvicorn, Pydantic, Docker.