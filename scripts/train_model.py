"""
Retrain the EV battery fault model using the leakage-free pipeline you built in
work3.ipynb: car-grouped train/test split, rolling-window + interaction feature
engineering, GroupKFold hyperparameter search, and a tuned decision threshold.

This replaces the very first (leaky) model that's currently baked into the
Docker image: that one used a row-level train_test_split (so a car's readings
could leak across train/test) and included `mileage` as a raw feature, which
acted as a near-direct proxy for the fault label.

Run this LOCALLY where you have access to data/final_training_dataset.csv
(the file your prework.ipynb pipeline produces). It is not runnable here since
the raw telemetry data isn't included in the project export (it's .gitignored).

Usage:
    cd scripts
    python train_model.py --data ../data/final_training_dataset.csv

Outputs (written into the current directory, ready to be copied into the
Docker build context):
    ev_battery_model.json   - the trained XGBoost model
    model_features.json     - the exact feature order + decision threshold
                               main.py needs to reproduce predictions correctly
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold

# Resolve paths relative to THIS FILE's location, not the current working
# directory. The working directory depends on how you launch the script (VS
# Code's Code Runner, a terminal in a different folder, etc.) and is not
# reliable -- this makes the script work the same way no matter where it's
# run from.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / ".." / "data" / "final_training_dataset.csv"


def engineer_battery_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as work3.ipynb's engineer_battery_features.

    1. Rolling statistical windows per vehicle (captures sustained degradation,
       not just a single noisy snapshot)
    2. Interaction features (thermal load, voltage-to-temperature ratio)

    NOTE: `mileage` is used here only to order each car's readings chronologically
    (higher mileage == later reading). It is dropped before training so the model
    can't just learn "old car -> label 1" as a shortcut.
    """
    df_engineered = df.copy()
    df_engineered = df_engineered.sort_values(by=["car", "mileage"]).reset_index(drop=True)

    df_engineered["rolling_max_temp_mean_10"] = df_engineered.groupby("car")["max_temp"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    df_engineered["rolling_voltage_gap_mean_10"] = df_engineered.groupby("car")["voltage_gap"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )
    df_engineered["rolling_voltage_gap_std_10"] = (
        df_engineered.groupby("car")["voltage_gap"]
        .transform(lambda x: x.rolling(window=10, min_periods=1).std())
        .fillna(0)
    )

    df_engineered["thermal_load"] = df_engineered["max_temp"] * np.abs(df_engineered["avg_current"])
    df_engineered["voltage_temp_ratio"] = df_engineered["voltage_gap"] / (df_engineered["max_temp"] + 1e-5)

    return df_engineered


def run_pipeline(df: pd.DataFrame, threshold: float, train_frac: float = 0.8):
    print("Applying feature engineering...")
    df_fe = engineer_battery_features(df)

    # Leakage-free split: whole cars go to either train or test, never both
    all_cars = df_fe["car"].unique()
    np.random.seed(42)
    n_train = int(len(all_cars) * train_frac)
    train_cars = np.random.choice(all_cars, size=n_train, replace=False)
    test_cars = np.array([c for c in all_cars if c not in train_cars])

    df_train = df_fe[df_fe["car"].isin(train_cars)].copy()
    df_test = df_fe[df_fe["car"].isin(test_cars)].copy()

    # mileage is dropped here -- it's only used above to order rolling windows
    drop_cols = ["label", "car", "mileage"]
    if "Unnamed: 0" in df_train.columns:
        drop_cols.append("Unnamed: 0")

    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train["label"]
    X_test = df_test.drop(columns=drop_cols)
    y_test = df_test["label"]

    feature_names = list(X_train.columns)

    num_healthy = int((y_train == 0).sum())
    num_faulty = int((y_train == 1).sum())
    imbalance_ratio = num_healthy / num_faulty
    print(f"Train cars: {len(train_cars)} ({len(df_train)} rows) | "
          f"Test cars: {len(test_cars)} ({len(df_test)} rows)")
    print(f"scale_pos_weight = {imbalance_ratio:.2f}")

    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "n_estimators": [100, 200],
    }

    base_model = xgb.XGBClassifier(
        random_state=42, scale_pos_weight=imbalance_ratio, eval_metric="logloss"
    )

    # GroupKFold keeps every row from a given car in the same fold, so
    # cross-validation can't leak either
    groups = df_train["car"]
    gkf = GroupKFold(n_splits=5)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=list(gkf.split(X_train, y_train, groups=groups)),
        scoring="recall",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    probs = best_model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    print(f"\nAccuracy @ threshold {threshold}: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(f"Fault recall @ threshold {threshold}: {recall_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Healthy (0)", "Fault (1)"]))

    return best_model, feature_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH),
                         help="Path to final_training_dataset.csv")
    parser.add_argument("--out", default=str(SCRIPT_DIR / "ev_battery_model.json"),
                         help="Where to save the trained model")
    parser.add_argument("--threshold", type=float, default=0.15,
                         help="Decision threshold tuned for fault recall (from work3.ipynb)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.resolve()}. "
            f"Pass the correct path explicitly, e.g.:\n"
            f"    python train_model.py --data \"C:\\path\\to\\final_training_dataset.csv\""
        )

    df = pd.read_csv(data_path)
    model, feature_names = run_pipeline(df, threshold=args.threshold)

    out_path = Path(args.out)
    model.save_model(str(out_path))
    features_path = out_path.parent / "model_features.json"
    with open(features_path, "w") as f:
        json.dump({"feature_names": feature_names, "threshold": args.threshold}, f, indent=2)

    print(f"\nSaved model to {out_path.resolve()}")
    print(f"Saved feature_names + threshold to {features_path.resolve()}")
    print("\nBoth files are now in scripts/, alongside main.py, ready for the Docker build.")


if __name__ == "__main__":
    main()