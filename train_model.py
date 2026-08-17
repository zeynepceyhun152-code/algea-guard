"""
Train AlgaeGuard's forecasting models on real EPA buoy data.

Two models sharing one feature set:
  1. RandomForestRegressor  -> next-day phycocyanin level (continuous, RFU)
  2. RandomForestClassifier -> next-day risk tier (Watch / Elevated / Alert)

Evaluation uses a TIME-ORDERED split per waterbody (last 20% of each site's
timeline held out) — never a random shuffle — because random splits leak
future information into training for time-series data and would make the
reported metrics meaningless.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

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


def time_split(df: pd.DataFrame, test_frac=0.2):
    train_parts, test_parts = [], []
    for wb, g in df.groupby("waterbody"):
        g = g.sort_values("date")
        cut = int(len(g) * (1 - test_frac))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def main():
    df = pd.read_csv(DATA_DIR / "features_daily.csv", parse_dates=["date"])
    df = df.dropna(subset=FEATURE_COLS + ["target_next_day", "risk_tier"])
    print(f"Usable rows after dropping incomplete days: {len(df)}")

    # Predict the CHANGE from today, not the raw level. Cyanobacteria levels are
    # highly autocorrelated day-to-day, so modelling the delta (and adding it back
    # onto today's reading) is a much stronger, more honest formulation than asking
    # a model to reproduce the persistence baseline from scratch.
    train, test = time_split(df)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows (time-ordered split, per waterbody)")

    X_train, X_test = train[FEATURE_COLS], test[FEATURE_COLS]

    # ---- Regressor: next-day CHANGE in phycocyanin, added back onto today's value ----
    y_train_r = train["target_next_day"] - train["phycocyanin"]
    y_test_r = test["target_next_day"] - test["phycocyanin"]
    reg = RandomForestRegressor(n_estimators=500, max_depth=5, min_samples_leaf=4, random_state=42)
    reg.fit(X_train, y_train_r)
    pred_delta = reg.predict(X_test)
    pred_level = test["phycocyanin"].values + pred_delta

    mae = mean_absolute_error(test["target_next_day"], pred_level)
    r2 = r2_score(test["target_next_day"], pred_level)
    print(f"\n[Regressor] next-day phycocyanin (RFU), delta-model — MAE={mae:.3f}  R2={r2:.3f}")

    # Naive baseline: "tomorrow = today" (persistence model), the real bar to beat
    persistence_pred = test["phycocyanin"]
    base_mae = mean_absolute_error(test["target_next_day"], persistence_pred)
    print(f"[Baseline]  persistence (today's value)                — MAE={base_mae:.3f}")

    # ---- Classifier: next-day BLOOM ALERT (binary — top risk tier vs rest) ----
    # Binary framing matches how an operator actually acts (issue an alert or not),
    # and is far more learnable from ~450 days of data than a 3-way split.
    train["alert"] = (train["risk_tier"] == "Alert").astype(int)
    test["alert"] = (test["risk_tier"] == "Alert").astype(int)
    y_train_c, y_test_c = train["alert"], test["alert"]
    clf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=4,
                                  random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train_c)
    pred_c = clf.predict(X_test)
    acc = accuracy_score(y_test_c, pred_c)
    f1 = f1_score(y_test_c, pred_c, average="binary")
    print(f"\n[Classifier] next-day BLOOM ALERT (binary) — accuracy={acc:.3f}  F1={f1:.3f}")
    print(classification_report(y_test_c, pred_c, target_names=["No alert", "Alert"]))

    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances (risk classifier):")
    print(importances.head(8))

    joblib.dump(reg, MODEL_DIR / "regressor.pkl")
    joblib.dump(clf, MODEL_DIR / "classifier.pkl")

    metrics = {
        "regressor_mae": round(float(mae), 3),
        "regressor_r2": round(float(r2), 3),
        "baseline_persistence_mae": round(float(base_mae), 3),
        "classifier_accuracy": round(float(acc), 3),
        "classifier_macro_f1": round(float(f1), 3),
        "n_train": len(train),
        "n_test": len(test),
        "feature_importance_top8": importances.head(8).round(4).to_dict(),
        "waterbodies": sorted(df["waterbody"].unique().tolist()),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved models + metrics.json -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
