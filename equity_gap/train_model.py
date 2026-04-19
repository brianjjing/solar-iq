"""
SolarIQ Equity Gap - Phase 2: XGBoost Model + SHAP
Inputs:  ../equity_gap_data.csv
Outputs: ../model.pkl, ../predictions.csv
Run: python equity_gap/train_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "equity_gap_data.csv"
MODEL_PKL = ROOT / "model.pkl"
PRED_CSV = ROOT / "predictions.csv"

# CA eGRID CAMX subregion CO2 intensity (EPA eGRID 2022)
CA_CO2_LBS_PER_KWH = 0.476
AVG_SYSTEM_KW = 5.0       # CA LBNL median system size
SYSTEM_DERATE = 0.80
DAYS_PER_YEAR = 365
LBS_PER_TON = 2000

FEATURES = [
    "median_income",
    "solrad_annual",
    "owner_pct",
    "housing_units",
    "electricity_rate",
    "pct_third_party",
]


def compute_carbon_impact(row) -> float:
    missed_installs = max(row["gap_score"], 0) / 1000.0 * row["housing_units"]
    annual_kwh = AVG_SYSTEM_KW * row["solrad_annual"] * DAYS_PER_YEAR * SYSTEM_DERATE
    missed_co2_lbs = missed_installs * annual_kwh * CA_CO2_LBS_PER_KWH
    return round(missed_co2_lbs / LBS_PER_TON, 2)


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} ZIPs from {DATA_CSV.name}")

    # Filter: enough housing units for a reliable adoption rate signal
    model_df = df[df["housing_units"] >= 500].copy()
    model_df = model_df.dropna(subset=FEATURES + ["adoption_rate"])
    print(f"Training set after filters: {len(model_df)} ZIPs")

    # Log-transform target (many near-zero values)
    model_df["log_adoption"] = np.log1p(model_df["adoption_rate"])

    X = model_df[FEATURES].values
    y = model_df["log_adoption"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, model_df.index.values, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    print(f"Test RMSE (log scale): {rmse:.4f}   R2: {r2:.4f}")

    # SHAP values on test set
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_importance = dict(zip(FEATURES, mean_abs_shap.tolist()))
    print("SHAP feature importance (mean |SHAP|):")
    for feat, val in sorted(shap_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:<25} {val:.4f}")

    # Predict for ALL ZIPs in dataset
    all_df = df.dropna(subset=FEATURES).copy()
    X_all = all_df[FEATURES].values
    all_df["predicted_log_adoption"] = model.predict(X_all)
    all_df["predicted_adoption"] = np.expm1(all_df["predicted_log_adoption"])
    all_df["actual_adoption"] = all_df["adoption_rate"].fillna(0)
    all_df["gap_score"] = all_df["predicted_adoption"] - all_df["actual_adoption"]

    # Normalize gap score to 0-100 for display
    gap_min = all_df["gap_score"].min()
    gap_max = all_df["gap_score"].max()
    all_df["gap_score_normalized"] = (
        (all_df["gap_score"] - gap_min) / (gap_max - gap_min) * 100
    ).round(1)

    # Carbon impact
    all_df["missed_co2_tons_per_year"] = all_df.apply(compute_carbon_impact, axis=1)
    all_df["missed_installs"] = (
        all_df["gap_score"].clip(lower=0) / 1000.0 * all_df["housing_units"]
    ).round(1)

    # SHAP per-ZIP for key features (run on full dataset for app display)
    shap_all = explainer(X_all)
    for i, feat in enumerate(FEATURES):
        all_df[f"shap_{feat}"] = shap_all.values[:, i]

    # Save SHAP importance summary for app
    shap_summary = {f: round(v, 4) for f, v in shap_importance.items()}

    out_cols = [
        "zip", "city", "lat", "lon",
        "median_income", "housing_units", "owner_pct", "population",
        "solrad_annual", "solrad_source",
        "install_count", "actual_adoption", "predicted_adoption",
        "gap_score", "gap_score_normalized",
        "missed_installs", "missed_co2_tons_per_year",
        "electricity_rate", "nem_type",
        "median_system_kw", "median_cost_per_watt",
    ] + [f"shap_{f}" for f in FEATURES]

    out_cols = [c for c in out_cols if c in all_df.columns]
    all_df[out_cols].to_csv(PRED_CSV, index=False)

    # Save model + metadata
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "shap_importance": shap_summary}, f)

    print(f"\nSaved model -> {MODEL_PKL.name}")
    print(f"Saved predictions -> {PRED_CSV.name}  ({len(all_df)} ZIPs)")

    top10 = all_df.nlargest(10, "gap_score")[["zip", "city", "median_income", "actual_adoption", "predicted_adoption", "gap_score", "missed_co2_tons_per_year"]]
    print("\nTop 10 Equity Gap ZIPs:")
    print(top10.to_string(index=False))

    top100_co2 = all_df.nlargest(100, "gap_score")["missed_co2_tons_per_year"].sum()
    cars = top100_co2 / 4.6
    print(f"\nHeadline: Closing top-100 gaps = {top100_co2:,.0f} tons CO2/yr ({cars:,.0f} cars removed)")


if __name__ == "__main__":
    main()
