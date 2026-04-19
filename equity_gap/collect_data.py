"""
SolarIQ Equity Gap - Phase 1: Data Collection
Outputs: ../equity_gap_data.csv

Run: python equity_gap/collect_data.py --nrel-key YOUR_KEY [--eia-key KEY]
Keys also accepted via env vars NREL_KEY, EIA_KEY (or .env in project root)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.spatial import KDTree
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import _pvwatts_call, fetch_eia_data

load_dotenv(ROOT / ".env")

RECORDS_CSV = ROOT / "records - Sheet1.csv"
LBNL_CSV    = ROOT / "TTS_LBNL_public_file_29-Sep-2025_all.csv"
OUT_CSV     = ROOT / "equity_gap_data.csv"

CENSUS_ACS_URL  = "https://api.census.gov/data/2022/acs/acs5"
NREL_SAMPLE_N   = 200
CA_EIA_RATE     = 0.3029
CA_NEM_TYPE     = "NEM3"
CA_NEM_EXPORT_PCT = 0.20


# ---------------------------------------------------------------------------
# Step 1a: Records CSV — install counts per ZIP
# ---------------------------------------------------------------------------

def aggregate_records(centroids: pd.DataFrame) -> pd.DataFrame:
    print("[1/5] Aggregating records install data...")
    df = pd.read_csv(RECORDS_CSV, header=None, low_memory=False)

    df.columns = [f"c{i}" for i in range(df.shape[1])]
    df["zip"] = df["c17"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["c11"], errors="coerce")
    df["lon"] = pd.to_numeric(df["c12"], errors="coerce")
    df["system_kw"] = pd.to_numeric(df["c8"], errors="coerce")
    df["state"] = df["c15"].astype(str).str.strip().str.upper()

    df = df[df["state"] == "CA"].copy()

    bad = df["zip"].isna() | ~df["zip"].str.match(r"^\d{5}$")
    missing = df[bad & df["lat"].notna() & df["lon"].notna()]
    print(f"      -> {bad.sum()} rows missing ZIP; filling {len(missing)} via nearest centroid...")

    if not missing.empty and not centroids.empty:
        cent = centroids.dropna(subset=["lat", "lon"])
        tree = KDTree(cent[["lat", "lon"]].values)
        _, idxs = tree.query(missing[["lat", "lon"]].values)
        df.loc[missing.index, "zip"] = cent.iloc[idxs]["zip"].values
        print(f"      -> {len(missing)} ZIPs filled instantly via nearest centroid (no API calls)")

    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df = df[df["zip"].str.match(r"^\d{5}$")]

    agg = df.groupby("zip").agg(
        install_count=("zip", "count"),
        median_system_kw=("system_kw", "median"),
    ).reset_index()

    print(f"      -> {len(agg)} ZIPs with install data, {agg['install_count'].sum():,} total installs")
    return agg


# ---------------------------------------------------------------------------
# Step 1b: LBNL — cost and third-party metadata only (not install counts)
# ---------------------------------------------------------------------------

def lbnl_supplemental() -> pd.DataFrame:
    """Returns per-ZIP cost/rebate and third-party ownership from LBNL (no install counts)."""
    df = pd.read_csv(
        LBNL_CSV,
        usecols=["zip_code", "customer_segment", "installation_date",
                 "PV_system_size_DC", "total_installed_price",
                 "rebate_or_grant", "third_party_owned"],
        low_memory=False,
    )
    df = df[df["customer_segment"].isin(["RES", "RES_SF"])]
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    df = df[df["zip_code"].str.match(r"^\d{5}$")]

    df["installation_date"] = pd.to_datetime(df["installation_date"], format="mixed", errors="coerce", dayfirst=False)
    df["PV_system_size_DC"] = pd.to_numeric(df["PV_system_size_DC"], errors="coerce")
    df["total_installed_price"] = pd.to_numeric(df["total_installed_price"], errors="coerce")
    df["third_party_owned"] = pd.to_numeric(df["third_party_owned"], errors="coerce")

    df = df.dropna(subset=["PV_system_size_DC"]).copy()
    df = df[df["PV_system_size_DC"] > 0]
    df["cost_per_watt"] = df["total_installed_price"] / (df["PV_system_size_DC"] * 1000)
    df = df[(df["cost_per_watt"] > 0.5) & (df["cost_per_watt"] < 15)]

    agg = df.groupby("zip_code").agg(
        median_cost_per_watt=("cost_per_watt", "median"),
        pct_third_party=("third_party_owned", lambda x: (pd.to_numeric(x, errors="coerce") == 1).mean()),
    ).reset_index()

    return agg


# ---------------------------------------------------------------------------
# Step 2: Census ACS data for all CA ZIPs
# ---------------------------------------------------------------------------

def fetch_census_ca() -> pd.DataFrame:
    print("[2/5] Fetching Census ACS data for CA ZIPs...")
    params = {
        "get": "B19013_001E,B25001_001E,B25003_002E,B01003_001E",
        "for": "zip code tabulation area:*",
    }
    r = requests.get(CENSUS_ACS_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    df.rename(columns={
        "zip code tabulation area": "zip",
        "B19013_001E": "median_income",
        "B25001_001E": "housing_units",
        "B25003_002E": "owner_occupied",
        "B01003_001E": "population",
    }, inplace=True)

    for col in ["median_income", "housing_units", "owner_occupied", "population"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df = df[df["zip"].between("90000", "96162")].copy()
    df = df[df["housing_units"] > 0].copy()
    df["owner_pct"] = (df["owner_occupied"] / df["housing_units"]).clip(0, 1)
    df["median_income"] = df["median_income"].clip(lower=0)

    print(f"      -> {len(df)} CA ZIPs from Census ACS")
    return df[["zip", "median_income", "housing_units", "owner_pct", "population"]]


# ---------------------------------------------------------------------------
# Step 3: ZIP centroids (lat/lon)
# ---------------------------------------------------------------------------

def fetch_zip_centroids(ca_zips: list) -> pd.DataFrame:
    print("[3/5] Fetching ZIP centroids from Zippopotam...")
    rows = []
    failed = 0
    for i, z in enumerate(ca_zips):
        try:
            r = requests.get(f"https://api.zippopotam.us/us/{z}", timeout=8)
            if r.status_code == 200:
                d = r.json()
                place = d["places"][0]
                rows.append({
                    "zip": z,
                    "city": place["place name"],
                    "lat": float(place["latitude"]),
                    "lon": float(place["longitude"]),
                })
            else:
                failed += 1
        except Exception:
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"      -> {i+1}/{len(ca_zips)} ZIPs fetched ({failed} failed)...")
            time.sleep(0.2)

    print(f"      -> {len(rows)} centroids retrieved, {failed} failed")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: NREL irradiance sample + Ridge interpolation
# ---------------------------------------------------------------------------

def fetch_nrel_sample(zip_centroids: pd.DataFrame, nrel_key: str) -> pd.DataFrame:
    print(f"[4/5] Sampling NREL irradiance for {NREL_SAMPLE_N} ZIPs...")
    df = zip_centroids.dropna(subset=["lat", "lon"]).copy()

    df["lat_bin"] = pd.cut(df["lat"], bins=10, labels=False)
    df["lon_bin"] = pd.cut(df["lon"], bins=10, labels=False)
    sample = (df.groupby(["lat_bin", "lon_bin"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), 3), random_state=42)))
    sample = sample.sample(min(NREL_SAMPLE_N, len(sample)), random_state=42)

    solrad_rows = []
    failed = 0
    for i, row in enumerate(sample.itertuples()):
        try:
            out = _pvwatts_call(row.lat, row.lon, 1.0, nrel_key)
            solrad_rows.append({"zip": row.zip, "solrad_annual": float(out["solrad_annual"]), "solrad_source": "nrel_actual"})
        except Exception:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"      -> {i+1}/{len(sample)} NREL calls done ({failed} failed)...")
            time.sleep(0.5)

    nrel_df = pd.DataFrame(solrad_rows)
    print(f"      -> {len(nrel_df)} successful NREL calls")

    train = df.merge(nrel_df, on="zip")
    X_train = train[["lat", "lon"]].values
    y_train = train["solrad_annual"].values

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_train)
    ridge = Ridge(alpha=1.0).fit(X_poly, y_train)

    X_all = poly.transform(df[["lat", "lon"]].values)
    df["solrad_predicted"] = ridge.predict(X_all).clip(3.0, 8.5)

    df = df.merge(nrel_df[["zip", "solrad_annual", "solrad_source"]], on="zip", how="left")
    mask = df["solrad_annual"].isna()
    df.loc[mask, "solrad_annual"] = df.loc[mask, "solrad_predicted"]
    df.loc[mask, "solrad_source"] = "ridge_interpolated"

    r2 = ridge.score(X_poly, y_train)
    print(f"      -> Ridge R2 on training ZIPs: {r2:.3f}")
    return df[["zip", "solrad_annual", "solrad_source"]]


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nrel-key", default=os.getenv("NREL_KEY"))
    p.add_argument("--eia-key",  default=os.getenv("EIA_KEY"))
    args = p.parse_args()

    if not args.nrel_key:
        sys.exit("ERROR: --nrel-key or NREL_KEY env var required")

    lbnl_meta = lbnl_supplemental()
    census = fetch_census_ca()

    ca_zips = sorted(census["zip"].unique().tolist())
    centroids = fetch_zip_centroids(ca_zips)

    records = aggregate_records(centroids)
    solrad = fetch_nrel_sample(centroids, args.nrel_key)

    print("[5/5] Joining all data sources...")
    df = census.merge(centroids, on="zip", how="left")
    df = df.merge(solrad, on="zip", how="left")
    df = df.merge(records, on="zip", how="left")
    df = df.merge(lbnl_meta, left_on="zip", right_on="zip_code", how="left")
    df.drop(columns=["zip_code"], errors="ignore", inplace=True)

    df["install_count"] = df["install_count"].fillna(0).astype(int)
    df["adoption_rate"] = (df["install_count"] / df["housing_units"] * 1000).round(4)
    df["pct_third_party"] = df["pct_third_party"].fillna(0)
    df["median_system_kw"] = df["median_system_kw"].fillna(df["median_system_kw"].median())
    df["median_cost_per_watt"] = df["median_cost_per_watt"].fillna(df["median_cost_per_watt"].median())

    df["electricity_rate"] = CA_EIA_RATE
    df["nem_type"] = CA_NEM_TYPE
    df["nem_export_pct"] = CA_NEM_EXPORT_PCT

    df = df.dropna(subset=["lat", "lon", "solrad_annual", "median_income"])

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_CSV.name}")
    print(f"  ZIPs with install data:  {(df['install_count'] > 0).sum()}")
    print(f"  ZIPs with zero installs: {(df['install_count'] == 0).sum()}")
    print(f"  Median adoption rate:    {df['adoption_rate'].median():.2f} per 1,000 units")


if __name__ == "__main__":
    main()
