"""
SolarIQ: Pre-compute 25-yr solar ROI for every CA ZIP in equity_gap_data.csv.
Reuses solrad_annual already in equity_gap_data.csv — no per-ZIP NREL calls needed.
Makes one EIA call for CA and one LBNL load, then loops all ZIPs offline.

Outputs: ../roi_cache.pkl  (dict: zip_str -> roi_summary dict)
Run: python equity_gap/precompute_roi.py --eia-key KEY
     (or set EIA_KEY in .env)
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import (
    calculate_roi, size_system, fetch_eia_data, load_lbnl_benchmarks,
    FEDERAL_ITC, NEM_POLICIES, SYSTEM_DERATE,
)

load_dotenv(ROOT / ".env")

DATA_CSV  = ROOT / "equity_gap_data.csv"
ROI_CACHE = ROOT / "roi_cache.pkl"
DAYS_PER_YEAR = 365


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eia-key", default=os.getenv("EIA_KEY"))
    args = p.parse_args()

    if not args.eia_key:
        sys.exit("ERROR: --eia-key or EIA_KEY env var required")

    df = pd.read_csv(DATA_CSV)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df = df.dropna(subset=["solrad_annual", "lat", "lon"])
    print(f"Loaded {len(df)} ZIPs from {DATA_CSV.name}")

    # One EIA call for all of CA
    print("\nFetching EIA electricity data for CA (one call for all ZIPs)...")
    eia = fetch_eia_data("CA", args.eia_key)
    print(f"  Rate: {eia['current_rate_cents_kwh']:.2f}c/kWh | "
          f"Avg usage: {eia['avg_monthly_kwh']:,.0f} kWh/mo | "
          f"Escalation: {eia['escalation_rate']*100:.2f}%/yr")

    # One LBNL load for CA cost benchmarks
    print("Loading LBNL cost benchmarks for CA...")
    lbnl_ca = load_lbnl_benchmarks("CA")
    print(f"  ${lbnl_ca['median_cost_per_watt']:.2f}/W | Rebate: ${lbnl_ca['median_rebate']:,.0f} | "
          f"n={lbnl_ca['sample_n']:,} installs")

    nem = NEM_POLICIES["CA"]
    install_year = datetime.now().year
    itc_rate = FEDERAL_ITC.get(install_year, 0.0)
    retail_rate = eia["current_rate"]

    roi_cache = {}
    n_no_payback = 0

    print(f"\nComputing ROI for {len(df)} ZIPs...")
    for i, row in enumerate(df.itertuples(index=False)):
        solrad = float(row.solrad_annual)
        system_kw = size_system(eia["avg_monthly_kwh"], solrad)

        # Annual production: same formula used in train_model.py carbon calc
        ac_annual_kwh = system_kw * solrad * DAYS_PER_YEAR * SYSTEM_DERATE

        # Per-ZIP cost if available from LBNL, else CA-wide median
        cost_per_watt = getattr(row, "median_cost_per_watt", None)
        if cost_per_watt is None or (isinstance(cost_per_watt, float) and np.isnan(cost_per_watt)):
            cost_per_watt = lbnl_ca["median_cost_per_watt"]

        lbnl_zip = {
            "median_cost_per_watt": float(cost_per_watt),
            "median_rebate": lbnl_ca["median_rebate"],
            "sample_n": lbnl_ca["sample_n"],
            "scope": lbnl_ca["scope"],
            "source": lbnl_ca["source"],
        }

        roi = calculate_roi(
            system_kw=system_kw,
            ac_annual_kwh=ac_annual_kwh,
            lbnl=lbnl_zip,
            retail_rate=retail_rate,
            escalation_rate=eia["escalation_rate"],
            nem=nem,
            itc_rate=itc_rate,
            install_year=install_year,
        )

        if roi["payback_years"] is None:
            n_no_payback += 1

        roi_cache[row.zip] = {
            "system_kw": round(system_kw, 1),
            "ac_annual_kwh": round(ac_annual_kwh),
            "solrad_annual": round(solrad, 2),
            "gross_cost": roi["gross_install_cost"],
            "itc_credit": roi["itc_credit"],
            "net_cost": roi["net_cost"],
            "payback_years": roi["payback_years"],
            "payback_year": roi["payback_year"],
            "npv_25yr": roi["npv_at_4pct"],
            "irr": roi["irr"],
            "cost_per_watt": round(float(cost_per_watt), 2),
            "electricity_rate": round(retail_rate, 4),
            "annual_savings_yr1": roi["projection"][0]["annual_savings"] if roi["projection"] else None,
        }

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(df)} ZIPs...")

    with open(ROI_CACHE, "wb") as f:
        pickle.dump(roi_cache, f)

    paybacks = [v["payback_years"] for v in roi_cache.values() if v["payback_years"] is not None]
    print(f"\nSaved {len(roi_cache)} ZIPs to {ROI_CACHE.name}")
    print(f"  ZIPs with payback <= 25yr: {len(paybacks)} ({len(paybacks)/len(roi_cache)*100:.0f}%)")
    print(f"  ZIPs no payback within 25yr: {n_no_payback}")
    if paybacks:
        print(f"  Avg payback: {np.mean(paybacks):.1f} yr | "
              f"Min: {min(paybacks)} yr | Max: {max(paybacks)} yr")


if __name__ == "__main__":
    main()
