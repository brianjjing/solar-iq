"""
SolarIQ Pipeline - Residential Solar ROI Analyzer
Usage: python pipeline.py --zip 92037 [--eia-key KEY] [--nrel-key KEY] [--openei-key KEY]
Keys also accepted via env vars: EIA_KEY, NREL_KEY, OPENEI_KEY (or .env file)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.optimize import brentq

load_dotenv()

# ---------------------------------------------------------------------------
# Hardcoded policy tables
# ---------------------------------------------------------------------------

FEDERAL_ITC = {
    **{y: 0.30 for y in range(2022, 2033)},
    2033: 0.26,
    2034: 0.22,
}

NEM_POLICIES = {
    "CA": {"type": "NEM3",   "export_pct": 0.20, "note": "CA NEM 3.0 - export at ~avoided cost (~$0.05/kWh)"},
    "NY": {"type": "VDER",   "export_pct": 0.85, "note": "NY Value Stack - ~85% retail"},
    "TX": {"type": "varies", "export_pct": 0.50, "note": "Deregulated market - varies by utility"},
    "FL": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM through 2029"},
    "NJ": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "AZ": {"type": "NEM",    "export_pct": 0.75, "note": "Excess credited at avoided cost"},
    "CO": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "MA": {"type": "SMART",  "export_pct": 1.10, "note": "MA SMART program - premium above retail"},
    "WA": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "OR": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "IL": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "MD": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "VA": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "NC": {"type": "NEM",    "export_pct": 1.00, "note": "Full retail NEM"},
    "DEFAULT": {"type": "NEM", "export_pct": 0.85, "note": "Assumed ~85% retail NEM (state policy not in table)"},
}

LBNL_CSV = Path(__file__).parent / "TTS_LBNL_public_file_29-Sep-2025_all.csv"
NATIONAL_FALLBACK_COST_PER_WATT = 2.95
NATIONAL_FALLBACK_REBATE = 0.0
SYSTEM_DERATE = 0.80
MAX_SYSTEM_KW = 20.0
PROJECTION_YEARS = 25
DISCOUNT_RATE = 0.04


# ---------------------------------------------------------------------------
# Stage 1: Location
# ---------------------------------------------------------------------------

def fetch_location(zip_code: str) -> dict:
    url = f"https://api.zippopotam.us/us/{zip_code}"
    r = requests.get(url, timeout=10)
    if r.status_code == 404:
        sys.exit(f"ERROR: ZIP code {zip_code} not found in Zippopotam.")
    r.raise_for_status()
    data = r.json()
    place = data["places"][0]
    return {
        "zip": zip_code,
        "city": place["place name"],
        "state": place["state abbreviation"],
        "latitude": float(place["latitude"]),
        "longitude": float(place["longitude"]),
    }


# ---------------------------------------------------------------------------
# Stage 2: EIA Electricity Prices
# ---------------------------------------------------------------------------

def fetch_eia_data(state: str, api_key: str) -> dict:
    url = "https://api.eia.gov/v2/electricity/retail-sales/data/"
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[]": ["price", "sales", "customers"],
        "facets[sectorid][]": "RES",
        "facets[stateid][]": state,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 60,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    rows = r.json().get("response", {}).get("data", [])
    if not rows:
        sys.exit(f"ERROR: EIA returned no data for state {state}. Check your EIA_KEY.")

    df = pd.DataFrame(rows)
    df["period"] = pd.to_datetime(df["period"], format="%Y-%m")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["customers"] = pd.to_numeric(df["customers"], errors="coerce")
    df = df.dropna(subset=["price"]).sort_values("period", ascending=False)

    current_rate_cents = float(df.iloc[0]["price"])
    current_rate = current_rate_cents / 100.0

    last_12 = df.head(12).copy()
    last_12["monthly_kwh"] = (last_12["sales"] * 1_000_000) / last_12["customers"]
    avg_monthly_kwh = float(last_12["monthly_kwh"].mean())

    df["year"] = df["period"].dt.year
    annual = df.groupby("year")["price"].mean().sort_index()
    if len(annual) >= 2:
        years_span = annual.index[-1] - annual.index[0]
        if years_span > 0:
            escalation = (annual.iloc[-1] / annual.iloc[0]) ** (1.0 / years_span) - 1.0
        else:
            escalation = 0.025
    else:
        escalation = 0.025

    return {
        "current_rate_cents_kwh": current_rate_cents,
        "current_rate": current_rate,
        "avg_monthly_kwh": avg_monthly_kwh,
        "escalation_rate": max(0.0, escalation),
        "source": "EIA retail-sales API (RES sector)",
        "period_latest": str(df.iloc[0]["period"].date()),
        "data_points": len(df),
    }


# ---------------------------------------------------------------------------
# Stage 3: NREL PVWatts (two-pass: probe for sun hours, then size + produce)
# ---------------------------------------------------------------------------

def _pvwatts_call(lat: float, lon: float, system_kw: float, api_key: str) -> dict:
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {
        "api_key": api_key,
        "lat": lat,
        "lon": lon,
        "system_capacity": system_kw,
        "module_type": 1,
        "losses": 14,
        "array_type": 1,
        "tilt": 20,
        "azimuth": 180,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()
    if "errors" in body and body["errors"]:
        sys.exit(f"NREL PVWatts error: {body['errors']}")
    return body["outputs"]


def size_system(avg_monthly_kwh: float, solrad_annual: float) -> float:
    annual_kwh = avg_monthly_kwh * 12
    raw_kw = annual_kwh / (365 * solrad_annual * SYSTEM_DERATE)
    sized = round(raw_kw * 10) / 10
    return min(sized, MAX_SYSTEM_KW)


def fetch_nrel_data(lat: float, lon: float, avg_monthly_kwh: float, api_key: str) -> dict:
    probe = _pvwatts_call(lat, lon, 1.0, api_key)
    solrad_annual = float(probe["solrad_annual"])

    system_kw = size_system(avg_monthly_kwh, solrad_annual)

    outputs = _pvwatts_call(lat, lon, system_kw, api_key)
    return {
        "system_size_kw": system_kw,
        "ac_annual_kwh": float(outputs["ac_annual"]),
        "solrad_annual": solrad_annual,
        "capacity_factor": float(outputs["capacity_factor"]),
        "source": "NREL PVWatts v8",
        "params": {"module_type": "Premium", "array_type": "Fixed roof", "tilt": 20, "azimuth": 180, "losses_pct": 14},
    }


# ---------------------------------------------------------------------------
# Stage 4: OpenEI Utility Rate Database
# ---------------------------------------------------------------------------

def fetch_openei_rate(zip_code: str, api_key: str, eia_rate_fallback: float) -> dict:
    url = "https://api.openei.org/utility_rates"
    params = {
        "version": 7,
        "format": "json",
        "api_key": api_key,
        "address": zip_code,
        "sector": "Residential",
        "limit": 3,
        "approved": "true",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as e:
        print(f"  [WARN] OpenEI request failed ({e}); falling back to EIA rate.")
        items = []

    for item in items:
        utility_name = item.get("utility", "Unknown utility")
        energy_structure = item.get("energyratestructure", [])
        flat_rate = None

        for period in energy_structure:
            for tier in period:
                rate_val = tier.get("rate")
                if rate_val is not None:
                    flat_rate = float(rate_val)
                    break
            if flat_rate is not None:
                break

        if flat_rate and flat_rate > 0.01:
            return {
                "retail_rate": flat_rate,
                "utility_name": utility_name,
                "source": "OpenEI URDB",
                "guid": item.get("guid", ""),
            }

    return {
        "retail_rate": eia_rate_fallback,
        "utility_name": "Unknown (OpenEI returned no flat rate)",
        "source": "EIA (OpenEI fallback)",
        "guid": "",
    }


# ---------------------------------------------------------------------------
# Stage 5: LBNL Cost Benchmarks
# ---------------------------------------------------------------------------

def load_lbnl_benchmarks(state: str) -> dict:
    df = pd.read_csv(
        LBNL_CSV,
        usecols=["state", "customer_segment", "installation_date",
                 "PV_system_size_DC", "total_installed_price", "rebate_or_grant"],
        low_memory=False,
    )
    df = df[df["customer_segment"].isin(["RES", "RES_SF", "RES_MF"])]
    df["installation_date"] = pd.to_datetime(df["installation_date"], format="mixed", errors="coerce", dayfirst=False)

    # Dataset covers installs through ~2020; use most recent 5 years available
    max_date = df["installation_date"].max()
    cutoff = max_date - pd.DateOffset(years=5)
    df = df[df["installation_date"] >= cutoff]

    df["total_installed_price"] = pd.to_numeric(df["total_installed_price"], errors="coerce")
    df["PV_system_size_DC"] = pd.to_numeric(df["PV_system_size_DC"], errors="coerce")
    df["rebate_or_grant"] = pd.to_numeric(df["rebate_or_grant"], errors="coerce")
    df = df.dropna(subset=["total_installed_price", "PV_system_size_DC"])
    df = df[df["PV_system_size_DC"] > 0]
    df["cost_per_watt"] = df["total_installed_price"] / (df["PV_system_size_DC"] * 1000)
    df = df[(df["cost_per_watt"] > 0.5) & (df["cost_per_watt"] < 15)]

    state_df = df[df["state"].str.upper() == state.upper()]
    scope = state
    used_hardcoded = False

    if len(state_df) < 10:
        # Try whole dataset as a proxy
        if len(df) >= 10:
            print(f"  [WARN] Only {len(state_df)} LBNL installs for {state}. Using dataset-wide median as proxy.")
            state_df = df
            scope = "dataset-wide proxy"
        else:
            # Dataset has no usable rows - fall back to hardcoded national benchmarks
            print(f"  [WARN] LBNL dataset insufficient for {state}. Using hardcoded national benchmark.")
            used_hardcoded = True

    if used_hardcoded:
        return {
            "median_cost_per_watt": NATIONAL_FALLBACK_COST_PER_WATT,
            "median_rebate": NATIONAL_FALLBACK_REBATE,
            "sample_n": 0,
            "scope": "national hardcoded benchmark",
            "source": "LBNL TTS (insufficient data) - national avg $2.95/W (NREL 2023)",
        }

    median_cpw = float(state_df["cost_per_watt"].median())
    median_rebate = float(state_df["rebate_or_grant"].dropna().median()) if state_df["rebate_or_grant"].notna().any() else 0.0
    if pd.isna(median_rebate):
        median_rebate = 0.0

    return {
        "median_cost_per_watt": median_cpw,
        "median_rebate": median_rebate,
        "sample_n": len(state_df),
        "scope": scope,
        "source": "LBNL Tracking the Sun (Sep 2025)",
    }


# ---------------------------------------------------------------------------
# ROI Calculation
# ---------------------------------------------------------------------------

def calculate_roi(
    system_kw: float,
    ac_annual_kwh: float,
    lbnl: dict,
    retail_rate: float,
    escalation_rate: float,
    nem: dict,
    itc_rate: float,
    install_year: int,
) -> dict:
    gross_cost = system_kw * 1000 * lbnl["median_cost_per_watt"]
    itc_credit = gross_cost * itc_rate
    rebate = lbnl["median_rebate"]
    net_cost = gross_cost - itc_credit - rebate

    nem_pct = nem["export_pct"]

    rows = []
    cumulative_savings = 0.0
    payback_year = None

    for t in range(1, PROJECTION_YEARS + 1):
        rate_t = retail_rate * ((1 + escalation_rate) ** t)
        savings_t = ac_annual_kwh * rate_t * nem_pct
        cumulative_savings += savings_t
        net_position = cumulative_savings - net_cost
        if payback_year is None and net_position >= 0:
            payback_year = install_year + t
        rows.append({
            "year": install_year + t,
            "year_number": t,
            "rate_per_kwh": round(rate_t, 4),
            "annual_savings": round(savings_t, 2),
            "cumulative_savings": round(cumulative_savings, 2),
            "net_position": round(net_position, 2),
        })

    npv = -net_cost + sum(r["annual_savings"] / ((1 + DISCOUNT_RATE) ** r["year_number"]) for r in rows)

    cash_flows = [-net_cost] + [r["annual_savings"] for r in rows]
    try:
        irr = _compute_irr(cash_flows)
    except Exception:
        irr = None

    return {
        "gross_install_cost": round(gross_cost, 2),
        "itc_credit": round(itc_credit, 2),
        "itc_rate": itc_rate,
        "lbnl_rebate": round(rebate, 2),
        "net_cost": round(net_cost, 2),
        "payback_year": payback_year,
        "payback_years": (payback_year - install_year) if payback_year else None,
        "npv_at_4pct": round(npv, 2),
        "irr": round(irr * 100, 2) if irr is not None else None,
        "projection": rows,
    }


def _compute_irr(cash_flows: list) -> float:
    def npv_at_rate(r):
        return sum(cf / (1 + r) ** i for i, cf in enumerate(cash_flows))

    try:
        return brentq(npv_at_rate, -0.999, 10.0, xtol=1e-6)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

HIGHLIGHT_YEARS = {1, 5, 10, 15, 20, 25}


def print_report(loc: dict, eia: dict, nrel: dict, openei: dict, lbnl: dict, nem: dict, roi: dict, install_year: int):
    z = loc["zip"]
    city = loc["city"]
    state = loc["state"]
    itc_pct = int(roi["itc_rate"] * 100)

    print()
    print(f"{'=' * 62}")
    print(f"  SolarIQ Report: {z}  ({city}, {state})")
    print(f"{'=' * 62}")
    print(f"  Data sources: EIA, NREL PVWatts v8, OpenEI URDB, LBNL TTS")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    print("  --- System ---")
    print(f"  Avg monthly usage (EIA {state} residential): {eia['avg_monthly_kwh']:,.0f} kWh/mo")
    print(f"  Peak sun hours (NREL):                       {nrel['solrad_annual']:.2f} hrs/day")
    print(f"  Recommended system size:                     {nrel['system_size_kw']:.1f} kW")
    print(f"  Est. annual production (PVWatts):            {nrel['ac_annual_kwh']:,.0f} kWh/yr")
    print(f"  Capacity factor:                             {nrel['capacity_factor']:.1f}%")
    print()

    print("  --- Cost & Incentives ---")
    print(f"  Gross install cost  [{lbnl['source']}, ${lbnl['median_cost_per_watt']:.2f}/W, n={lbnl['sample_n']:,} installs ({lbnl['scope']})]")
    print(f"    {nrel['system_size_kw']:.1f} kW x {lbnl['median_cost_per_watt']:.2f} $/W x 1000:   ${roi['gross_install_cost']:>10,.0f}")
    print(f"  Federal ITC ({itc_pct}%  - IRA through 2032):      -${roi['itc_credit']:>9,.0f}")
    print(f"  State/local rebate (LBNL median):           -${roi['lbnl_rebate']:>9,.0f}")
    print(f"  {'-' * 54}")
    print(f"  Net cost after incentives:                   ${roi['net_cost']:>10,.0f}")
    print()

    print("  --- Electricity Rate ---")
    print(f"  Current retail rate [{openei['source']} / {openei['utility_name']}]")
    print(f"    Rate:                                      ${openei['retail_rate']:.4f}/kWh")
    print(f"  5-yr price escalation (EIA CAGR):            {eia['escalation_rate']*100:.2f}%/yr")
    print(f"  NEM policy ({state}): {nem['type']}  - export at {int(nem['export_pct']*100)}% retail")
    print(f"    Note: {nem['note']}")
    print()

    print("  --- 25-Year ROI Projection ---")
    print(f"  {'Year':<6} {'Ann. Savings':>12} {'Cum. Savings':>13} {'Net Position':>13}")
    print(f"  {'-'*6} {'-'*12} {'-'*13} {'-'*13}")

    pb = roi["payback_year"]
    pb_year_num = roi["payback_years"]
    display_year_nums = HIGHLIGHT_YEARS | ({pb_year_num} if pb_year_num else set())
    for row in roi["projection"]:
        if row["year_number"] not in display_year_nums:
            continue
        marker = "  <- PAYBACK" if pb and row["year"] == pb else ""
        print(
            f"  {row['year']:<6}"
            f"  ${row['annual_savings']:>10,.0f}"
            f"  ${row['cumulative_savings']:>11,.0f}"
            f"  ${row['net_position']:>11,.0f}"
            f"{marker}"
        )

    print()
    if roi["payback_years"]:
        print(f"  Payback period:   {roi['payback_years']} years  (break-even in {roi['payback_year']})")
    else:
        print(f"  Payback period:   >25 years (does not break even within projection window)")
    print(f"  25-yr NPV (@{int(DISCOUNT_RATE*100)}%): ${roi['npv_at_4pct']:,.0f}")
    if roi["irr"] is not None:
        print(f"  IRR:              {roi['irr']:.1f}%")
    print()


def save_json(loc, eia, nrel, openei, lbnl, nem, roi, install_year, keys_used):
    output = {
        "generated": datetime.now().isoformat(),
        "install_year": install_year,
        "inputs": {"zip": loc["zip"]},
        "location": loc,
        "eia": eia,
        "nrel": nrel,
        "openei": openei,
        "lbnl": lbnl,
        "nem_policy": nem,
        "roi": roi,
        "assumptions": {
            "system_derate": SYSTEM_DERATE,
            "max_system_kw": MAX_SYSTEM_KW,
            "projection_years": PROJECTION_YEARS,
            "discount_rate": DISCOUNT_RATE,
            "itc_rate": roi["itc_rate"],
            "itc_source": "IRA 2022 - Federal residential solar ITC schedule",
            "nem_source": "Hardcoded by state; see NEM_POLICIES in pipeline.py",
            "savings_formula": "ac_annual_kwh x rate_t x nem_export_pct  (all production modeled as NEM credit)",
            "rate_projection": "retail_rate x (1 + escalation_rate)^t",
        },
        "data_sources": {
            "location": "Zippopotam (api.zippopotam.us)",
            "electricity_price": "EIA Open Data API v2 - retail-sales, RES sector",
            "avg_monthly_kwh": "EIA retail-sales (MWh sales / customers, last 12 months)",
            "price_escalation": "EIA 5-year CAGR of annual average residential rate",
            "solar_resource": "NREL PVWatts v8",
            "utility_rate": openei["source"],
            "install_cost": "LBNL Tracking the Sun (Sep 2025)",
            "incentives_federal": "IRS / IRA - hardcoded schedule",
            "incentives_nem": "State NEM policies - hardcoded table",
        },
    }
    out_path = Path(__file__).parent / f"solar_iq_{loc['zip']}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Report saved to: {out_path.name}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SolarIQ - Residential Solar ROI Analyzer")
    p.add_argument("--zip", required=True, help="5-digit US ZIP code")
    p.add_argument("--eia-key",    default=os.getenv("EIA_KEY"),    help="EIA API key (or set EIA_KEY env var)")
    p.add_argument("--nrel-key",   default=os.getenv("NREL_KEY"),   help="NREL API key (or set NREL_KEY env var)")
    p.add_argument("--openei-key", default=os.getenv("OPENEI_KEY"), help="OpenEI API key (or set OPENEI_KEY env var)")
    return p.parse_args()


def main():
    args = parse_args()

    missing = [name for name, val in [("EIA_KEY", args.eia_key), ("NREL_KEY", args.nrel_key), ("OPENEI_KEY", args.openei_key)] if not val]
    if missing:
        sys.exit(f"ERROR: Missing API keys: {', '.join(missing)}. Pass via --flag or env var.")

    install_year = datetime.now().year
    itc_rate = FEDERAL_ITC.get(install_year, 0.0)

    print(f"\n[1/5] Fetching location for ZIP {args.zip}...")
    loc = fetch_location(args.zip)
    print(f"      -> {loc['city']}, {loc['state']}  ({loc['latitude']:.4f}, {loc['longitude']:.4f})")

    print(f"[2/5] Fetching EIA electricity data for {loc['state']}...")
    eia = fetch_eia_data(loc["state"], args.eia_key)
    print(f"      -> Rate: {eia['current_rate_cents_kwh']:.2f}c/kWh | Avg usage: {eia['avg_monthly_kwh']:,.0f} kWh/mo | Escalation: {eia['escalation_rate']*100:.2f}%/yr")

    print(f"[3/5] Fetching NREL PVWatts solar data...")
    nrel = fetch_nrel_data(loc["latitude"], loc["longitude"], eia["avg_monthly_kwh"], args.nrel_key)
    print(f"      -> {nrel['system_size_kw']:.1f} kW system | {nrel['ac_annual_kwh']:,.0f} kWh/yr | {nrel['solrad_annual']:.2f} sun-hrs/day")

    print(f"[4/5] Fetching OpenEI utility rate for {args.zip}...")
    openei = fetch_openei_rate(args.zip, args.openei_key, eia["current_rate"])
    print(f"      -> {openei['utility_name']}: ${openei['retail_rate']:.4f}/kWh  [{openei['source']}]")

    print(f"[5/5] Loading LBNL cost benchmarks for {loc['state']}...")
    lbnl = load_lbnl_benchmarks(loc["state"])
    print(f"      -> ${lbnl['median_cost_per_watt']:.2f}/W | Rebate: ${lbnl['median_rebate']:,.0f} | n={lbnl['sample_n']:,} installs ({lbnl['scope']})")

    nem = NEM_POLICIES.get(loc["state"], NEM_POLICIES["DEFAULT"])
    roi = calculate_roi(
        system_kw=nrel["system_size_kw"],
        ac_annual_kwh=nrel["ac_annual_kwh"],
        lbnl=lbnl,
        retail_rate=openei["retail_rate"],
        escalation_rate=eia["escalation_rate"],
        nem=nem,
        itc_rate=itc_rate,
        install_year=install_year,
    )

    print_report(loc, eia, nrel, openei, lbnl, nem, roi, install_year)
    save_json(loc, eia, nrel, openei, lbnl, nem, roi, install_year, {
        "eia_key": bool(args.eia_key),
        "nrel_key": bool(args.nrel_key),
        "openei_key": bool(args.openei_key),
    })


if __name__ == "__main__":
    main()
