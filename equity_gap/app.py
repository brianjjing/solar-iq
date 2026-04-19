"""
SolarIQ: Solar Equity Gap Scanner - Streamlit Web App
Run: streamlit run equity_gap/app.py
Requires: predictions.csv, model.pkl, roi_cache.pkl in project root
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PRED_CSV    = ROOT / "predictions.csv"
MODEL_PKL   = ROOT / "model.pkl"
ROI_CACHE   = ROOT / "roi_cache.pkl"
GEOJSON_PATH = ROOT / "equity_gap" / "ca_zips.geojson"

CA_CO2_LBS_PER_KWH = 0.476
CARS_CO2_TONS = 4.6

FEATURE_LABELS = {
    "median_income": "Median Household Income",
    "solrad_annual": "Solar Resource (sun-hrs/day)",
    "owner_pct": "Homeownership Rate",
    "housing_units": "Housing Units",
    "electricity_rate": "Electricity Rate ($/kWh)",
    "pct_third_party": "Third-Party Ownership Availability",
}

st.set_page_config(
    page_title="SolarIQ: Solar Equity Gap Scanner",
    page_icon="☀",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading (all cached — loaded once, reused across reruns)
# ---------------------------------------------------------------------------

@st.cache_data
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(PRED_CSV)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df["median_income_fmt"] = df["median_income"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    df["adoption_fmt"] = df["actual_adoption"].apply(lambda x: f"{x:.2f}")
    df["predicted_fmt"] = df["predicted_adoption"].apply(lambda x: f"{x:.2f}")
    df["gap_fmt"] = df["gap_score"].apply(lambda x: f"{x:+.2f}")
    df["co2_fmt"] = df["missed_co2_tons_per_year"].apply(lambda x: f"{x:,.1f}")
    return df


@st.cache_data
def load_shap_importance() -> dict:
    with open(MODEL_PKL, "rb") as f:
        data = pickle.load(f)
    return data.get("shap_importance", {})


@st.cache_data
def load_roi_cache() -> dict:
    if not ROI_CACHE.exists():
        return {}
    with open(ROI_CACHE, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_geojson():
    if GEOJSON_PATH.exists():
        with open(GEOJSON_PATH) as f:
            return json.load(f)
    try:
        url = "https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ca_california_zip_codes_geo.min.json"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        gj = r.json()
        with open(GEOJSON_PATH, "w") as f:
            json.dump(gj, f)
        return gj
    except Exception as e:
        st.warning(f"Could not load ZIP GeoJSON: {e}. Map will use scatter plot instead.")
        return None


def compute_headline(df: pd.DataFrame, n: int = 100) -> tuple:
    top = df.nlargest(n, "gap_score")
    total_co2 = top["missed_co2_tons_per_year"].sum()
    return total_co2, total_co2 / CARS_CO2_TONS


# ---------------------------------------------------------------------------
# Map — cached so it doesn't rebuild on every text-input keystroke
# ---------------------------------------------------------------------------

@st.cache_data
def build_map(df: pd.DataFrame, _geojson) -> go.Figure:
    """Cache key is the DataFrame hash; _geojson prefix skips hashing the large dict."""
    if _geojson:
        fig = px.choropleth_mapbox(
            df,
            geojson=_geojson,
            locations="zip",
            featureidkey="properties.ZCTA5CE10",
            color="gap_score_normalized",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            mapbox_style="carto-positron",
            center={"lat": 36.7, "lon": -119.4},
            zoom=5,
            opacity=0.7,
            hover_data={
                "zip": True,
                "city": True,
                "gap_score_normalized": ":.1f",
                "median_income_fmt": True,
                "adoption_fmt": True,
                "co2_fmt": True,
            },
            labels={
                "gap_score_normalized": "Equity Gap Score",
                "median_income_fmt": "Median Income",
                "adoption_fmt": "Actual Adoption (per 1k units)",
                "co2_fmt": "Missed CO2 (tons/yr)",
            },
        )
    else:
        fig = px.scatter_mapbox(
            df,
            lat="lat", lon="lon",
            color="gap_score_normalized",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            size_max=8,
            size=[6] * len(df),
            mapbox_style="carto-positron",
            center={"lat": 36.7, "lon": -119.4},
            zoom=5,
            opacity=0.75,
            hover_data={"zip": True, "city": True, "gap_score_normalized": ":.1f"},
        )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title="Equity Gap<br>Score",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0<br>(Served)", "25", "50", "75", "100<br>(Underserved)"],
        ),
        height=520,
    )
    return fig


def build_shap_chart(shap_importance: dict) -> go.Figure:
    total = sum(shap_importance.values())
    items = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    pcts = [v / total * 100 for _, v in items]

    fig = go.Figure(go.Bar(
        x=pcts,
        y=labels,
        orientation="h",
        marker_color=["#e74c3c" if p == max(pcts) else "#3498db" for p in pcts],
        text=[f"{p:.1f}%" for p in pcts],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="% Contribution to Predicted Adoption",
        yaxis=dict(autorange="reversed"),
        margin={"l": 20, "r": 60, "t": 10, "b": 30},
        height=260,
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
    )
    return fig


def build_detail_panel(row: pd.Series, shap_cols: list, roi: dict | None):
    gap = row["gap_score"]
    severity = "High" if row["gap_score_normalized"] > 66 else ("Medium" if row["gap_score_normalized"] > 33 else "Low")
    color = "#e74c3c" if severity == "High" else ("#f39c12" if severity == "Medium" else "#27ae60")

    st.markdown(f"### ZIP {row['zip']} — {row.get('city', 'CA')}")
    st.markdown(
        f"**Equity Gap: <span style='color:{color}'>{severity} ({row['gap_score_normalized']:.0f}/100)</span>**",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Actual Adoption", f"{row['actual_adoption']:.2f}",
                  help="Solar installs per 1,000 housing units")
        st.metric("Median Income", row["median_income_fmt"])
        st.metric("Homeownership", f"{row['owner_pct']*100:.1f}%")
    with col2:
        st.metric("Predicted Adoption", f"{row['predicted_adoption']:.2f}",
                  delta=f"{gap:+.2f} gap", delta_color="inverse")
        st.metric("Solar Resource", f"{row['solrad_annual']:.2f} hrs/day")
        st.metric("Housing Units", f"{row['housing_units']:,.0f}")

    st.markdown("---")
    st.markdown("**Estimated Carbon Impact if Gap Closed**")
    st.metric(
        "Missed CO2 Offset",
        f"{row['missed_co2_tons_per_year']:,.1f} tons/yr",
    )
    cars_eq = row["missed_co2_tons_per_year"] / CARS_CO2_TONS
    st.caption(f"Equivalent to {cars_eq:,.0f} additional cars on the road each year.")

    # ROI panel
    if roi:
        st.markdown("---")
        st.markdown("**25-Year Solar ROI**")
        r1, r2 = st.columns(2)
        with r1:
            st.metric("System Size", f"{roi['system_kw']:.1f} kW")
            st.metric("Net Install Cost", f"${roi['net_cost']:,.0f}",
                      help=f"Gross ${roi['gross_cost']:,.0f} - ITC ${roi['itc_credit']:,.0f}")
            pb = roi["payback_years"]
            st.metric("Payback Period", f"{pb} yrs" if pb else ">25 yrs")
        with r2:
            st.metric("Est. Annual Production", f"{roi['ac_annual_kwh']:,} kWh/yr")
            st.metric("25-yr NPV (4%)", f"${roi['npv_25yr']:,.0f}")
            irr = roi["irr"]
            st.metric("IRR", f"{irr:.1f}%" if irr is not None else "N/A")
        yr1 = roi.get("annual_savings_yr1")
        if yr1:
            st.caption(f"Est. year-1 savings: ${yr1:,.0f}  |  Rate: ${roi['electricity_rate']:.4f}/kWh  |  CA NEM 3.0")

    if shap_cols:
        st.markdown("---")
        st.markdown("**Top drivers of this ZIP's gap (SHAP)**")
        shap_data = [
            (FEATURE_LABELS.get(c.replace("shap_", ""), c), row[c])
            for c in shap_cols if c in row.index
        ]
        shap_data.sort(key=lambda x: abs(x[1]), reverse=True)
        for feat_label, val in shap_data[:4]:
            direction = "increases" if val > 0 else "decreases"
            st.caption(f"- **{feat_label}** {direction} predicted adoption ({val:+.3f})")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.markdown("""
    <style>
    .main-title { font-size: 2rem; font-weight: 700; color: #2c3e50; }
    .subtitle { color: #666; font-size: 1.05rem; margin-bottom: 1rem; }
    .headline-box { background: #fff3cd; border-left: 4px solid #f39c12;
                    padding: 0.8rem 1rem; border-radius: 4px; margin: 0.5rem 0 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">SolarIQ: Solar Equity Gap Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Identifying California communities left out of the clean energy transition.</div>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        try:
            df = load_predictions()
        except FileNotFoundError:
            st.error("predictions.csv not found. Run collect_data.py then train_model.py first.")
            st.stop()

        try:
            shap_importance = load_shap_importance()
        except FileNotFoundError:
            shap_importance = {}

        roi_cache = load_roi_cache()
        geojson = load_geojson()

    shap_cols = [c for c in df.columns if c.startswith("shap_")]

    total_co2, cars = compute_headline(df, 100)
    st.markdown(
        f'<div class="headline-box">Closing the top-100 solar equity gaps in California would offset '
        f'<strong>{total_co2:,.0f} tons of CO2 per year</strong> — equivalent to removing '
        f'<strong>{cars:,.0f} cars</strong> from the road.</div>',
        unsafe_allow_html=True,
    )

    map_col, detail_col = st.columns([3, 1.4])

    with map_col:
        st.markdown("**Click any ZIP code to see details**")
        # build_map is @st.cache_data — rebuilds only when df changes, not on every keystroke
        fig = build_map(df, geojson)
        click_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="map")

    with detail_col:
        selected_zip = st.text_input("Or enter a ZIP code:", placeholder="e.g. 91331", max_chars=5)

        clicked_zip = None
        if hasattr(click_data, "selection") and click_data.selection and click_data.selection.get("points"):
            pt = click_data.selection["points"][0]
            clicked_zip = pt.get("location") or pt.get("customdata", [None])[0]

        active_zip = selected_zip.strip() or clicked_zip

        if active_zip and len(active_zip) == 5:
            match = df[df["zip"] == active_zip.zfill(5)]
            if not match.empty:
                roi = roi_cache.get(active_zip.zfill(5))
                build_detail_panel(match.iloc[0], shap_cols, roi)
            else:
                st.info(f"ZIP {active_zip} not found in dataset.")
        else:
            st.info("Click a ZIP on the map or type one above to see details.")

    st.divider()

    shap_col, table_col = st.columns([1, 1.8])
    with shap_col:
        st.markdown("#### What Drives Solar Adoption?")
        st.caption("Mean absolute SHAP values — higher = more influential in the model's predictions.")
        if shap_importance:
            st.plotly_chart(build_shap_chart(shap_importance), use_container_width=True)
        else:
            st.info("Run train_model.py to generate SHAP data.")

    with table_col:
        st.markdown("#### Top 20 Solar Equity Deserts")
        st.caption("Communities with the highest gap between predicted and actual solar adoption.")
        top20 = df.nlargest(20, "gap_score")[[
            "zip", "city", "median_income_fmt", "adoption_fmt",
            "predicted_fmt", "gap_fmt", "co2_fmt"
        ]].rename(columns={
            "zip": "ZIP",
            "city": "City",
            "median_income_fmt": "Median Income",
            "adoption_fmt": "Actual (per 1k)",
            "predicted_fmt": "Predicted (per 1k)",
            "gap_fmt": "Gap",
            "co2_fmt": "Missed CO2 (tons/yr)",
        })
        st.dataframe(top20, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Data sources: SolarApp+ permit records (CA installs), US Census ACS 5-Year 2022, "
        "NREL PVWatts v8, EIA retail electricity rates, LBNL Tracking the Sun (cost benchmarks). "
        f"CA grid CO2 intensity: {CA_CO2_LBS_PER_KWH} lbs/kWh (EPA eGRID 2022 CAMX). "
        "Model: XGBoost regression with SHAP explainability."
    )


if __name__ == "__main__":
    main()
