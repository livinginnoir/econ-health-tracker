"""
app.py
------
PNW Regional Economic Health Dashboard — Streamlit entry point.

Run locally:
    streamlit run app.py

The app loads data from ``data/processed/all_indicators.csv`` if it exists,
otherwise it fetches live from the FRED API.  All heavy data work is cached
with ``@st.cache_data`` so re-renders are instant.

Layout
------
  Sidebar   : Date range, indicator selector, data controls
  Main      : Header → Snapshot cards → Charts → Raw data table
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PNW Economic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Imports that depend on the src/ package
# ---------------------------------------------------------------------------
from src.config import APP_SUBTITLE, APP_TITLE, FRED_SERIES
from src.data_fetcher import fetch_all_fred, load_processed
from src.chart_helpers import (
    CHART_COLORS,
    compute_snapshot,
    make_line_chart,
)

# ---------------------------------------------------------------------------
# Custom CSS — design tokens applied at the page level
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Typography ─────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ── App background ─────────────────────────────────────────────────── */
    .stApp {
        background-color: #F7F5F0;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #1C1C1E;
    }
    [data-testid="stSidebar"] * {
        color: #E8E4DC !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #F7F5F0 !important;
    }

    /* ── Dashboard title ────────────────────────────────────────────────── */
    .dash-title {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        font-weight: 600;
        color: #1C1C1E;
        line-height: 1.1;
        margin-bottom: 0.1rem;
    }
    .dash-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #888880;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    .last-updated {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #AAAAAA;
        margin-top: 0.25rem;
    }

    /* ── Metric cards ───────────────────────────────────────────────────── */
    .metric-card {
        background: #FFFFFF;
        border-radius: 6px;
        padding: 1.1rem 1.3rem 1rem;
        border-left: 3px solid var(--card-accent, #888);
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.68rem;
        color: #888880;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 600;
        color: #1C1C1E;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .metric-delta {
        font-size: 0.72rem;
        color: #666660;
    }
    .metric-delta.up   { color: #2E7D32; }
    .metric-delta.down { color: #C62828; }

    /* ── Section headers ────────────────────────────────────────────────── */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #AAAAAA;
        border-bottom: 1px solid #E0DDD8;
        padding-bottom: 0.4rem;
        margin: 2rem 0 1rem;
    }

    /* ── Chart container ────────────────────────────────────────────────── */
    .chart-container {
        background: #FFFFFF;
        border-radius: 6px;
        padding: 1.2rem 1rem 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    .chart-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.1rem;
    }

    /* ── Placeholder (Phase 3) ─────────────────────────────────────────── */
    .phase3-placeholder {
        background: #F0EDE8;
        border: 1.5px dashed #CCCCCC;
        border-radius: 6px;
        padding: 1.5rem;
        text-align: center;
        color: #AAAAAA;
        font-size: 0.75rem;
        letter-spacing: 0.06em;
    }

    /* ── Streamlit overrides ────────────────────────────────────────────── */
    div[data-testid="stMetric"] { display: none; }   /* hide native metrics */
    .stPlotlyChart { border-radius: 6px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

PROCESSED_FILE = "data/processed/all_indicators.csv"


@st.cache_data(ttl=3600, show_spinner="Fetching latest data from FRED…")
def load_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load the merged indicators DataFrame.

    Prefers the processed CSV for speed; falls back to live FRED API pull.
    ``force_refresh=True`` bypasses the cache (triggered by sidebar button).
    """
    if not force_refresh and Path(PROCESSED_FILE).exists():
        df = load_processed("all_indicators.csv")
    else:
        df = fetch_all_fred()
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📊 PNW Dashboard")
    st.markdown("---")

    # --- Indicator selector ---
    st.markdown("**Indicators**")
    all_keys    = list(FRED_SERIES.keys())
    all_labels  = {k: FRED_SERIES[k]["label"] for k in all_keys}
    selected_keys: list[str] = []

    for key in all_keys:
        checked = st.checkbox(
            all_labels[key],
            value=True,
            key=f"cb_{key}",
        )
        if checked:
            selected_keys.append(key)

    st.markdown("---")

    # --- Date range ---
    st.markdown("**Date Range**")
    date_start = st.date_input("From", value=pd.Timestamp("2000-01-01"), key="date_start")
    date_end   = st.date_input("To",   value=pd.Timestamp.today(),       key="date_end")

    st.markdown("---")

    # --- Data refresh ---
    refresh = st.button("🔄 Refresh from FRED", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.65rem;color:#666;line-height:1.6">'
        "Data: FRED (St. Louis Fed)<br>"
        "Recession bands: NBER<br>"
        "Phase 3: Forecasting →"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load & filter data
# ---------------------------------------------------------------------------

df_raw = load_data()
df = df_raw.loc[
    (df_raw.index >= pd.Timestamp(date_start)) &
    (df_raw.index <= pd.Timestamp(date_end))
].copy()

last_updated = df_raw.index.max().strftime("%B %Y")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f'<div class="dash-title">{APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="dash-subtitle">{APP_SUBTITLE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="last-updated">Data through {last_updated}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Snapshot cards
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Current Snapshot</div>', unsafe_allow_html=True)

if not selected_keys:
    st.info("Select at least one indicator in the sidebar.")
else:
    card_cols = st.columns(len(selected_keys))

    ACCENT_COLORS = {
        "unemployment_rate": CHART_COLORS["unemployment_rate"],
        "cpi_west":          CHART_COLORS["cpi_west"],
        "home_price_index":  CHART_COLORS["home_price_index"],
        "fed_funds_rate":    CHART_COLORS["fed_funds_rate"],
    }

    for col, key in zip(card_cols, selected_keys):
        meta     = FRED_SERIES[key]
        snap     = compute_snapshot(df_raw, key)   # use full history for snapshot
        accent   = ACCENT_COLORS.get(key, "#888888")

        if snap["latest_value"] is None:
            with col:
                st.warning(f"No data for {meta['label']}")
            continue

        value_str = f"{snap['latest_value']:.2f}"
        date_str  = snap["latest_date"].strftime("%b %Y") if snap["latest_date"] else "—"

        # Delta formatting
        if snap["delta"] is not None:
            sign        = "+" if snap["delta"] >= 0 else ""
            direction   = "up" if snap["delta"] >= 0 else "down"
            arrow       = "▲" if snap["delta"] >= 0 else "▼"
            delta_html  = (
                f'<div class="metric-delta {direction}">'
                f'{arrow} {sign}{snap["delta"]:.2f} ({sign}{snap["delta_pct"]:.1f}%) '
                f'{snap["delta_label"]}'
                f'</div>'
            )
        else:
            delta_html = '<div class="metric-delta">— insufficient history</div>'

        with col:
            st.markdown(
                f"""
                <div class="metric-card" style="--card-accent:{accent}">
                  <div class="metric-label">{meta['label']}</div>
                  <div class="metric-value">{value_str}</div>
                  <div class="metric-delta" style="font-size:0.65rem;color:#AAAAAA;margin-bottom:0.2rem">{meta['units']} · {date_str}</div>
                  {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Indicators</div>', unsafe_allow_html=True)

if selected_keys:
    for key in selected_keys:
        if key not in df.columns:
            continue

        meta   = FRED_SERIES[key]
        color  = CHART_COLORS.get(key, "#888888")
        series = df[key].dropna()

        if series.empty:
            st.warning(f"No data in selected range for {meta['label']}.")
            continue

        fig = make_line_chart(
            df=df,
            key=key,
            label=meta["label"],
            units=meta["units"],
            color=color,
            show_recession_bands=True,
        )

        # Wrap in a white card via markdown + plotly chart
        with st.container():
            st.markdown(
                f'<div class="chart-label" style="color:{color}">'
                f'{meta["label"].upper()}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("Select at least one indicator in the sidebar to display charts.")

# ---------------------------------------------------------------------------
# Phase 3 placeholder — "So What?" narrative
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">So What?</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="phase3-placeholder">
        ✦ Phase 3 &nbsp;—&nbsp; Forecasting &amp; narrative insights coming soon<br>
        <span style="font-size:0.65rem">Prophet forecasts · Anomaly detection · Plain-language summary</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Raw data table
# ---------------------------------------------------------------------------

with st.expander("📋  Raw Data", expanded=False):
    if not selected_keys:
        st.info("No indicators selected.")
    else:
        display_df = df[selected_keys].copy()

        # Rename columns to human-readable labels for the table
        rename_map = {k: FRED_SERIES[k]["label"] for k in selected_keys if k in display_df.columns}
        display_df.rename(columns=rename_map, inplace=True)
        display_df.index = display_df.index.strftime("%Y-%m-%d")

        st.dataframe(display_df, use_container_width=True)

        csv_bytes = display_df.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️  Download CSV",
            data=csv_bytes,
            file_name="pnw_indicators.csv",
            mime="text/csv",
        )
