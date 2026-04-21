"""
chart_helpers.py
----------------
Reusable Plotly chart-building functions for the PNW Economic Health Dashboard.

All functions return a ``plotly.graph_objects.Figure`` ready to pass directly
to ``st.plotly_chart()``.  No Streamlit imports here — this module is pure
presentation logic, decoupled from the app layout.

Conventions
-----------
- Every public function accepts a ``pd.DataFrame`` with a DatetimeIndex named
  "date" and at least one column whose name is a key from config.FRED_SERIES.
- Recession shading is applied via ``add_recession_bands(fig)``, which reads
  NBER_RECESSIONS from config.py.
- Color palette is defined once in CHART_COLORS and referenced everywhere else.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

CHART_COLORS: dict[str, str] = {
    "unemployment_rate": "#E05C3A",   # terracotta
    "cpi_west":          "#4A9EBF",   # steel blue
    "home_price_index":  "#6DBF7E",   # sage green
    "fed_funds_rate":    "#C17FD6",   # muted violet
    "recession_fill":    "rgba(180, 180, 180, 0.18)",
    "recession_line":    "rgba(150, 150, 150, 0.0)",
    "grid":              "rgba(200, 200, 200, 0.25)",
    "bg":                "rgba(0,0,0,0)",   # transparent — let Streamlit theme show
    "text":              "#2C2C2C",
}

# Font applied to all chart text
_FONT_FAMILY = "IBM Plex Mono, monospace"

# ---------------------------------------------------------------------------
# NBER recession bands (hardcoded — NBER dates are fixed historical facts)
# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
# ---------------------------------------------------------------------------

NBER_RECESSIONS: list[tuple[str, str]] = [
    ("2001-03-01", "2001-11-01"),   # Dot-com / 9-11
    ("2007-12-01", "2009-06-01"),   # Great Recession
    ("2020-02-01", "2020-04-01"),   # COVID-19
]


# ---------------------------------------------------------------------------
# Shared layout helper
# ---------------------------------------------------------------------------

def _apply_base_layout(
    fig: go.Figure,
    *,
    title: str = "",
    y_title: str = "",
    y_format: str | None = None,
) -> go.Figure:
    """
    Apply the dashboard's standard layout to a figure.

    Parameters
    ----------
    fig       : Figure to modify in place (also returned for chaining).
    title     : Chart title string.
    y_title   : Y-axis label.
    y_format  : Optional tickformat string for the y-axis (e.g. ".1f", ".0%").
    """
    y_axis_kwargs: dict = dict(
        title=y_title,
        title_font=dict(size=11, color="#666666"),
        tickfont=dict(size=10, family=_FONT_FAMILY, color="#555555"),
        gridcolor=CHART_COLORS["grid"],
        zeroline=False,
        showline=False,
    )
    if y_format:
        y_axis_kwargs["tickformat"] = y_format

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, family=_FONT_FAMILY, color=CHART_COLORS["text"]),
            x=0,
            xanchor="left",
            pad=dict(l=4, b=8),
        ),
        plot_bgcolor=CHART_COLORS["bg"],
        paper_bgcolor=CHART_COLORS["bg"],
        font=dict(family=_FONT_FAMILY),
        margin=dict(l=12, r=12, t=44, b=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, family=_FONT_FAMILY),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=10, family=_FONT_FAMILY, color="#555555"),
            zeroline=False,
            showline=False,
        ),
        yaxis=y_axis_kwargs,
        hoverlabel=dict(
            font_family=_FONT_FAMILY,
            font_size=11,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Recession shading
# ---------------------------------------------------------------------------

def add_recession_bands(fig: go.Figure, x_min: pd.Timestamp, x_max: pd.Timestamp) -> go.Figure:
    """
    Add NBER recession shading to an existing figure.

    Only recessions that overlap the visible date range [x_min, x_max] are drawn.
    A single invisible scatter trace is added to the legend to represent all bands.

    Parameters
    ----------
    fig   : Figure to modify in place.
    x_min : Start of the visible x range.
    x_max : End of the visible x range.
    """
    legend_added = False

    for start_str, end_str in NBER_RECESSIONS:
        r_start = pd.Timestamp(start_str)
        r_end   = pd.Timestamp(end_str)

        # Skip if recession is entirely outside the visible window
        if r_end < x_min or r_start > x_max:
            continue

        # Clamp to visible range
        r_start = max(r_start, x_min)
        r_end   = min(r_end, x_max)

        fig.add_vrect(
            x0=r_start,
            x1=r_end,
            fillcolor=CHART_COLORS["recession_fill"],
            line_color=CHART_COLORS["recession_line"],
            layer="below",
            annotation_text="Recession" if not legend_added else "",
            annotation_position="top left",
            annotation_font=dict(size=9, color="#999999", family=_FONT_FAMILY),
        )
        legend_added = True

    return fig


# ---------------------------------------------------------------------------
# Individual indicator charts
# ---------------------------------------------------------------------------

def make_line_chart(
    df: pd.DataFrame,
    key: str,
    label: str,
    units: str,
    color: str,
    show_recession_bands: bool = True,
) -> go.Figure:
    """
    Build a single-series line chart for one economic indicator.

    Parameters
    ----------
    df                   : DataFrame with DatetimeIndex and a column named ``key``.
    key                  : Column name in ``df`` to plot.
    label                : Human-readable series name (used in title + hover).
    units                : Y-axis label string.
    color                : Hex or rgba line color.
    show_recession_bands : Whether to overlay NBER recession shading.

    Returns
    -------
    go.Figure
    """
    series = df[key].dropna()

    fig = go.Figure()

    # Shaded area under the line for visual depth
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=label,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba") if color.startswith("rgb") else
                  color + "14",   # 14 = ~8% opacity hex suffix
        hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}} {units}<extra></extra>",
    ))

    _apply_base_layout(fig, y_title=units)

    if show_recession_bands and len(series) > 0:
        add_recession_bands(fig, series.index.min(), series.index.max())

    return fig


def make_multi_line_chart(
    df: pd.DataFrame,
    keys: list[str],
    labels: dict[str, str],
    units: str,
    title: str = "",
    show_recession_bands: bool = True,
) -> go.Figure:
    """
    Build a multi-series line chart (used for overlaying comparable indicators).

    Parameters
    ----------
    df                   : DataFrame with DatetimeIndex and columns for each key.
    keys                 : List of column names to plot.
    labels               : Mapping of key → display label.
    units                : Shared y-axis label.
    title                : Chart title.
    show_recession_bands : Whether to overlay NBER recession shading.
    """
    fig = go.Figure()

    for key in keys:
        if key not in df.columns:
            continue
        series = df[key].dropna()
        color  = CHART_COLORS.get(key, "#888888")

        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=labels.get(key, key),
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{labels.get(key, key)}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}} {units}<extra></extra>",
        ))

    _apply_base_layout(fig, title=title, y_title=units)

    if show_recession_bands and not df.empty:
        valid_index = df[keys].dropna(how="all").index
        if len(valid_index):
            add_recession_bands(fig, valid_index.min(), valid_index.max())

    return fig


# ---------------------------------------------------------------------------
# Snapshot card helpers
# ---------------------------------------------------------------------------

def compute_snapshot(df: pd.DataFrame, key: str) -> dict:
    """
    Compute the latest value and year-over-year delta for a metric card.

    The prior-year value is determined by matching the same calendar month
    (and quarter for quarterly series) one year prior — not a fixed N-step
    offset.  This avoids landing on the wrong observation when the series has
    gaps or irregular spacing.

    Returns a dict with keys:
      - latest_value  : float
      - latest_date   : pd.Timestamp
      - delta         : float | None  (None if prior-year obs not found)
      - delta_pct     : float | None
      - delta_label   : str           ("YoY")
    """
    series = df[key].dropna()

    if series.empty:
        return dict(
            latest_value=None, latest_date=None,
            delta=None, delta_pct=None, delta_label="",
        )

    latest_value = series.iloc[-1]
    latest_date  = series.index[-1]
    label        = "YoY"

    # Build the target prior-year timestamp: same month, one year back.
    # For quarterly series the day-of-month may differ slightly across years,
    # so we search within a ±15-day window around the target date.
    prior_target = latest_date - pd.DateOffset(years=1)
    window_start = prior_target - pd.Timedelta(days=15)
    window_end   = prior_target + pd.Timedelta(days=15)

    candidates = series.loc[window_start:window_end]

    if candidates.empty:
        delta = delta_pct = None
    else:
        # Use the observation closest to the exact prior-year target date.
        # idxmin() returns the index label of the closest date — safe for DatetimeIndex
        time_deltas  = pd.Series(candidates.index.to_list(), index=candidates.index) - prior_target
        closest_date = time_deltas.abs().idxmin()
        prior_value  = candidates.loc[closest_date]
        delta        = latest_value - prior_value
        delta_pct    = (delta / prior_value * 100) if prior_value != 0 else None

    return dict(
        latest_value=latest_value,
        latest_date=latest_date,
        delta=delta,
        delta_pct=delta_pct,
        delta_label=label,
    )
