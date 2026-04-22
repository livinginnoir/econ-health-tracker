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

Phase 3 additions
-----------------
- ``add_forecast_overlay`` — overlays a Prophet forecast (yhat + confidence
  band) on an existing figure with a visual break at the last historical date.
- ``add_anomaly_markers`` — overlays scatter markers at flagged anomaly dates.
- ``make_line_chart`` now accepts optional ``forecast_df`` and ``anomaly_flags``
  keyword arguments so callers don't have to call the overlay functions manually.
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
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0,
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
# Phase 3 — Forecast overlay
# ---------------------------------------------------------------------------

def add_forecast_overlay(
    fig: go.Figure,
    forecast_df: pd.DataFrame,
    last_historical_date: pd.Timestamp,
    color: str,
    label: str,
    units: str,
) -> go.Figure:
    """
    Overlay a Prophet forecast on an existing historical chart.

    Visual design
    -------------
    - Confidence band: lightly filled area between ``yhat_lower`` and
      ``yhat_upper``, same hue as the historical line but very transparent.
    - Forecast line: dashed, same color as historical but slightly lighter.
    - Vertical divider: a thin dashed grey line at ``last_historical_date``
      with a "Forecast →" annotation.
    - One data point of overlap at ``last_historical_date`` so the forecast
      line connects visually to the end of the historical line.

    Parameters
    ----------
    fig                  : Existing figure (already has historical trace).
    forecast_df          : Prophet forecast DataFrame with columns
                           ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    last_historical_date : The last date in the historical series — used to
                           split history from forecast and draw the divider.
    color                : Hex color for the forecast line (matching the
                           historical line color).
    label                : Human-readable series name for hover text.
    units                : Y-axis unit string for hover text.

    Returns
    -------
    go.Figure
        Modified figure (same object, returned for chaining).
    """
    # --- Isolate the future portion (plus one overlap point) ---
    future = forecast_df[forecast_df["ds"] >= last_historical_date].copy()

    if future.empty:
        return fig

    # --- Confidence band (filled area between lower and upper) ---
    # Render as a filled-area trace using the "tonexty" fill mode.
    # We add lower first (invisible line), then upper with fill="tonexty".
    fig.add_trace(go.Scatter(
        x=future["ds"],
        y=future["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name="_lower_bound",  # internal name, not shown
    ))

    fig.add_trace(go.Scatter(
        x=future["ds"],
        y=future["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=_hex_to_rgba(color, alpha=0.12),
        showlegend=False,
        hoverinfo="skip",
        name="_upper_bound",
    ))

    # --- Forecast centre line (dashed) ---
    fig.add_trace(go.Scatter(
        x=future["ds"],
        y=future["yhat"],
        mode="lines",
        name=f"{label} (forecast)",
        line=dict(
            color=_hex_to_rgba(color, alpha=0.70),
            width=2,
            dash="dash",
        ),
        hovertemplate=(
            f"<b>{label} — Forecast</b><br>"
            "%{x|%b %Y}<br>"
            f"%{{y:.2f}} {units}<extra></extra>"
        ),
    ))

    # --- Vertical divider line at the forecast start ---
    fig.add_vline(
        x=last_historical_date,
        line_width=1,
        line_dash="dot",
        line_color="rgba(160, 160, 160, 0.6)",
        annotation_text="Forecast →",
        annotation_position="top right",
        annotation_font=dict(size=9, color="#AAAAAA", family=_FONT_FAMILY),
    )

    return fig


# ---------------------------------------------------------------------------
# Phase 3 — Anomaly markers
# ---------------------------------------------------------------------------

def add_anomaly_markers(
    fig: go.Figure,
    series: pd.Series,
    flags: pd.Series,
    color: str,
    label: str,
    units: str,
) -> go.Figure:
    """
    Add scatter markers at anomalous observations on an existing figure.

    Anomaly points are rendered as open circles (hollow) slightly larger than
    the line width.  This style is visually distinctive without obscuring the
    underlying line.

    Parameters
    ----------
    fig    : Existing figure (already has the historical line trace).
    series : The full historical series (same data as in the line trace).
    flags  : Boolean Series from ``anomaly_detector.detect_anomalies()``,
             aligned to ``series`` index.  True = flagged.
    color  : Hex color matching the indicator's line (markers use same hue).
    label  : Human-readable series name for legend + hover text.
    units  : Y-axis unit string for hover text.

    Returns
    -------
    go.Figure
        Modified figure (same object, returned for chaining).
    """
    # Align flags to the series index (flags may be a subset if series had NaNs)
    aligned_flags = flags.reindex(series.index, fill_value=False)
    anomalous     = series[aligned_flags]

    if anomalous.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=anomalous.index,
        y=anomalous.values,
        mode="markers",
        name=f"{label} (anomaly)",
        marker=dict(
            color="rgba(0,0,0,0)",          # transparent fill → hollow circle
            size=10,
            line=dict(color=color, width=2),
            symbol="circle-open",
        ),
        hovertemplate=(
            f"<b>{label} — Unusual</b><br>"
            "%{x|%b %Y}<br>"
            f"%{{y:.2f}} {units}<extra></extra>"
        ),
    ))

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
    forecast_df: pd.DataFrame | None = None,
    anomaly_flags: pd.Series | None = None,
) -> go.Figure:
    """
    Build a single-series line chart for one economic indicator.

    Phase 3 additions: optional forecast overlay and anomaly markers.

    Parameters
    ----------
    df                   : DataFrame with DatetimeIndex and a column named ``key``.
    key                  : Column name in ``df`` to plot.
    label                : Human-readable series name (used in title + hover).
    units                : Y-axis label string.
    color                : Hex or rgba line color.
    show_recession_bands : Whether to overlay NBER recession shading.
    forecast_df          : Optional Prophet forecast DataFrame.  If provided,
                           the forecast overlay is drawn automatically.
                           Schema: ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    anomaly_flags        : Optional boolean Series of anomaly flags aligned to
                           ``df[key]`` index.  If provided, anomaly markers are
                           drawn automatically.

    Returns
    -------
    go.Figure
    """
    series = df[key].dropna()

    fig = go.Figure()

    # --- Historical line + area fill ---
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=label,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor="rgba(0,0,0,0.04)",
        hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}<br>%{{y:.2f}} {units}<extra></extra>",
    ))

    _apply_base_layout(fig, y_title=units)

    if show_recession_bands and len(series) > 0:
        add_recession_bands(fig, series.index.min(), series.index.max())

    # --- Phase 3: anomaly markers (drawn before forecast so forecast is on top) ---
    if anomaly_flags is not None and not series.empty:
        add_anomaly_markers(fig, series, anomaly_flags, color, label, units)

    # --- Phase 3: forecast overlay ---
    if forecast_df is not None and not series.empty:
        add_forecast_overlay(
            fig,
            forecast_df=forecast_df,
            last_historical_date=series.index[-1],
            color=color,
            label=label,
            units=units,
        )

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


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """
    Convert a 6-digit hex color string to an ``rgba(r, g, b, a)`` CSS string.

    Parameters
    ----------
    hex_color : str
        Color in ``#RRGGBB`` format.
    alpha : float
        Opacity in [0, 1].

    Returns
    -------
    str
        e.g. ``"rgba(224, 92, 58, 0.12)"``
    """
    hex_color = hex_color.lstrip("#")
    r, g, b   = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"
