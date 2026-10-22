"""Overview - the landing page: headline KPIs, risk distribution, nine-box heatmap."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app import theme
from app.services import data_service as data

PERFORMANCE_ROWS = ["High", "Medium", "Low"]
POTENTIAL_COLS = ["Low", "Medium", "High"]


def render() -> None:
    df = data.load_employees()
    summary = data.load_summary()
    ninebox = data.load_ninebox_metrics()

    st.markdown(
        theme.app_header(
            "Workforce Risk Overview",
            "Attrition risk, talent placement, and recommended retention actions for "
            f"{len(df):,} employees.",
        ),
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------- KPIs
    high = summary.get("high_risk_count", 0)
    medium = summary.get("medium_risk_count", 0)
    low = summary.get("low_risk_count", 0)
    attrition_rate = summary.get("attrition_rate", 0.0)

    st.markdown(
        theme.kpi_row(
            [
                {"label": "Employees", "value": f"{len(df):,}"},
                {"label": "Attrition rate", "value": f"{attrition_rate:.1%}"},
                {
                    "label": "High risk",
                    "value": f"{high:,}",
                    "delta": f"{high / len(df):.1%} of workforce",
                    "delta_tone": "bad",
                },
                {"label": "Medium risk", "value": f"{medium:,}"},
                {"label": "Low risk", "value": f"{low:,}"},
            ]
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    # ------------------------------------------------- risk by department
    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("Attrition risk by department"), unsafe_allow_html=True)
            dept = (
                df.groupby("Department", as_index=False)
                .agg(
                    Mean_probability=("attrition_probability", "mean"),
                    High_risk=("risk_band", lambda s: (s == "High").sum()),
                    Headcount=("EmployeeNumber", "count"),
                )
                .sort_values("Mean_probability", ascending=True)  # Plotly puts the first
                # row at the bottom, so ascending puts the highest risk at the top.
            )
            fig = px.bar(
                dept,
                x="Mean_probability",
                y="Department",
                orientation="h",
                labels={"Mean_probability": "Mean attrition probability", "Department": ""},
                hover_data={"Headcount": True, "High_risk": True},
            )
            fig.update_traces(
                marker_color=theme.TOKENS["primary"],
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>Mean risk %{x:.1%}<br>Headcount %{customdata[0]}"
                    "<br>High risk %{customdata[1]}<extra></extra>"
                ),
            )
            fig.update_layout(height=260, margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ------------------------------------------------------ risk band split
    with right:
        with st.container(border=True):
            st.markdown(theme.section_label("Risk band distribution"), unsafe_allow_html=True)
            bands = df["risk_band"].value_counts().reindex(data.RISK_ORDER).fillna(0)
            fig = go.Figure(
                go.Pie(
                    labels=[str(i) for i in bands.index],
                    values=bands.values,
                    hole=0.62,
                    marker={
                        "colors": [theme.RISK_SERIES[b] for b in bands.index],
                        "line": {"color": "#ffffff", "width": 2},
                    },
                    textinfo="label+percent",
                    textfont={"size": 12},
                )
            )
            fig.update_layout(height=260, showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ------------------------------------------------------- nine-box grid
    st.markdown(theme.section_label("Talent matrix - engagement x potential"), unsafe_allow_html=True)
    labels = ninebox.get("ninebox_labels", {})
    cell = (
        df.groupby(["performance_tier", "potential_tier"], as_index=False)
        .agg(
            Mean_risk=("attrition_probability", "mean"),
            Count=("EmployeeNumber", "count"),
        )
    )
    cells: dict[tuple[str, str], tuple[str, str, float]] = {}
    values: list[float] = []
    for row in cell.itertuples():
        key = (row.performance_tier, row.potential_tier)
        cells[key] = (
            labels.get(f"{key[0]}|{key[1]}", "-"),
            f"{int(row.Count)} employees &middot; mean risk {row.Mean_risk:.0%}",
            float(row.Mean_risk),
        )
        values.append(float(row.Mean_risk))

    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9

    st.markdown(
        theme.matrix_html(
            rows=PERFORMANCE_ROWS,
            cols=POTENTIAL_COLS,
            cells=cells,
            vmin=vmin,
            vmax=vmax,
            col_caption="POTENTIAL",
            row_caption=(
                "Tile shading is the mean predicted attrition risk in that segment - "
                "the darker the tile, the higher the risk. All nine cells are occupied."
            ),
        ),
        unsafe_allow_html=True,
    )

    # ------------------------------------------- risk inside each nine-box
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                theme.section_label("Mean risk inside each talent segment"),
                unsafe_allow_html=True,
            )
            cell = (
                df.groupby("ninebox_label", as_index=False)
                .agg(Mean_risk=("attrition_probability", "mean"), Employees=("EmployeeNumber", "count"))
                .sort_values("Mean_risk", ascending=True)  # highest at the top
            )
            fig = px.bar(
                cell,
                x="Mean_risk",
                y="ninebox_label",
                orientation="h",
                labels={"Mean_risk": "Mean attrition probability", "ninebox_label": ""},
                hover_data={"Employees": True},
            )
            fig.update_traces(
                marker_color=theme.TOKENS["primary"],
                marker_line_width=0,
                hovertemplate="<b>%{y}</b><br>Mean risk %{x:.1%}<br>%{customdata[0]} employees<extra></extra>",
            )
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            st.markdown(theme.section_label("Recommended actions"), unsafe_allow_html=True)
            actions = (
                df[df["risk_band"] != "Low"]
                .groupby("retention_action", as_index=False)
                .agg(Employees=("EmployeeNumber", "count"))
                .sort_values("Employees", ascending=False)
            )
            if actions.empty:
                st.info("No employees currently need a retention action.")
            else:
                for row in actions.itertuples():
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; '
                        f'padding:0.5rem 0; border-bottom:1px solid {theme.TOKENS["border_subtle"]};">'
                        f'<span style="font-size:0.875rem; color:{theme.TOKENS["text_primary"]};">'
                        f"{row.retention_action}</span>"
                        f'<span style="font-size:0.875rem; font-weight:600; '
                        f'color:{theme.TOKENS["text_secondary"]};">{row.Employees}</span></div>',
                        unsafe_allow_html=True,
                    )
