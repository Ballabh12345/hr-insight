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


def _sequential_scale() -> list:
    """Build a Plotly colourscale from the Carbon sequential Blue ramp."""
    n = len(theme.SEQUENTIAL) - 1
    return [[i / n, c] for i, c in enumerate(theme.SEQUENTIAL)]


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
                .sort_values("Mean_probability", ascending=False)
            )
            fig = px.bar(
                dept,
                x="Mean_probability",
                y="Department",
                orientation="h",
                color="Mean_probability",
                color_continuous_scale=theme.SEQUENTIAL,
                labels={"Mean_probability": "Mean attrition probability"},
                hover_data={"Headcount": True, "High_risk": True},
            )
            fig.update_layout(height=260, coloraxis_showscale=False, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_traces(marker_line_width=0, hovertemplate=(
                "<b>%{y}</b><br>Mean risk %{x:.1%}<br>Headcount %{customdata[0]}"
                "<br>High risk %{customdata[1]}<extra></extra>"
            ))
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
                        "colors": [theme.RISK_COLORS[b][0] for b in bands.index],
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
    grid = pd.crosstab(df["performance_tier"], df["potential_tier"])
    grid = grid.reindex(index=PERFORMANCE_ROWS, columns=POTENTIAL_COLS, fill_value=0)
    labels = ninebox.get("ninebox_labels", {})
    text = [
        [
            f"{labels.get(f'{r}|{c}', '')}\n{grid.loc[r, c]} employees"
            for c in POTENTIAL_COLS
        ]
        for r in PERFORMANCE_ROWS
    ]

    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=[f"Potential: {c}" for c in POTENTIAL_COLS],
            y=[f"Performance: {r}" for r in PERFORMANCE_ROWS],
            text=text,
            texttemplate="%{text}",
            colorscale=_sequential_scale(),
            showscale=False,
            hovertemplate="%{y} / %{x}<br>%{z} employees<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(height=320, margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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
                .sort_values("Mean_risk", ascending=False)
                .head(9)
            )
            fig = px.bar(
                cell,
                x="Mean_risk",
                y="ninebox_label",
                orientation="h",
                color="Mean_risk",
                color_continuous_scale=theme.SEQUENTIAL,
                labels={"Mean_risk": "Mean attrition probability", "ninebox_label": ""},
                hover_data={"Employees": True},
            )
            fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_traces(marker_line_width=0)
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
