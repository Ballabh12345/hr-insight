"""Overview - the landing page: headline KPIs, risk distribution, nine-box heatmap."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app import theme
from app.services import data_service as data
from ml.common import drift as DRIFT
from ml.common import metrics as M_METRICS

PERFORMANCE_ROWS = ["High", "Medium", "Low"]
POTENTIAL_COLS = ["Low", "Medium", "High"]


def _go_employees(department: str) -> None:
    """Seed the Employees page with a department pre-filter (callback: pre-rerun)."""
    st.session_state["preselect_department"] = department
    st.session_state["nav"] = "Employees"


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
                {"label": "Employees", "value": f"{len(df):,}", "accent": "primary"},
                {"label": "Attrition rate", "value": f"{attrition_rate:.1%}", "accent": "medium"},
                {
                    "label": "High risk",
                    "value": f"{high:,}",
                    "delta": f"{high / len(df):.1%} of workforce",
                    "delta_tone": "bad",
                    "accent": "high",
                },
                {"label": "Medium risk", "value": f"{medium:,}", "accent": "medium"},
                {"label": "Low risk", "value": f"{low:,}", "accent": "low"},
            ]
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    # ------------------------------------------------- risk by department
    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("Attrition risk by department"), unsafe_allow_html=True)
            st.markdown(
                '<div class="small-note" style="margin-bottom:0.25rem;">Click a bar to open that department in the Employees table.</div>',
                unsafe_allow_html=True,
            )
            dept = (
                df.groupby("Department", as_index=False)
                .agg(
                    Mean_probability=("attrition_probability", "mean"),
                    High_risk=("risk_band", lambda s: (s == "High").sum()),
                    Headcount=("EmployeeNumber", "count"),
                )
                .sort_values("Mean_probability", ascending=True)  # highest at the top
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
                marker={"cornerradius": 8},
                hovertemplate=(
                    "<b>%{y}</b><br>Mean risk %{x:.1%}<br>Headcount %{customdata[0]}"
                    "<br>High risk %{customdata[1]}<extra></extra>"
                ),
            )
            fig.update_layout(height=270, margin=dict(l=8, r=8, t=8, b=8))
            event = theme.chart(fig, key="ov_dept_chart", on_select="rerun", selection_mode=["points"])
            points = event.selection.points if event else []
            clicked = points[-1].get("y") if points else None
            if clicked:
                clicked = str(clicked)
                st.info(f"**{clicked}** selected - jump straight in?", icon="🔍")
                if st.button(
                    f"Open {clicked} in Employees →",
                    type="primary",
                    key="ov_jump_dept",
                    on_click=_go_employees,
                    args=(clicked,),
                ):
                    pass  # the callback navigates before rerun

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
                        "line": {"color": theme.TOKENS["card"], "width": 3},
                    },
                    textinfo="label+percent",
                    textposition="inside",
                    insidetextorientation="horizontal",
                    textfont={"size": 11},
                )
            )
            # Centre annotation - SVG does not parse named entities, use "·".
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text=(
                    f"<b style='color:{theme.TOKENS['text_primary']};'>{len(df):,}</b><br>"
                    f"<span style='color:{theme.TOKENS['text_helper']};font-size:0.7rem;'>"
                    f"employees \u00b7 {high} high</span>"
                ),
                showarrow=False,
            )
            fig.update_layout(height=270, showlegend=False, margin=dict(l=8, r=8, t=8, b=8))
            theme.chart(fig)

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
            f"{int(row.Count)} employees \u00b7 mean risk {row.Mean_risk:.0%}",
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
                "the deeper the blue, the higher the risk. All nine cells are occupied."
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
                marker={"cornerradius": 8},
                hovertemplate="<b>%{y}</b><br>Mean risk %{x:.1%}<br>%{customdata[0]} employees<extra></extra>",
            )
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8))
            theme.chart(fig)

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
                        f'<div style="display:flex; justify-content:space-between; align-items:center; '
                        f'padding:0.5rem 0; border-bottom:1px solid {theme.TOKENS["border_subtle"]};">'
                        f'<span style="font-size:0.875rem; color:{theme.TOKENS["text_primary"]};">'
                        f"{row.retention_action}</span>"
                        f'<span style="font-size:0.875rem; font-weight:600; '
                        f'color:{theme.TOKENS["text_secondary"]};">{row.Employees}</span></div>',
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------ cost-sensitive threshold
    with st.container(border=True):
        st.markdown(theme.section_label("Cost-sensitive threshold (expected value)"), unsafe_allow_html=True)
        st.markdown(
            '<div class="small-note">Explore how the decision threshold changes with the '
            "business cost of losing someone versus the cost of a retention action.</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            cost_leaver = st.slider("Cost of a departure", 10000, 100000, 50000, step=5000, key="cost_leaver")
        with c2:
            cost_action = st.slider("Cost of an action", 1000, 20000, 5000, step=500, key="cost_action")
        with c3:
            success = st.slider("Action success probability", 0.0, 1.0, 0.3, step=0.05, key="cost_success")
        y_true = (df["actual_attrition"] == "Yes").astype(int).to_numpy()
        result = M_METRICS.cost_optimal_threshold(
            y_true, df["attrition_probability"].to_numpy(), cost_leaver, cost_action, success
        )
        st.markdown(
            f'<div class="small-note">Cost-optimal threshold <b>{result["threshold"]:.3f}</b> '
            f'(analytic {result["analytic_threshold"]:.3f}) &nbsp;·&nbsp; '
            f'expected cost {result["expected_cost"]:,.0f} vs {result["baseline_cost_at_0.5"]:,.0f} at 0.5.</div>',
            unsafe_allow_html=True,
        )

    # ------------------------------------------------ drift check (PSI)
    try:
        import pandas as pd

        from ml.common import io as _io

        ref = pd.read_csv(_io.RAW_DIR / "ibm_hr.csv")
        cur = pd.read_csv(_io.PROCESSED_DIR / "synthetic_employees_b2.csv")
        with st.container(border=True):
            st.markdown(theme.section_label("Data drift check (PSI)"), unsafe_allow_html=True)
            st.markdown(
                '<div class="small-note">Population Stability Index between the IBM reference '
                "data and a high-bias synthetic cohort. PSI > 0.2 suggests meaningful drift.</div>",
                unsafe_allow_html=True,
            )
            for col in ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany"]:
                if col in ref.columns and col in cur.columns:
                    value = DRIFT.psi(ref[col], cur[col])
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; font-size:0.8125rem;">'
                        f'<span>{col}</span><span>{value:.3f}</span></div>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        pass
