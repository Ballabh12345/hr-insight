"""Employees - the searchable, filterable workforce table.

Added interactivity: click a row and review the employee's recommendation on
the Employee detail page. A department pre-selected elsewhere (e.g. by clicking
a bar on the Overview chart) seeds the department filter.
"""

from __future__ import annotations

import streamlit as st

from app import theme
from app.services import data_service as data

COLUMNS = {
    "EmployeeNumber": "ID",
    "JobRole": "Role",
    "Department": "Department",
    "JobLevel": "Level",
    "Gender": "Gender",
    "AgeBand": "Age band",
    "YearsAtCompany": "Years",
    "attrition_probability": "Risk",
    "risk_band": "Band",
    "performance_tier": "Performance",
    "potential_tier": "Potential",
    "ninebox_label": "Talent segment",
    "retention_action": "Recommended action",
}

#: Compact columns (short values) render narrower in the data grid.
NARROW_COLUMNS = {"ID", "Level", "Gender", "Years", "Performance", "Potential"}


def _review(emp_number: int) -> None:
    """Navigate to the decision surface for one employee (callback: pre-rerun)."""
    st.session_state["detail_employee"] = emp_number
    st.session_state["nav"] = "Employee detail"


def render() -> None:
    df = data.load_employees()

    st.markdown(
        theme.app_header(
            "Employees",
            "Filter the workforce by department, risk band, or protected attribute, "
            "then select a row (tick its checkbox) to open the full recommendation.",
        ),
        unsafe_allow_html=True,
    )

    # A department clicked on the Overview page seeds this filter exactly once.
    preselect = st.session_state.pop("preselect_department", None)
    if preselect:
        st.session_state["emp_dept"] = [preselect]

    # ------------------------------------------------------------- filters
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 3], gap="medium")
    with c1:
        department = st.multiselect("Department", sorted(df["Department"].unique()), key="emp_dept")
    with c2:
        risk_band = st.multiselect("Risk band", data.RISK_ORDER, key="emp_band")
    with c3:
        gender = st.multiselect("Gender", sorted(df["Gender"].unique()), key="emp_gender")
    with c4:
        age_band = st.multiselect("Age band", sorted(df["AgeBand"].unique()), key="emp_age")
    with c5:
        search = st.text_input(
            "Search", placeholder="ID, role or department", key="emp_search"
        )

    filtered = data.filter_employees(
        df,
        department=department,
        risk_band=risk_band,
        gender=gender,
        age_band=age_band,
        search=search,
    )

    st.markdown(
        f'<div class="small-note" style="margin-bottom:0.5rem;">'
        f"{len(filtered):,} of {len(df):,} employees shown</div>",
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No employees match the current filters.")
        return

    # -------------------------------------------------------------- table
    display = filtered[list(COLUMNS)].rename(columns=COLUMNS).copy()
    # Risk stays numeric so the ProgressColumn can render it as a bar; the column
    # config scales it to a percentage.
    display["Risk"] = display["Risk"] * 100
    display = display.sort_values("Risk", ascending=False).reset_index(drop=True)

    event = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=480,
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
            **{
                col: st.column_config.TextColumn(col, width="small")
                for col in NARROW_COLUMNS
            },
        },
        key="emp_table",
        on_select="rerun",
        selection_mode="single-row",
    )

    rows = event.selection.rows if event else []
    if rows:
        picked = display.iloc[rows[0]]
        emp_number = int(picked["ID"])
        st.markdown(
            f'<div class="small-note" style="margin:0.25rem 0 0.5rem 0;">Selected employee '
            f'<b style="color:{theme.TOKENS["text_primary"]};">#{emp_number} &middot; '
            f'{picked["Role"]}</b> &middot; risk {picked["Risk"]:.0f}%</div>',
            unsafe_allow_html=True,
        )
        st.button(
            "Review this employee →",
            type="primary",
            key="emp_open_detail",
            on_click=_review,
            args=(emp_number,),
        )

    # ------------------------------------------------------------- export
    st.download_button(
        "Download filtered data (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="hr_insight_filtered_employees.csv",
        mime="text/csv",
        key="download_filtered",
        type="secondary",
    )
