"""Employees - the searchable, filterable workforce table."""

from __future__ import annotations

import pandas as pd
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
    "performance_tier": "Perf.",
    "potential_tier": "Pot.",
    "ninebox_label": "Talent segment",
    "retention_action": "Recommended action",
}


def render() -> None:
    df = data.load_employees()

    st.markdown(
        theme.app_header(
            "Employees",
            "Filter the workforce by department, risk band, or protected attribute, "
            "then open an employee to review the full recommendation.",
        ),
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------- filters
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 3], gap="medium")
    with c1:
        department = st.multiselect("Department", sorted(df["Department"].unique()), [])
    with c2:
        risk_band = st.multiselect("Risk band", data.RISK_ORDER, [])
    with c3:
        gender = st.multiselect("Gender", sorted(df["Gender"].unique()), [])
    with c4:
        age_band = st.multiselect("Age band", sorted(df["AgeBand"].unique()), [])
    with c5:
        search = st.text_input("Search", placeholder="ID, role or department", label_visibility="collapsed")

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
    display = display.sort_values("Risk", ascending=False)

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "Risk": st.column_config.ProgressColumn(
                "Risk", format="%.1f%%", min_value=0.0, max_value=100.0
            ),
        },
    )

    # ------------------------------------------------------------- export
    st.download_button(
        "Download filtered data (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="hr_insight_filtered_employees.csv",
        mime="text/csv",
        key="download_filtered",
    )

    st.markdown(
        '<div class="small-note" style="margin-top:1rem;">'
        "Open an employee from the <b>Employee detail</b> page to record an "
        "accept / override decision.</div>",
        unsafe_allow_html=True,
    )
