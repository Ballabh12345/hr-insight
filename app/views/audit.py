"""Audit log - every recommendation that was accepted or overridden."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import theme
from app.services import audit_service


def render() -> None:
    st.markdown(
        theme.app_header(
            "Decision log",
            "The human decision is the artefact of record. Every accept and override is "
            "stored with the model version, the probability at the time, and the reason.",
        ),
        unsafe_allow_html=True,
    )

    df = audit_service.load_decisions()

    if df.empty:
        with st.container(border=True):
            st.info(
                "No decisions recorded yet. Open **Employee detail**, review a "
                "recommendation, and accept or override it."
            )
        return

    counts = df["decision"].value_counts().to_dict()
    st.markdown(
        theme.kpi_row(
            [
                {"label": "Decisions recorded", "value": f"{len(df):,}"},
                {
                    "label": "Accepted",
                    "value": f"{counts.get('accepted', 0):,}",
                    "delta": f"{counts.get('accepted', 0) / len(df):.0%} of decisions",
                },
                {
                    "label": "Overridden",
                    "value": f"{counts.get('overridden', 0):,}",
                    "delta": f"{counts.get('overridden', 0) / len(df):.0%} of decisions",
                    "delta_tone": "bad",
                },
                {"label": "Model version", "value": audit_service.model_version()},
            ]
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("All decisions"), unsafe_allow_html=True)
            display = df[
                [
                    "created_at", "employee_number", "job_role", "department",
                    "risk_probability", "risk_band", "decision", "override_reason",
                    "model_version",
                ]
            ].copy()
            display["risk_probability"] = display["risk_probability"].map(lambda v: f"{v:.1%}")
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                height=460,
                column_config={
                    "risk_probability": st.column_config.NumberColumn(
                        "Risk", format="%.1f%%"
                    ),
                },
            )

    with right:
        with st.container(border=True):
            st.markdown(theme.section_label("Override reasons"), unsafe_allow_html=True)
            overrides = df[df["decision"] == "overridden"]
            if overrides.empty:
                st.markdown(
                    '<div class="small-note">No overrides have been recorded.</div>',
                    unsafe_allow_html=True,
                )
            else:
                counts = overrides["override_reason"].value_counts()
                fig = go.Figure(
                    go.Bar(
                        x=list(counts.values)[::-1],
                        y=list(counts.index)[::-1],
                        orientation="h",
                        marker={"color": theme.TOKENS["caution_major"]},
                        hovertemplate="%{y}<br>%{x} override(s)<extra></extra>",
                    )
                )
                fig.update_layout(height=320, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with st.container(border=True):
            st.markdown(theme.section_label("Overrides by risk band"), unsafe_allow_html=True)
            if overrides.empty:
                st.markdown(
                    '<div class="small-note">Nothing to show yet.</div>',
                    unsafe_allow_html=True,
                )
            else:
                band_counts = overrides["risk_band"].value_counts()
                fig = go.Figure(
                    go.Bar(
                        x=list(band_counts.values),
                        y=list(band_counts.index),
                        marker={
                            "color": [
                                theme.risk_color(b) for b in band_counts.index
                            ]
                        },
                    )
                )
                fig.update_layout(height=220, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
