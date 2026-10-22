"""Employee detail - the decision surface: risk, explanation, and accept / override."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import theme
from app.services import audit_service, data_service as data, model_service


def _risk_gauge(probability: float, band: str) -> go.Figure:
    """A horizontal risk meter with the band thresholds marked.

    Plotly's gauge indicator only supports ``angular`` and ``bullet`` shapes, and a
    bullet gauge reads poorly at this size. A single horizontal bar with threshold
    markers is both simpler and clearer.
    """
    color = theme.risk_color(band)
    fig = go.Figure(
        go.Bar(
            x=[probability],
            y=["Attrition risk"],
            orientation="h",
            marker={"color": color},
            width=0.4,
            text=f"<b>{probability:.0%}</b>",
            textposition="outside",
            textfont={"size": 22, "color": theme.TOKENS["text_primary"]},
            hovertemplate="Attrition risk %{x:.1%}<extra></extra>",
        )
    )
    for threshold in (0.25, 0.55):
        fig.add_vline(
            x=threshold,
            line={"color": theme.TOKENS["border_strong"], "width": 1, "dash": "dash"},
        )
    fig.update_layout(
        height=120,
        margin=dict(l=4, r=52, t=6, b=6),
        showlegend=False,
        bargap=0.5,
        xaxis={
            "range": [0, 1],
            "tickmode": "array",
            "tickvals": [0, 0.25, 0.55, 1.0],
            "ticktext": ["0%", "25%", "55%", "100%"],
            "gridcolor": theme.TOKENS["border_subtle"],
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "tickfont": {"size": 10, "color": theme.TOKENS["text_helper"]},
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "ticks": "",
        },
    )
    return fig


def render() -> None:
    df = data.load_employees()

    st.markdown(
        theme.app_header(
            "Employee detail",
            "Every recommendation carries a probability, a reason, and an audit trail. "
            "Accept it or override it - either way the decision is recorded.",
        ),
        unsafe_allow_html=True,
    )

    options = data.employee_options(df)
    ids = list(options)
    default_index = 0
    selected = st.selectbox(
        "Employee",
        ids,
        format_func=lambda i: options[i],
        index=default_index,
        label_visibility="collapsed",
    )
    emp = df[df["EmployeeNumber"] == selected].iloc[0]

    # ------------------------------------------------------------- headline
    left, middle, right = st.columns([1, 1, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("Attrition risk"), unsafe_allow_html=True)
            st.plotly_chart(
                _risk_gauge(float(emp["attrition_probability"]), str(emp["risk_band"])),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.markdown(
                f'<div style="text-align:center; margin-top:0.25rem;">'
                f'{theme.risk_badge(str(emp["risk_band"]))}'
                f'<div class="small-note" style="margin-top:0.5rem;">'
                f"Bands: &lt;25% &middot; 25-55% &middot; &gt;55%</div></div>",
                unsafe_allow_html=True,
            )

    with middle:
        with st.container(border=True):
            st.markdown(theme.section_label("Profile"), unsafe_allow_html=True)
            rows = [
                ("Role", str(emp["JobRole"])),
                ("Department", str(emp["Department"])),
                ("Job level", str(int(emp["JobLevel"]))),
                ("Gender", str(emp["Gender"])),
                ("Age band", str(emp["AgeBand"])),
                ("Years at company", f"{int(emp['YearsAtCompany'])}"),
                ("Years since promotion", f"{int(emp['YearsSinceLastPromotion'])}"),
                ("Overtime", str(emp["OverTime"])),
            ]
            for label, value in rows:
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; '
                    f'padding:0.3rem 0; border-bottom:1px solid {theme.TOKENS["border_subtle"]};">'
                    f'<span style="font-size:0.8125rem; color:{theme.TOKENS["text_helper"]};">{label}</span>'
                    f'<span style="font-size:0.8125rem; color:{theme.TOKENS["text_primary"]};">{value}</span></div>',
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            st.markdown(theme.section_label("Talent placement"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:1.75rem; font-weight:600; '
                f'color:{theme.TOKENS["text_primary"]};">{emp["ninebox_label"]}</div>'
                f'<div style="font-size:0.8125rem; color:{theme.TOKENS["text_helper"]}; margin-top:0.25rem;">'
                f'Performance {emp["performance_tier"]} &nbsp;&middot;&nbsp; '
                f'Potential {emp["potential_tier"]} &nbsp;&middot;&nbsp; '
                f'Persona cluster {int(emp["persona_cluster"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("", unsafe_allow_html=True)
            st.markdown(theme.section_label("Recommended action"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:1rem; color:{theme.TOKENS["text_primary"]}; '
                f'padding:0.75rem; background:{theme.TOKENS["surface"]}; '
                f'border-left:3px solid {theme.risk_color(str(emp["risk_band"]))}; '
                f'border-radius:0 4px 4px 0;">{emp["retention_action"]}</div>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------- explanation
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(
                theme.section_label("Why this recommendation (SHAP)"),
                unsafe_allow_html=True,
            )
            reasons = model_service.explain_employee(int(emp["EmployeeNumber"]))
            if not reasons:
                st.info("No explanation available for this employee.")
            else:
                fig = go.Figure(
                    go.Bar(
                        x=[r["impact"] for r in reasons],
                        y=[r["label"] for r in reasons],
                        orientation="h",
                        marker={
                            "color": [
                                theme.TOKENS["support_error"]
                                if r["impact"] > 0
                                else theme.TOKENS["support_success"]
                                for r in reasons
                            ]
                        },
                        hovertemplate="<b>%{y}</b><br>Impact %{x:.3f} log-odds<extra></extra>",
                    )
                )
                fig.update_layout(height=210, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
                fig.update_xaxes(title_text="Contribution to log-odds")
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.markdown(
                    '<div class="small-note">Red pushes the risk up, green pushes it down. '
                    "Values are additive contributions in log-odds space.</div>",
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            st.markdown(
                theme.section_label("Fairness context for this group"),
                unsafe_allow_html=True,
            )
            report = model_service.load_fairness_report()
            baseline = report.get("baseline_audit", {}).get("groups", {})
            st.markdown(
                '<div class="small-note">Selection rate of the model across protected groups. '
                "A gap between groups is a signal to review, not proof of unfairness.</div>",
                unsafe_allow_html=True,
            )
            for attribute, block in baseline.items():
                rates = block["selection_rate"]
                summary = block["summary"]
                st.markdown(
                    f'<div style="font-size:0.8125rem; font-weight:600; '
                    f'color:{theme.TOKENS["text_primary"]}; margin-top:0.75rem;">{attribute}</div>',
                    unsafe_allow_html=True,
                )
                for group, rate in rates.items():
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; '
                        f'font-size:0.8125rem; color:{theme.TOKENS["text_secondary"]};">'
                        f'<span>{group}</span><span>{rate:.1%}</span></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="small-note" style="margin-top:0.25rem;">'
                    f"Demographic parity gap {summary['demographic_parity_difference']:.3f}</div>",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------------- decision
    st.markdown(theme.section_label("Record the decision"), unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            st.markdown(
                f'<div style="font-size:0.8125rem; color:{theme.TOKENS["text_secondary"]};">'
                f"Recommendation for employee #{int(emp['EmployeeNumber'])}:<br>"
                f'<b style="color:{theme.TOKENS["text_primary"]};">{emp["retention_action"]}</b>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="small-note" style="margin-top:0.5rem;">'
                f"Model {audit_service.model_version()}</div>",
                unsafe_allow_html=True,
            )

        with c2:
            reason = st.text_input(
                "Override reason (required only when overriding)",
                placeholder="e.g. Recently promoted - retention risk not applicable",
                label_visibility="collapsed",
            )
            b1, b2, b3 = st.columns([1, 1, 2])
            with b1:
                if st.button("Accept", type="primary", width="stretch"):
                    audit_service.record_decision(
                        employee_number=int(emp["EmployeeNumber"]),
                        job_role=str(emp["JobRole"]),
                        department=str(emp["Department"]),
                        risk_probability=float(emp["attrition_probability"]),
                        risk_band=str(emp["risk_band"]),
                        recommended_action=str(emp["retention_action"]),
                        decision="accepted",
                    )
                    st.success("Decision recorded.")
            with b2:
                if st.button("Override", width="stretch"):
                    if not reason.strip():
                        st.error("An override needs a reason.")
                    else:
                        audit_service.record_decision(
                            employee_number=int(emp["EmployeeNumber"]),
                            job_role=str(emp["JobRole"]),
                            department=str(emp["Department"]),
                            risk_probability=float(emp["attrition_probability"]),
                            risk_band=str(emp["risk_band"]),
                            recommended_action=str(emp["retention_action"]),
                            decision="overridden",
                            override_reason=reason,
                        )
                        st.success("Override recorded.")
            with b3:
                st.markdown("", unsafe_allow_html=True)
