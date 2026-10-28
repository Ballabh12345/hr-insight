"""Employee detail - the decision surface: risk, explanation, and accept / override."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import theme
from app.services import audit_service, data_service as data, model_service


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
    # An arrival from the Employees table pre-selects this employee exactly once.
    preselect = st.session_state.pop("detail_employee", None)
    if preselect and preselect in ids:
        st.session_state["emp_select"] = preselect

    selected = st.selectbox(
        "Employee",
        ids,
        format_func=lambda i: options[i],
        key="emp_select",
    )
    emp = df[df["EmployeeNumber"] == selected].iloc[0]

    # ------------------------------------------------------------- headline
    left, middle, right = st.columns([1, 1, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("Attrition risk"), unsafe_allow_html=True)
            st.markdown(
                theme.risk_meter(float(emp["attrition_probability"]), str(emp["risk_band"])),
                unsafe_allow_html=True,
            )
            conformal = model_service.load_attrition_metrics().get("conformal", {})
            q_hat = conformal.get("q_hat")
            if q_hat is not None:
                p = float(emp["attrition_probability"])
                pred_set = []
                if 1.0 - p <= q_hat:
                    pred_set.append("Leave")
                if p <= q_hat:
                    pred_set.append("Stay")
                st.markdown(
                    f'<div class="small-note" style="margin-top:0.5rem;">Conformal set '
                    f'(α={conformal.get("alpha")}): {", ".join(pred_set) or "∅"}</div>',
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
                f'<div style="font-size:1.9rem; font-weight:700; letter-spacing:-0.02em; '
                f'color:{theme.TOKENS["text_primary"]};">{emp["ninebox_label"]}</div>'
                f'<div style="font-size:0.8125rem; color:{theme.TOKENS["text_helper"]}; margin-top:0.25rem;">'
                f'Performance {emp["performance_tier"]} &nbsp;\u00b7&nbsp; '
                f'Potential {emp["potential_tier"]} &nbsp;\u00b7&nbsp; '
                f'Persona cluster {int(emp["persona_cluster"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("", unsafe_allow_html=True)
            st.markdown(theme.section_label("Recommended action"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:1rem; color:{theme.TOKENS["text_primary"]}; '
                f'padding:0.75rem; background:{theme.TOKENS["card_2"]}; '
                f'border-left:3px solid {theme.risk_color(str(emp["risk_band"]))}; '
                f'border-radius:0 10px 10px 0;">{emp["retention_action"]}</div>',
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
                            ],
                            "cornerradius": 6,
                        },
                        hovertemplate="<b>%{y}</b><br>Impact %{x:.3f} log-odds<extra></extra>",
                    )
                )
                fig.update_layout(height=220, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
                fig.update_xaxes(title_text="")
                fig.update_yaxes(autorange="reversed")
                theme.chart(fig)
                st.markdown(
                    f'<div class="small-note">Red ({theme.TOKENS["support_error"]}) pushes the '
                    f'risk up, green ({theme.TOKENS["support_success"]}) pushes it down. '
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
        c1, c2 = st.columns([2, 3], gap="large")
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
                key="override_reason",
            )
            b1, b2 = st.columns(2)
            with b1:
                # The single dominant action is the blue primary; the Override is
                # a tertiary (danger-outline) button.
                if st.button("Accept", type="primary", width="stretch", key="accept_decision"):
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
                if st.button("Override", type="tertiary", width="stretch", key="override_decision"):
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

    # ------------------------------------------------ what-if simulator
    with st.expander("What-if simulator (model sensitivity, not causal proof)"):
        st.markdown(
            '<div class="small-note">Change one or more features and re-score this employee. '
            "This shows how the model responds; it is not proof that changing the factor "
            "would change the outcome.</div>",
            unsafe_allow_html=True,
        )
        base_values = model_service.get_employee_feature_values(int(emp["EmployeeNumber"]))
        overrides: dict = {}
        if str(emp.get("OverTime")) == "Yes":
            if st.checkbox("Set overtime to No", key="wi_overtime"):
                overrides["OverTime"] = "No"
        feature = st.selectbox(
            "Feature to change", list(base_values.keys()), key="wi_feature"
        )
        current = base_values.get(feature)
        new_value = st.text_input(
            f"New value (current: {current})", key="wi_value"
        )
        if st.button("Compute new risk", key="wi_compute"):
            if new_value.strip():
                val: Any = new_value.strip()
                if isinstance(current, (int, float, np.integer, np.floating)):
                    try:
                        parsed = float(val)
                        val = parsed
                    except ValueError:
                        st.error("Please enter a numeric value for this feature.")
                        return
                overrides[feature] = val
            new_proba = model_service.predict_with_overrides(
                int(emp["EmployeeNumber"]), overrides
            )
            if new_proba is not None:
                delta = new_proba - float(emp["attrition_probability"])
                st.markdown(
                    f"New predicted risk: **{new_proba:.1%}** "
                    f"(delta {delta:+.1%} vs. {float(emp['attrition_probability']):.1%})."
                )
