"""Fairness audit - disparate impact, mitigation comparison, and the detector power curve."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import theme
from app.services import model_service

GROUP_COLORS = {
    "Gender": theme.CATEGORICAL[:2],
    "AgeBand": theme.CATEGORICAL[:4],
    "MaritalStatus": theme.CATEGORICAL[:3],
    "Department": theme.CATEGORICAL[:3],
}


def _metric_bars(block: dict, metric: str, title: str) -> go.Figure:
    """Grouped bar chart of one metric across the groups of an attribute."""
    groups = list(block[metric].keys())
    values = [block[metric][g] for g in groups]
    colors = GROUP_COLORS.get(title, theme.CATEGORICAL)
    fig = go.Figure(
        go.Bar(
            x=groups,
            y=values,
            marker={"color": [colors[i % len(colors)] for i in range(len(groups))]},
            hovertemplate="<b>%{x}</b><br>" + metric.replace("_", " ") + " %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=220, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
    fig.update_yaxes(title_text=metric.replace("_", " "))
    return fig


def render() -> None:
    report = model_service.load_fairness_report()
    if not report:
        st.error(
            "No fairness report found. Run `python ml/train_fairness.py` to build it."
        )
        return

    st.markdown(
        theme.app_header(
            "Fairness audit",
            "Who does this model treat differently? Every number here is measured on the "
            "IBM dataset, which is fictional - the findings are a method demonstration, "
            "not a claim about real employment decisions.",
        ),
        unsafe_allow_html=True,
    )

    baseline = report.get("baseline_audit", {}).get("groups", {})
    mitigation = report.get("mitigation", {})

    # ------------------------------------------------------------ headline
    cards = []
    for attribute, block in baseline.items():
        summary = block["summary"]
        cards.append(
            {
                "label": f"{attribute} parity gap",
                "value": f"{summary['demographic_parity_difference']:.3f}",
                "delta": f"equal opp. gap {summary['equal_opportunity_difference']:.3f}",
                "delta_tone": "bad"
                if summary["demographic_parity_difference"] > 0.1
                else "",
                "accent": "high" if summary["demographic_parity_difference"] > 0.1 else "low",
            }
        )
    st.markdown(theme.kpi_row(cards), unsafe_allow_html=True)

    # --------------------------------------------------- disparate impact
    st.markdown(theme.section_label("Disparate impact - model behaviour by group"), unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note" style="margin-bottom:0.5rem;">'
        "Open an attribute to see its group-level selection, true-positive and "
        "false-positive rates.</div>",
        unsafe_allow_html=True,
    )
    worst = max(
        baseline,
        key=lambda att: baseline[att]["summary"]["demographic_parity_difference"],
        default=None,
    )
    for attribute, block in baseline.items():
        summary = block["summary"]
        gap = summary["demographic_parity_difference"]
        counts = ", ".join(f"{g}: {c}" for g, c in block["count"].items())
        n = sum(block["count"].values())
        with st.expander(
            f"{attribute}  -  parity gap {gap:.3f}  -  {n} employees",
            expanded=(attribute == worst),
        ):
            st.markdown(
                f'<div class="small-note" style="margin-bottom:0.25rem;">'
                f"Group sizes - {counts}</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3, gap="medium")
            metric_specs = [
                ("selection_rate", "Selection rate"),
                ("true_positive_rate", "True positive rate"),
                ("false_positive_rate", "False positive rate"),
            ]
            for column, (metric, title) in zip((c1, c2, c3), metric_specs):
                with column:
                    st.markdown(theme.section_label(title), unsafe_allow_html=True)
                    theme.chart(_metric_bars(block, metric, attribute))
            st.markdown(
                f'<div class="small-note">Selection-rate gap {gap:.3f} &middot; '
                f"equal-opportunity gap {summary['equal_opportunity_difference']:.3f} "
                "&middot; equalized-odds gap "
                f"{summary['equalized_odds_difference']:.3f}</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------- mitigation
    st.markdown(
        theme.section_label("Mitigation - does it close the gap?"), unsafe_allow_html=True
    )
    strategies = [
        ("baseline", "Unmitigated"),
        ("reweighting", "Reweighting"),
        ("equalized_odds", "Equalized odds"),
    ]
    rows = []
    for key, label in strategies:
        block = mitigation.get(key, {}).get("groups", {}).get("Gender", {})
        if not block:
            continue
        summary = block["summary"]
        rows.append(
            {
                "Strategy": label,
                "Demographic parity gap": summary["demographic_parity_difference"],
                "Equal opportunity gap": summary["equal_opportunity_difference"],
                "Equalized odds gap": summary["equalized_odds_difference"],
            }
        )
    if rows:
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            column_config={
                c: st.column_config.NumberColumn(c, format="%.4f")
                for c in rows[0]
                if c != "Strategy"
            },
        )
        st.markdown(
            '<div class="small-note">Reweighting balances the label distribution across '
            "groups, so it closes the parity gap but leaves the equal-opportunity gap "
            "intact - the rate at which each group's actual leavers are caught. "
            "Equalized-odds post-processing targets opportunity directly, but requires "
            "the sensitive attribute to be known at decision time.</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------- power curve
    st.markdown(
        theme.section_label("Detector power curve - does the audit find bias we injected?"),
        unsafe_allow_html=True,
    )
    curve = report.get("power_curve", {}).get("datasets", [])
    if curve:
        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[d["bias_strength"] for d in curve],
                    y=[d["detected_gap"] for d in curve],
                    mode="lines+markers",
                    line={"color": theme.TOKENS["support_error"], "width": 2},
                    marker={"size": 9},
                    name="Detected gap",
                    hovertemplate="bias %{x}<br>gap %{y:.4f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[d["bias_strength"] for d in curve],
                    y=[d["selection_rate_female_rnd"] for d in curve],
                    mode="lines+markers",
                    line={"color": theme.CATEGORICAL[1], "width": 2},
                    name="Female selection rate (R&D)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[d["bias_strength"] for d in curve],
                    y=[d["selection_rate_male_rnd"] for d in curve],
                    mode="lines+markers",
                    line={"color": theme.CATEGORICAL[2], "width": 2},
                    name="Male selection rate (R&D)",
                )
            )
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_xaxes(title_text="Injected bias strength")
            fig.update_yaxes(title_text="Selection rate")
            theme.chart(fig)
            st.markdown(
                '<div class="small-note">At bias strength 0 there is nothing to find. As the '
                "unfair component grows, the gap the audit reports grows with it - which "
                "is what demonstrates the detector works rather than merely reporting "
                "a number.</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------- limitations
    with st.container(border=True):
        st.markdown(theme.section_label("Limitations"), unsafe_allow_html=True)
        for item in report.get("limitations", []):
            st.markdown(f"- {item}")
