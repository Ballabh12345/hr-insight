"""HR Insight - Smart HR Decision Support Platform.

Application entry point. Run with:

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import theme  # noqa: E402

st.set_page_config(
    page_title="HR Insight",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject_theme()
theme.register_plotly_template()

PAGES = {
    "Overview": ("app.views.overview", "Workforce risk at a glance"),
    "Employees": ("app.views.employees", "Search and filter the workforce"),
    "Employee detail": ("app.views.employee_detail", "Review and decide on a recommendation"),
    "Fairness audit": ("app.views.fairness", "Disparate impact and mitigation"),
    "Model performance": ("app.views.model_performance", "How the model was built and tested"),
    "Decision log": ("app.views.audit", "Accepted and overridden recommendations"),
    "Assistant": ("app.views.assistant", "Ask about the workforce or the labour market"),
}


def _sidebar() -> str:
    """Render the navigation shell and return the selected page."""
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.25rem 0 1rem 0.75rem;">'
            '<div style="font-size:1.25rem; font-weight:600; color:#ffffff; '
            'letter-spacing:-0.01em;">HR Insight</div>'
            '<div style="font-size:0.75rem; color:#a8a8a8; margin-top:0.25rem;">'
            "Smart HR Decision Support Platform</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-label" style="color:#6f6f6f;">Navigation</div>',
            unsafe_allow_html=True,
        )
        labels = list(PAGES)
        selected = st.radio(
            "Navigation",
            labels,
            label_visibility="collapsed",
            format_func=lambda x: x,
        )
        st.markdown(
            '<div style="position:fixed; bottom:1rem; left:0.75rem; right:0.75rem; '
            'font-size:0.75rem; color:#6f6f6f;">'
            "Machine Learning &middot; CCSML0301<br>"
            "IBM HR Analytics dataset (fictional, CC0)<br>"
            "IBM Carbon Design System</div>",
            unsafe_allow_html=True,
        )
    return selected


def main() -> None:
    selected = _sidebar()
    module_name, _ = PAGES[selected]
    module = __import__(module_name, fromlist=["render"])
    module.render()


if __name__ == "__main__":
    main()
