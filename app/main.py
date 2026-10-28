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

PAGES = {
    "Overview": ("app.views.overview", "Workforce risk at a glance"),
    "Employees": ("app.views.employees", "Search and filter the workforce"),
    "Employee detail": ("app.views.employee_detail", "Review and decide on a recommendation"),
    "Fairness audit": ("app.views.fairness", "Disparate impact and mitigation"),
    "Model performance": ("app.views.model_performance", "How the model was built and tested"),
    "Decision log": ("app.views.audit", "Accepted and overridden recommendations"),
    "Assistant": ("app.views.assistant", "Ask about the workforce or the labour market"),
}

#: Glyph prefixes for nav items (display-only; the radio options stay bare).
NAV_ICONS = {
    "Overview": "◎",
    "Employees": "⌗",
    "Employee detail": "⧉",
    "Fairness audit": "⚖",
    "Model performance": "∿",
    "Decision log": "⿴",
    "Assistant": "✦",
}

_MODE_CHOICE = {"☀️ Day": "day", "🌙 Night": "night"}


def _resolve_mode() -> str:
    """Day/Night is widget state first, then ?mode= in the URL, then Day."""
    if "app_mode" not in st.session_state:
        qp = st.query_params.get("mode", "")
        st.session_state["app_mode"] = qp if qp in theme.MODES else "day"
    return st.session_state["app_mode"]


def _sidebar() -> str:
    """Render the navigation shell and return the selected page."""
    current = _resolve_mode()
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.25rem 0 0.75rem 0.75rem;">'
            '<div style="font-size:1.3rem; font-weight:700; letter-spacing:-0.02em; '
            f'color:{theme.TOKENS["sidebar_text"]};">HR Insight</div>'
            f'<div style="font-size:0.75rem; color:{theme.TOKENS["sidebar_text_2"]}; '
            'margin-top:0.2rem;">Smart HR Decision Support Platform</div></div>',
            unsafe_allow_html=True,
        )

        # Appearance toggle - a real widget state, applied on this rerun.
        choice = st.segmented_control(
            "Appearance",
            list(_MODE_CHOICE),
            default=st.session_state["app_mode"] == "night" and "🌙 Night" or "☀️ Day",
            key="appearance",
        )
        if choice is not None and _MODE_CHOICE[choice] != current:
            st.session_state["app_mode"] = _MODE_CHOICE[choice]
            st.query_params["mode"] = _MODE_CHOICE[choice]
            st.rerun()

        # Hard refresh - rebuild every cached load without restarting the server.
        if st.button(
            "⟳ Hard refresh",
            help=(
                "Clear all Streamlit caches (dataset, models, fairness report) and "
                "rebuild this page from the current artifacts on disk. Use it after "
                "retraining, or press Ctrl+F5 for a browser-level hard refresh."
            ),
            use_container_width=True,
        ):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state["just_hard_refreshed"] = True
            st.rerun()

        if st.session_state.pop("just_hard_refreshed", False):
            st.toast("Cache cleared - data rebuilt from disk.", icon="⟳")

        st.markdown(
            f'<div class="section-label" style="color:{theme.TOKENS["sidebar_text_2"]}; '
            'margin-left:0.75rem;">Navigation</div>',
            unsafe_allow_html=True,
        )
        labels = list(PAGES)
        selected = st.radio(
            "Navigation",
            labels,
            label_visibility="collapsed",
            format_func=lambda x: f"{NAV_ICONS.get(x, '')}  {x}",
            key="nav",
        )
        # Flows with the nav (not position:fixed) so it can never overlay items.
        st.markdown(
            f'<div style="margin-top:2rem; padding:1rem 0 1rem 0.75rem; '
            f'border-top:1px solid {theme.TOKENS["sidebar_border"]}; '
            f'font-size:0.75rem; color:{theme.TOKENS["sidebar_text_2"]}; line-height:1.7;">'
            "Machine Learning &middot; CCSML0301<br>"
            "IBM HR Analytics dataset (fictional, CC0)<br>"
            "Aurora Glass design system</div>",
            unsafe_allow_html=True,
        )
    return selected


def main() -> None:
    # Palette first so both the sidebar and the page body render in one style.
    current = _resolve_mode()
    theme.set_mode(current)
    theme.inject_theme()
    theme.register_plotly_template()

    selected = _sidebar()
    module_name, _ = PAGES[selected]
    module = __import__(module_name, fromlist=["render"])
    module.render()


if __name__ == "__main__":
    main()
