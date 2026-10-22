"""
Carbon Design System theme engine for the HR Insight Streamlit app.

Implements the 11 Streamlit overrides and the shared Plotly template specified in
``docs/design.md``.

Source of truth: https://carbondesignsystem.com/  (IBM Carbon Design System, Apache-2.0)
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------------------
# 1. Design tokens (Carbon White theme + Gray 100 dark shell)
# ---------------------------------------------------------------------------

TOKENS: dict[str, str] = {
    # Surfaces & layers
    "bg": "#ffffff",
    "surface": "#f4f4f4",
    "surface_2": "#ffffff",
    "surface_accent": "#e0e0e0",
    # Borders
    "border_subtle": "#e0e0e0",
    "border_subtle_2": "#c6c6c6",
    "border_strong": "#8d8d8d",
    # Text
    "text_primary": "#161616",
    "text_secondary": "#525252",
    "text_helper": "#6f6f6f",
    "text_placeholder": "#a8a8a8",
    "text_inverse": "#ffffff",
    "text_link": "#0f62fe",
    # Sidebar (Gray 100)
    "sidebar_bg": "#161616",
    "sidebar_text": "#f4f4f4",
    "sidebar_text_2": "#a8a8a8",
    "sidebar_border": "#393939",
    "sidebar_active_bg": "#393939",
    "sidebar_active_text": "#ffffff",
    "sidebar_hover_bg": "#262626",
    # Brand
    "primary": "#0f62fe",
    "primary_hover": "#0043ce",
    "highlight": "#d0e2ff",
    # Status / alert palette
    "support_error": "#da1e28",
    "support_success": "#24a148",
    "support_warning": "#f1c21b",
    "support_info": "#0043ce",
    "caution_major": "#ff832b",
}

#: Categorical chart palette - Carbon's curated order, apply strictly in sequence.
CATEGORICAL: list[str] = [
    "#6929c4",  # Purple 70
    "#1192e8",  # Cyan 50
    "#005d5d",  # Teal 70
    "#9f1853",  # Magenta 70
    "#fa4d56",  # Red 50
    "#570408",  # Red 90
    "#198038",  # Green 60
    "#002d9c",  # Blue 80
    "#ee538b",  # Magenta 50
    "#b28600",  # Yellow 50
    "#009d9a",  # Teal 50
    "#012749",  # Cyan 90
    "#8a3800",  # Orange 70
    "#a56eff",  # Purple 50
]

#: Sequential Blue ramp (9-box heatmap).
SEQUENTIAL: list[str] = [
    "#edf5ff", "#d0e2ff", "#a6c8ff", "#78a9ff", "#4589ff",
    "#0f62fe", "#0043ce", "#002d9c", "#001d6c", "#001141",
]

#: Diverging Purple <-> Teal (fairness gap; zero = parity).
DIVERGING: list[str] = [
    "#1c0f30", "#31135e", "#491d8b", "#6929c4", "#8a3ffc",
    "#a56eff", "#be95ff", "#d4bbff", "#e8daff", "#f6f2ff",
    "#d9fbfb", "#9ef0f0", "#3ddbd9", "#009d9a", "#007d79",
    "#005d5d", "#00413e", "#002d2c", "#001a19",
]

#: Risk band -> (text colour, badge background). Text always passes WCAG AA on its own fill.
RISK_COLORS: dict[str, tuple[str, str]] = {
    "High": ("#da1e28", "#ffd7d9"),    # Red 60 on Red 20
    "Medium": ("#8a3800", "#ffdecf"),  # Orange 70 on Orange 20
    "Low": ("#198038", "#a7f0ba"),     # Green 60 on Green 20
}

#: Sequential Blue ramp, reversed, for a 3x3 nine-box heatmap.
NINEBOX_SCALE: list[list[str]] = [
    ["#e0e0e0", "#d0e2ff", "#a6c8ff"],
    ["#d0e2ff", "#78a9ff", "#4589ff"],
    ["#a6c8ff", "#4589ff", "#0f62fe"],
]

FONT_STACK = "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"


# ---------------------------------------------------------------------------
# 2. The 11 Streamlit overrides
# ---------------------------------------------------------------------------

CSS = f"""
<style>
/* (6) IBM Plex Sans + (2) remove top gradient bar ---------------------- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea {{
    font-family: {FONT_STACK};
}}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stStatusWidget"] {{ display: none !important; }}

/* (3) menu / footer / deploy chrome ------------------------------------ */
#MainMenu {{ visibility: hidden !important; }}
footer {{ visibility: hidden !important; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* (11) content width ---------------------------------------------------- */
.block-container {{
    max-width: 1440px;
    padding: 2rem 2rem 4rem 2rem;
}}
@media (max-width: 672px) {{
    .block-container {{ padding: 1rem 1rem 2rem 1rem; }}
}}

/* (1) Dark sidebar - Gray 100 theme ------------------------------------ */
[data-testid="stSidebar"] {{
    background-color: {TOKENS['sidebar_bg']};
    border-right: 1px solid {TOKENS['sidebar_border']};
}}
[data-testid="stSidebar"] * {{ color: {TOKENS['sidebar_text']}; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{ color: {TOKENS['sidebar_text']}; }}
[data-testid="stSidebar"] hr {{ border-color: {TOKENS['sidebar_border']}; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: {TOKENS['sidebar_text_2']} !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.45rem 0.75rem;
    border-radius: 0;
    border-left: 3px solid transparent;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background-color: {TOKENS['sidebar_hover_bg']};
}}
[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background-color: {TOKENS['sidebar_active_bg']};
    border-left-color: {TOKENS['primary']};
    color: {TOKENS['sidebar_active_text']};
}}

/* (4) Cards ------------------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background-color: {TOKENS['surface_2']};
    border: 1px solid {TOKENS['border_subtle']};
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 1px 2px rgba(22,22,22,0.06);
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    padding: 0;
}}
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {{
    padding: 1rem;
}}

/* (7) Buttons ----------------------------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {{
    background-color: {TOKENS['primary']};
    color: #ffffff;
    border: none;
    border-radius: 4px;
    height: 2.5rem;
    font-weight: 400;
    font-size: 0.875rem;
    box-shadow: none;
    transition: background-color 120ms ease;
}}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
    background-color: {TOKENS['primary_hover']};
    color: #ffffff;
    border: none;
}}
.stButton > button:focus,
.stDownloadButton > button:focus {{ box-shadow: 0 0 0 2px {TOKENS['primary']}; }}

/* (8) Tabs -------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; border-bottom: 1px solid {TOKENS['border_subtle']}; }}
.stTabs [data-baseweb="tab"] {{
    height: 2.75rem;
    padding: 0 0 0.75rem 0;
    font-size: 0.875rem;
    color: {TOKENS['text_secondary']};
}}
.stTabs [aria-selected="true"] {{
    color: {TOKENS['text_primary']};
    font-weight: 600;
}}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {TOKENS['primary']}; }}
.stTabs [data-baseweb="tab-border"] {{ background-color: {TOKENS['primary']}; }}

/* (9) Data tables ------------------------------------------------------- */
[data-testid="stDataFrame"] {{ border: 1px solid {TOKENS['border_subtle']}; border-radius: 8px; }}
[data-testid="stDataFrame"] * {{ font-family: {FONT_STACK}; font-size: 0.875rem; }}

/* Inputs --------------------------------------------------------------- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    border-radius: 4px !important;
    border-color: {TOKENS['border_subtle_2']} !important;
}}
input:focus, textarea:focus {{
    border-color: {TOKENS['primary']} !important;
    box-shadow: 0 0 0 1px {TOKENS['primary']} !important;
}}

/* App header ------------------------------------------------------------ */
.app-title {{
    font-size: 1.75rem;
    font-weight: 600;
    color: {TOKENS['text_primary']};
    line-height: 1.2;
    margin-bottom: 0.25rem;
}}
.app-subtitle {{
    font-size: 0.875rem;
    color: {TOKENS['text_helper']};
    margin-bottom: 1.5rem;
}}
.section-label {{
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {TOKENS['text_helper']};
    margin-bottom: 0.5rem;
}}

/* KPI cards ------------------------------------------------------------- */
.kpi-card {{
    background: {TOKENS['surface_2']};
    border: 1px solid {TOKENS['border_subtle']};
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 1px 2px rgba(22,22,22,0.06);
    height: 100%;
}}
.kpi-label {{
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {TOKENS['text_helper']};
    margin-bottom: 0.5rem;
}}
.kpi-value {{
    font-size: 1.75rem;
    font-weight: 600;
    color: {TOKENS['text_primary']};
    line-height: 1.1;
}}
.kpi-delta {{ font-size: 0.75rem; margin-top: 0.375rem; color: {TOKENS['text_helper']}; }}
.kpi-delta.good {{ color: {TOKENS['support_success']}; }}
.kpi-delta.bad  {{ color: {TOKENS['support_error']}; }}

/* Badges ---------------------------------------------------------------- */
.badge {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    line-height: 1.2;
}}

/* Helper text ----------------------------------------------------------- */
.small-note {{ font-size: 0.75rem; color: {TOKENS['text_helper']}; }}
</style>
"""


# ---------------------------------------------------------------------------
# 3. Plotly template
# ---------------------------------------------------------------------------


def carbon_plotly_template() -> go.layout.Template:
    """Build the shared Plotly template from the Carbon tokens."""
    axis = dict(
        showgrid=True,
        gridcolor=TOKENS["border_subtle"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(family=FONT_STACK, size=12, color=TOKENS["text_helper"]),
        title=dict(font=dict(family=FONT_STACK, size=12, color=TOKENS["text_helper"])),
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_STACK, size=13, color=TOKENS["text_primary"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=CATEGORICAL,
            margin=dict(l=8, r=8, t=32, b=8),
            xaxis=axis,
            yaxis=axis,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_STACK, size=12, color=TOKENS["text_secondary"]),
            ),
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor=TOKENS["border_subtle_2"],
                font=dict(family=FONT_STACK, size=12, color=TOKENS["text_primary"]),
            ),
            bargap=0.2,
        )
    )


def register_plotly_template() -> None:
    """Register the Carbon template globally with Plotly."""
    pio.templates["carbon"] = carbon_plotly_template()
    pio.templates.default = "carbon"


# ---------------------------------------------------------------------------
# 4. Component helpers
# ---------------------------------------------------------------------------


def inject_theme() -> None:
    """Inject the Carbon stylesheet into the running Streamlit app."""
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = "", delta_tone: str = "") -> str:
    """Render a Carbon-style KPI card as HTML."""
    tone = ""
    if delta_tone in {"good", "bad"}:
        tone = f" {delta_tone}"
    delta_html = f'<div class="kpi-delta{tone}">{delta}</div>' if delta else ""
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{delta_html}"
        "</div>"
    )


def kpi_row(cards: list[dict[str, Any]]) -> str:
    """Render a row of KPI cards. Each card: {label, value, delta, delta_tone}."""
    items = "".join(
        '<div style="flex:1 1 0; min-width:0;">'
        + kpi_card(
            c.get("label", ""),
            c.get("value", ""),
            c.get("delta", ""),
            c.get("delta_tone", ""),
        )
        + "</div>"
        for c in cards
    )
    return (
        '<div style="display:flex; gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap;">'
        + items
        + "</div>"
    )


def risk_badge(band: str) -> str:
    """Render an attrition risk band as a Carbon badge."""
    text_color, bg_color = RISK_COLORS.get(band, (TOKENS["text_secondary"], TOKENS["surface"]))
    return (
        f'<span class="badge" style="color:{text_color}; background:{bg_color};">{band}</span>'
    )


def risk_color(band: str) -> str:
    """Return the accent colour for a risk band."""
    return RISK_COLORS.get(band, (TOKENS["text_secondary"], TOKENS["surface"]))[0]


def section_label(text: str) -> str:
    """Render an uppercase section label."""
    return f'<div class="section-label">{text}</div>'


def app_header(title: str, subtitle: str = "") -> str:
    """Render the page title block."""
    sub = f'<div class="app-subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="app-title">{title}</div>{sub}'
