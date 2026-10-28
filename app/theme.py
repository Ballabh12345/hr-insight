"""
Aurora Glass — HR Insight design system (v5 · stitch_ecoglass_nature_ui spec).

Executive-grade intelligence interface. Azure / sky-blue / slate palette.
IBM Plex Sans typography. Frosted optical glass surfaces. Surgical precision.
NO heavy drifting-orb animations — motion is limited to fast, smooth transitions
that feel instant and responsive, not laggy.

Design source: stitch_ecoglass_nature_ui / aurora_glass / DESIGN.md
Reference screen: hr_insight_overview_attrition_intelligence / screen.png

Two modes (day = light azure default · night = deep navy):
  • Day  : ultra-pale azure canvas #f0f7ff, frosted white glass, azure accents
  • Night: deep navy #0b1120, dark glass, cyan/sky accents

Animation philosophy (fixing choppy / laggy issues from v4):
  • All CSS transitions use `ease` or `ease-out`, max 200ms (card hover 160ms)
  • NO backdrop-filter on the canvas itself (only on cards — GPU-composited layer)
  • NO @keyframes on body background (the single biggest performance killer)
  • NO filter:blur() on entrances — it forces a stacking context every frame
  • Entrance: simple opacity + translateY(10px), 300ms, no filter
  • Sidebar draw: 280ms cubic-bezier spring, no filter
  • Orbs: one SINGLE static radial-gradient on body::before, NO animation
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

MODES = ("day", "night")

# ---------------------------------------------------------------------------
# 1. Per-mode token sheets   (Aurora Glass palette from DESIGN.md)
# ---------------------------------------------------------------------------

_DAY: dict[str, Any] = {
    # Canvas
    "canvas": "#faf8ff",
    "canvas_gradient": "linear-gradient(160deg, #eef6ff 0%, #f4f7ff 50%, #faf8ff 100%)",
    # Ambient orb colours — ONE static layer, no animation
    "orb1": "rgba(14, 165, 233, 0.12)",    # sky
    "orb2": "rgba(99, 102, 241, 0.09)",    # indigo
    "orb3": "rgba(16, 185, 129, 0.07)",    # emerald
    # Glass card surfaces  (from DESIGN.md Level-1 / Level-2)
    "card":        "rgba(255, 255, 255, 0.80)",
    "card_2":      "rgba(255, 255, 255, 0.95)",
    "card_border": "rgba(255, 255, 255, 0.82)",
    "card_edge":   "rgba(255, 255, 255, 0.95)",
    "card_shadow": "0 8px 32px -8px rgba(14, 165, 233, 0.10), 0 1px 4px rgba(15, 23, 42, 0.05)",
    "card_shadow_hover": "0 16px 40px -12px rgba(14, 165, 233, 0.16), 0 2px 8px rgba(15, 23, 42, 0.06)",
    # Borders
    "border_subtle":   "rgba(14, 165, 233, 0.12)",
    "border_subtle_2": "rgba(14, 165, 233, 0.20)",
    "border_strong":   "rgba(2, 132, 199, 0.30)",
    # Text (DESIGN.md: on-surface #131b31, on-surface-variant #3f4850)
    "text_primary":     "#131b31",
    "text_secondary":   "#3f4850",
    "text_helper":      "#64748b",
    "text_placeholder": "#94a3b8",
    "text_inverse":     "#ffffff",
    "text_link":        "#0284c7",
    # Sidebar — navy glass (DESIGN.md: inverse-surface #283047)
    "sidebar_bg":        "rgba(11, 19, 41, 0.94)",
    "sidebar_bg_solid":  "#0b1329",
    "sidebar_text":      "#eef0ff",
    "sidebar_text_2":    "#8fa7c4",
    "sidebar_border":    "rgba(56, 189, 248, 0.15)",
    "sidebar_active_bg": "rgba(2, 132, 199, 0.85)",
    "sidebar_hover_bg":  "rgba(255, 255, 255, 0.07)",
    "pill_bg":           "rgba(255, 255, 255, 0.92)",
    "pill_text":         "#0b1329",
    "pill_border":       "rgba(2, 132, 199, 0.40)",
    "pill_shadow":       "0.18",
    # Interactive  (DESIGN.md: primary #006194 → sky-500 #0284c7)
    "primary":       "#0284c7",
    "primary_hover": "#0369a1",
    "primary_text":  "#0284c7",
    "primary_soft":  "rgba(14, 165, 233, 0.12)",
    "highlight":     "#0284c7",
    # Status
    "support_error":   "#e11d48",
    "support_success": "#059669",
    "support_warning": "#d97706",
    "support_info":    "#0284c7",
    "caution_major":   "#e11d48",
    # Chart chrome
    "grid":        "rgba(14, 165, 233, 0.08)",
    "tick":        "#64748b",
    "hover_bg":    "#ffffff",
    "hover_border": "rgba(2, 132, 199, 0.28)",
    # KPI values — plain dark ink (no gradient fill in day mode — matches screenshot)
    "value_grad_top":    "#0f172a",
    "value_grad_bottom": "#0f172a",
    "value_grad_mid":    "#0f172a",
    "track_bg": "rgba(14, 165, 233, 0.10)",
}

_NIGHT: dict[str, Any] = {
    "canvas": "#0b1120",
    "canvas_gradient": "linear-gradient(160deg, #0b1120 0%, #111827 50%, #0d1526 100%)",
    "orb1": "rgba(14, 165, 233, 0.18)",
    "orb2": "rgba(99, 102, 241, 0.14)",
    "orb3": "rgba(16, 185, 129, 0.10)",
    "card":        "rgba(255, 255, 255, 0.06)",
    "card_2":      "rgba(255, 255, 255, 0.10)",
    "card_border": "rgba(255, 255, 255, 0.11)",
    "card_edge":   "rgba(56, 189, 248, 0.20)",
    "card_shadow": "0 8px 32px -8px rgba(0, 0, 0, 0.50), 0 1px 4px rgba(0, 0, 0, 0.30)",
    "card_shadow_hover": "0 16px 40px -12px rgba(14, 165, 233, 0.22), 0 2px 8px rgba(0,0,0, 0.30)",
    "border_subtle":   "rgba(255, 255, 255, 0.08)",
    "border_subtle_2": "rgba(255, 255, 255, 0.14)",
    "border_strong":   "rgba(255, 255, 255, 0.22)",
    "text_primary":     "#eef0ff",
    "text_secondary":   "#a8b8d8",
    "text_helper":      "#7f8fa8",
    "text_placeholder": "#4e6080",
    "text_inverse":     "#0b1329",
    "text_link":        "#38bdf8",
    "sidebar_bg":        "rgba(6, 10, 22, 0.96)",
    "sidebar_bg_solid":  "#06080f",
    "sidebar_text":      "#eef0ff",
    "sidebar_text_2":    "#7f9ab8",
    "sidebar_border":    "rgba(56, 189, 248, 0.12)",
    "sidebar_active_bg": "rgba(2, 132, 199, 0.80)",
    "sidebar_hover_bg":  "rgba(255, 255, 255, 0.06)",
    "pill_bg":           "rgba(17, 24, 39, 0.95)",
    "pill_text":         "#eef0ff",
    "pill_border":       "rgba(56, 189, 248, 0.40)",
    "pill_shadow":       "0.45",
    "primary":       "#0ea5e9",
    "primary_hover": "#0284c7",
    "primary_text":  "#38bdf8",
    "primary_soft":  "rgba(14, 165, 233, 0.16)",
    "highlight":     "#38bdf8",
    "support_error":   "#f43f5e",
    "support_success": "#10b981",
    "support_warning": "#f59e0b",
    "support_info":    "#38bdf8",
    "caution_major":   "#f43f5e",
    "grid":        "rgba(255, 255, 255, 0.07)",
    "tick":        "#7f8fa8",
    "hover_bg":    "rgba(11, 19, 41, 0.90)",
    "hover_border": "rgba(56, 189, 248, 0.35)",
    "value_grad_top":    "#eef0ff",
    "value_grad_bottom": "#eef0ff",
    "value_grad_mid":    "#eef0ff",
    "track_bg": "rgba(255, 255, 255, 0.09)",
}

# Risk band -> (text colour, fill)
_DAY_RISK_COLORS: dict[str, tuple[str, str]] = {
    "High":   ("#e11d48", "rgba(244, 63, 94, 0.10)"),
    "Medium": ("#d97706", "rgba(245, 158, 11, 0.12)"),
    "Low":    ("#059669", "rgba(16, 185, 129, 0.10)"),
}
_NIGHT_RISK_COLORS: dict[str, tuple[str, str]] = {
    "High":   ("#f43f5e", "rgba(244, 63, 94, 0.14)"),
    "Medium": ("#f59e0b", "rgba(245, 158, 11, 0.12)"),
    "Low":    ("#10b981", "rgba(16, 185, 129, 0.12)"),
}
_DAY_RISK_SERIES: dict[str, str] = {
    "High": "#e11d48", "Medium": "#d97706", "Low": "#059669",
}
_NIGHT_RISK_SERIES: dict[str, str] = {
    "High": "#f43f5e", "Medium": "#f59e0b", "Low": "#10b981",
}
_DAY_ACCENTS: dict[str, str] = {
    "primary": "#0284c7", "high": "#e11d48", "medium": "#d97706", "low": "#059669",
}
_NIGHT_ACCENTS: dict[str, str] = {
    "primary": "#0ea5e9", "high": "#f43f5e", "medium": "#f59e0b", "low": "#10b981",
}

# Categorical — sky/indigo/emerald/rose/amber palette
_DAY_CATEGORICAL: list[str] = [
    "#0284c7", "#4f46e5", "#059669", "#e11d48", "#d97706",
    "#0ea5e9", "#6366f1", "#10b981", "#f43f5e", "#f59e0b",
    "#0891b2", "#7c3aed", "#047857", "#be123c",
]
_NIGHT_CATEGORICAL: list[str] = [
    "#38bdf8", "#818cf8", "#34d399", "#fb7185", "#fbbf24",
    "#7dd3fc", "#a5b4fc", "#6ee7b7", "#fda4af", "#fde68a",
    "#67e8f9", "#c4b5fd", "#6ee7b7", "#fca5a5",
]

_DAY_SEQUENTIAL: list[str] = [
    "#e0f2fe", "#bae6fd", "#7dd3fc", "#38bdf8", "#0ea5e9",
    "#0284c7", "#0369a1", "#075985", "#0c4a6e", "#082f49",
]
_NIGHT_SEQUENTIAL: list[str] = [
    "#082f49", "#0c4a6e", "#075985", "#0369a1", "#0284c7",
    "#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd", "#e0f2fe",
]

# 9-box: azure progressive (matching DESIGN.md chromatic scale)
_DAY_NINEBOX: list[list[str]] = [
    ["#e0f2fe", "#bae6fd", "#7dd3fc"],
    ["#bae6fd", "#38bdf8", "#0ea5e9"],
    ["#7dd3fc", "#0ea5e9", "#0284c7"],
]
_NIGHT_NINEBOX: list[list[str]] = [
    ["#082f49", "#0c4a6e", "#075985"],
    ["#0c4a6e", "#0369a1", "#0284c7"],
    ["#075985", "#0284c7", "#0ea5e9"],
]

_DAY_DIVERGING: list[str] = [
    "#1e1b4b", "#312e81", "#4338ca", "#4f46e5", "#6366f1",
    "#818cf8", "#a5b4fc", "#e0e7ff",
    "#064e3b", "#065f46", "#059669", "#10b981", "#34d399",
    "#6ee7b7", "#a7f3d0", "#d1fae5",
]
_NIGHT_DIVERGING: list[str] = [
    "#160c38", "#221640", "#331d5e", "#4338ca", "#6366f1",
    "#a5b4fc", "#c7d2fe", "#ede9fe",
    "#064e3b", "#065f46", "#059669", "#34d399", "#6ee7b7",
    "#a7f3d0", "#d1fae5", "#ecfdf5",
]

MODE_SHEETS: dict[str, dict[str, Any]] = {
    "day": {
        "tokens": _DAY,
        "risk_colors": _DAY_RISK_COLORS,
        "risk_series": _DAY_RISK_SERIES,
        "accents": _DAY_ACCENTS,
        "categorical": _DAY_CATEGORICAL,
        "sequential": _DAY_SEQUENTIAL,
        "ninebox": _DAY_NINEBOX,
        "diverging": _DAY_DIVERGING,
    },
    "night": {
        "tokens": _NIGHT,
        "risk_colors": _NIGHT_RISK_COLORS,
        "risk_series": _NIGHT_RISK_SERIES,
        "accents": _NIGHT_ACCENTS,
        "categorical": _NIGHT_CATEGORICAL,
        "sequential": _NIGHT_SEQUENTIAL,
        "ninebox": _NIGHT_NINEBOX,
        "diverging": _NIGHT_DIVERGING,
    },
}

# Mutable module state
_mode: str = "day"
TOKENS: dict[str, str] = dict(_DAY)
RISK_COLORS: dict[str, tuple[str, str]] = dict(_DAY_RISK_COLORS)
RISK_SERIES: dict[str, str] = dict(_DAY_RISK_SERIES)
ACCENT_COLORS: dict[str, str] = dict(_DAY_ACCENTS)
CATEGORICAL: list[str] = list(_DAY_CATEGORICAL)
SEQUENTIAL: list[str] = list(_DAY_SEQUENTIAL)
NINEBOX_SCALE: list[list[str]] = [row[:] for row in _DAY_NINEBOX]
DIVERGING: list[str] = list(_DAY_DIVERGING)

_DEFAULT_KPI_ACCENT: str = "#0284c7"

# IBM Plex Sans — matches DESIGN.md typography spec exactly
FONT_STACK = "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
MONO_STACK = "'IBM Plex Mono', 'Fira Code', monospace"


def mode() -> str:
    return _mode


def set_mode(value: str) -> str:
    global _mode, TOKENS, RISK_COLORS, RISK_SERIES, ACCENT_COLORS
    global CATEGORICAL, SEQUENTIAL, NINEBOX_SCALE, DIVERGING, _DEFAULT_KPI_ACCENT
    if value not in MODES:
        raise ValueError(f"unknown mode: {value}")
    _mode = value
    sheet = MODE_SHEETS[value]
    TOKENS = dict(sheet["tokens"])
    RISK_COLORS = dict(sheet["risk_colors"])
    RISK_SERIES = dict(sheet["risk_series"])
    ACCENT_COLORS = dict(sheet["accents"])
    CATEGORICAL = list(sheet["categorical"])
    SEQUENTIAL = list(sheet["sequential"])
    NINEBOX_SCALE = [row[:] for row in sheet["ninebox"]]
    DIVERGING = list(sheet["diverging"])
    _DEFAULT_KPI_ACCENT = ACCENT_COLORS["primary"]
    return _mode


# ---------------------------------------------------------------------------
# 2. Stylesheet  (Aurora Glass v5 — performance-first, spec-faithful)
# ---------------------------------------------------------------------------

def build_css() -> str:
    """Build the full Aurora Glass stylesheet for the active mode.

    Performance rules applied:
    - body::before is a STATIC radial-gradient background (no animation)
    - backdrop-filter only on glass cards (GPU-composited, efficient)
    - entrance motion: opacity + translateY only — no filter:blur
    - all transitions ≤ 200ms; sidebar drawer ≤ 280ms
    - will-change used sparingly, only where GPU promotion is certain
    """
    t = TOKENS
    return f"""
<style>
/* Aurora Glass v5 — stitch_ecoglass_nature_ui spec ========================= */

/* Fonts ------------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, button, input, textarea, select {{
    font-family: {FONT_STACK};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
code, pre, .mono {{ font-family: {MONO_STACK}; }}

/* Canvas — STATIC gradient + ambient orbs (no @keyframes) ----------------- */
body {{
    background: {t['canvas_gradient']};
    min-height: 100vh;
}}
body::before {{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background:
        radial-gradient(60rem 60rem at 90% 5%,  {t['orb1']}, transparent 70%),
        radial-gradient(50rem 50rem at 10% 95%,  {t['orb2']}, transparent 70%),
        radial-gradient(40rem 40rem at 55% 50%,  {t['orb3']}, transparent 70%);
    /* NO animation — this is the single largest perf fix */
}}
.stApp {{ background: transparent !important; }}

/* Streamlit Chrome -------------------------------------------------------- */
[data-testid="stDecoration"]      {{ display: none !important; }}
[data-testid="stStatusWidget"]    {{ display: none !important; }}
[data-testid="stAppDeployButton"] {{ display: none !important; }}
[data-testid="stMainMenu"],
[data-testid="stMainMenuButton"],
#MainMenu {{ display: none !important; }}
footer {{ visibility: hidden !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1.25rem !important; }}

/* Sidebar Drawer — fast spring, no filter blur ---------------------------- */
[data-testid="stSidebar"] {{
    transition:
        transform    280ms cubic-bezier(0.22, 1, 0.36, 1),
        min-width    280ms cubic-bezier(0.22, 1, 0.36, 1),
        max-width    280ms cubic-bezier(0.22, 1, 0.36, 1),
        top          240ms ease,
        padding      240ms ease,
        background-color 0s,
        border-right-color 200ms ease,
        box-shadow   200ms ease !important;
    will-change: min-width, max-width, transform;
}}
[data-testid="stSidebar"][aria-expanded="false"] {{
    overflow: hidden !important;
    border-right-color: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    transition:
        transform          280ms cubic-bezier(0.22, 1, 0.36, 1),
        min-width          280ms cubic-bezier(0.22, 1, 0.36, 1),
        max-width          280ms cubic-bezier(0.22, 1, 0.36, 1),
        top                240ms ease,
        padding            240ms ease,
        background-color   160ms ease 200ms,
        border-right-color 140ms ease 180ms,
        box-shadow         180ms ease 100ms !important;
}}
/* Content opacity only — NO blur filter (too expensive during drag) */
[data-testid="stSidebar"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
    transition: opacity 200ms ease 40ms, transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
}}
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarUserContent"] {{
    opacity: 0;
    transform: translateX(-16px);
}}
[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarContent"],
[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarUserContent"] {{
    opacity: 1;
    transform: translateX(0);
}}

/* Collapse / expand controls */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"] {{
    visibility: visible !important;
    color: {t['sidebar_text_2']} !important;
    border-radius: 8px !important;
    transition: background 150ms ease, color 150ms ease !important;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:hover {{
    background: rgba(56, 189, 248, 0.15) !important;
    color: {t['sidebar_text']} !important;
}}

/* Mobile expand pill */
[data-testid="stExpandSidebarButton"] {{
    background-color: {t['pill_bg']} !important;
    color: {t['pill_text']} !important;
    border: 1px solid {t['pill_border']} !important;
    border-radius: 999px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, {t['pill_shadow']}) !important;
    padding: 0.4rem 0.875rem 0.4rem 0.65rem !important;
    display: inline-flex !important;
    align-items: center !important;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: transform 150ms ease, box-shadow 150ms ease;
}}
[data-testid="stExpandSidebarButton"]::after {{
    content: "Sidebar";
    margin-left: 0.45rem;
    white-space: nowrap;
}}
[data-testid="stExpandSidebarButton"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.22) !important;
}}

/* Appearance chips */
[data-testid="stSidebar"] .stButtonGroup [role="radio"] {{
    background: transparent !important;
    color: {t['sidebar_text_2']} !important;
    border: 1px solid {t['sidebar_border']} !important;
    border-radius: 8px !important;
    transition: background 150ms ease, color 150ms ease !important;
}}
[data-testid="stSidebar"] .stButtonGroup [role="radio"][aria-checked="true"] {{
    background: rgba(2, 132, 199, 0.70) !important;
    color: #ffffff !important;
    border-color: rgba(56, 189, 248, 0.50) !important;
}}

/* Layout ------------------------------------------------------------------ */
.block-container {{
    max-width: 1440px;
    padding: 1.25rem 2rem 4rem 2rem;
}}
@media (max-width: 672px) {{
    .block-container {{ padding: 0.75rem 1rem 2rem 1rem; }}
}}

/* Glass Sidebar  (DESIGN.md Level-3 navy glass) --------------------------- */
[data-testid="stSidebar"] {{
    background-color: {t['sidebar_bg']} !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.3);
    backdrop-filter: blur(20px) saturate(1.3);
    border-right: 1px solid {t['sidebar_border']} !important;
    box-shadow: 2px 0 24px rgba(0, 0, 0, 0.18) !important;
}}
[data-testid="stSidebar"] * {{ color: {t['sidebar_text']}; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{ color: {t['sidebar_text']}; }}
[data-testid="stSidebar"] hr {{ border-color: {t['sidebar_border']}; }}

/* Nav radio items */
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 0.5rem 0.875rem;
    border-radius: 8px;
    border-left: 3px solid transparent;
    cursor: pointer;
    display: block;
    margin: 0.1rem 0.5rem;
    font-size: 0.875rem;
    font-weight: 500;
    transition: background-color 150ms ease, border-left-color 150ms ease;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{
    display: none !important;
}}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div {{ align-items: center; }}
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"],
[data-testid="stSidebar"] [role="radiogroup"] input[type="checkbox"] {{
    position: absolute; opacity: 0; width: 1px; height: 1px;
}}
[data-testid="stSidebar"] [role="radiogroup"] > div {{ gap: 0 !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background-color: {t['sidebar_hover_bg']};
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background-color: {t['sidebar_active_bg']};
    border-left-color: rgba(56, 189, 248, 0.70);
    color: #ffffff;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible) {{
    outline: 2px solid rgba(56, 189, 248, 0.60);
    outline-offset: -2px;
}}

/* Glass Cards  (DESIGN.md Level-1 / Level-2) ------------------------------ */
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: {t['card']};
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid {t['card_border']};
    border-radius: 12px;
    padding: 1rem;
    box-shadow: {t['card_shadow']}, inset 0 1px 0 {t['card_edge']};
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}}
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:hover,
[data-testid="stVerticalBlockBorderWrapper"]:hover > div {{
    transform: translateY(-2px);
    box-shadow: {t['card_shadow_hover']}, inset 0 1px 0 {t['card_edge']};
    border-color: rgba(14, 165, 233, 0.22);
}}

/* Buttons ----------------------------------------------------------------- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {{
    background-color: {t['card_2']};
    color: {t['text_primary']};
    border: 1px solid {t['border_strong']};
    border-radius: 8px;
    height: 2.5rem;
    font-weight: 600;
    font-size: 0.875rem;
    letter-spacing: 0.01em;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    transition: background-color 150ms ease, border-color 150ms ease,
                transform 150ms ease, box-shadow 150ms ease;
}}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
    background-color: {t['primary_soft']};
    border-color: {t['primary']};
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.18);
}}
.stButton > button:focus,
.stDownloadButton > button:focus {{
    box-shadow: 0 0 0 3px {t['primary_soft']};
}}
button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
    color: #ffffff;
    border: none;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35), inset 0 1px 0 rgba(255,255,255,0.22);
}}
button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
    color: #ffffff;
    border: none;
    box-shadow: 0 6px 18px rgba(2, 132, 199, 0.45);
    transform: translateY(-1px);
}}
button[kind="tertiary"],
.stFormSubmitButton > button[kind="tertiary"] {{
    background-color: transparent;
    color: {t['support_error']};
    border: 1px solid {t['support_error']};
    box-shadow: none;
}}
button[kind="tertiary"]:hover {{
    background-color: rgba(225, 29, 72, 0.06);
    box-shadow: 0 2px 8px rgba(225, 29, 72, 0.15);
}}
button[kind="tertiary"]:focus {{ box-shadow: 0 0 0 3px rgba(225, 29, 72, 0.20); }}

/* Tabs -------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.5rem;
    border-bottom: 1px solid {t['border_subtle']};
}}
.stTabs [data-baseweb="tab"] {{
    height: 2.75rem;
    padding: 0 0 0.75rem 0;
    font-size: 0.875rem;
    font-weight: 500;
    color: {t['text_secondary']};
    transition: color 150ms ease;
}}
.stTabs [aria-selected="true"] {{ color: {t['text_primary']}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {t['primary']}; height: 2px; }}
.stTabs [data-baseweb="tab-border"]    {{ background-color: {t['primary']}; }}

/* Data Tables ------------------------------------------------------------- */
[data-testid="stDataFrame"] {{
    border: 1px solid {t['card_border']};
    border-radius: 12px;
    box-shadow: {t['card_shadow']};
    overflow: hidden;
}}
[data-testid="stDataFrame"] * {{ font-family: {FONT_STACK}; font-size: 0.875rem; }}

/* Inputs ------------------------------------------------------------------ */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    border-radius: 8px !important;
    border-color: {t['border_subtle_2']} !important;
    background-color: {t['card_2']};
    font-family: {FONT_STACK};
    transition: border-color 150ms ease, box-shadow 150ms ease !important;
}}
input:focus, textarea:focus {{
    border-color: {t['primary']} !important;
    box-shadow: 0 0 0 3px {t['primary_soft']} !important;
}}
label, .stMarkdown p {{ color: {t['text_secondary']}; font-size: 0.875rem; }}

/* Inline Notifications ---------------------------------------------------- */
[data-testid="stAlert"] > div {{
    background: {t['card']} !important;
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    border: 1px solid {t['card_border']} !important;
    border-left: 3px solid {t['primary']} !important;
    border-radius: 0 10px 10px 0 !important;
    color: {t['text_primary']};
}}

/* Chat -------------------------------------------------------------------- */
[data-testid="stChatMessage"] {{
    background: {t['card']};
    -webkit-backdrop-filter: blur(14px);
    backdrop-filter: blur(14px);
    border: 1px solid {t['card_border']};
    border-radius: 12px;
    padding: 0.5rem 0.25rem;
    margin-bottom: 0.5rem;
}}
[data-testid="stChatMessage"] [data-testid="stAvatar"] {{ display: none; }}
[data-testid="stBottom"] > div {{ background: transparent; }}
[data-testid="stChatInput"] {{ border-radius: 12px !important; }}
[data-testid="stChatInput"] > div {{
    background: {t['card_2']} !important;
    border: 1px solid {t['card_border']} !important;
    border-radius: 12px;
    box-shadow: {t['card_shadow']};
}}

/* Expanders --------------------------------------------------------------- */
[data-testid="stExpander"] {{
    background: {t['card']};
    -webkit-backdrop-filter: blur(14px);
    backdrop-filter: blur(14px);
    border: 1px solid {t['card_border']} !important;
    border-radius: 12px !important;
    box-shadow: {t['card_shadow']};
    overflow: visible;
    transition: box-shadow 160ms ease;
}}
[data-testid="stExpander"]:hover {{ box-shadow: {t['card_shadow_hover']}; }}
[data-testid="stExpander"] summary {{
    color: {t['text_primary']} !important;
    font-weight: 600;
    font-size: 0.9375rem;
}}
[data-testid="stExpander"] details {{ background: transparent !important; border: none !important; }}

/* Page Title / Labels ----------------------------------------------------- */
.app-title {{
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.015em;
    color: {t['text_primary']};
    line-height: 1.18;
    margin-bottom: 0.3rem;
}}
.app-subtitle {{
    font-size: 0.9375rem;
    color: {t['text_helper']};
    margin-bottom: 1.4rem;
    font-weight: 400;
    line-height: 1.55;
}}
.section-label {{
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {t['text_helper']};
    margin: 0.75rem 0 0.5rem 0;
}}

/* KPI Cards --------------------------------------------------------------- */
.kpi-card {{
    background: {t['card']};
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid {t['card_border']};
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    box-shadow: {t['card_shadow']}, inset 0 1px 0 {t['card_edge']};
    height: 100%;
    transition: transform 160ms ease, box-shadow 160ms ease;
    position: relative;
    overflow: hidden;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: {t['card_shadow_hover']}, inset 0 1px 0 {t['card_edge']};
}}
.kpi-label {{
    font-size: 0.6875rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {t['text_helper']};
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}
.kpi-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}}
/* Pulse is FAST and GPU-composited (only opacity + transform) */
.kpi-dot.pulse {{
    animation: kpiPulse 2s ease-in-out infinite;
}}
@keyframes kpiPulse {{
    0%, 100% {{ opacity: 1;    transform: scale(1);    }}
    50%       {{ opacity: 0.50; transform: scale(1.20); }}
}}
.kpi-value {{
    font-size: 2.25rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {t['text_primary']};
    line-height: 1.08;
    font-variant-numeric: tabular-nums;
    overflow-wrap: anywhere;
}}
/* Long textual values (e.g. model version "logistic_regression-sigmoid-42")
   drop to a compact monospace size instead of pushing out of the card. */
.kpi-value.long {{
    font-size: 1.05rem;
    letter-spacing: 0;
    line-height: 1.35;
    font-family: "Consolas", "JetBrains Mono", monospace;
}}
.kpi-delta {{ font-size: 0.75rem; margin-top: 0.4rem; color: {t['text_helper']}; }}
.kpi-delta.good {{ color: {t['support_success']}; font-weight: 600; }}
.kpi-delta.bad  {{ color: {t['support_error']};   font-weight: 600; }}

/* Badges ------------------------------------------------------------------ */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    line-height: 1.2;
    border: 1px solid {t['card_border']};
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
}}
.badge::before {{
    content: "";
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.9;
    flex-shrink: 0;
}}

/* Risk Meter -------------------------------------------------------------- */
.risk-meter {{ margin: 0.5rem 0.25rem 0.35rem 0.25rem; }}
.risk-meter .track {{
    position: relative;
    height: 10px;
    border-radius: 6px;
    background: {t['track_bg']};
    border: 1px solid {t['border_subtle_2']};
    overflow: hidden;
}}
.risk-meter .fill {{
    position: absolute;
    top: 0; left: 0; bottom: 0;
    border-radius: 6px;
    /* Simple grow — opacity+width, no blur filter */
    animation: meterGrow 600ms ease-out both;
}}
@keyframes meterGrow {{ from {{ width: 0; opacity: 0.6; }} to {{ opacity: 1; }} }}
.risk-meter .ticks {{
    display: flex;
    justify-content: space-between;
    font-size: 0.6875rem;
    color: {t['text_helper']};
    margin-top: 0.3rem;
    font-variant-numeric: tabular-nums;
}}

/* Small note -------------------------------------------------------------- */
.small-note {{ font-size: 0.75rem; color: {t['text_helper']}; line-height: 1.5; overflow-wrap: anywhere; }}

/* Staggered Entrance — opacity + translate only, no filter:blur ----------- */
[data-testid="stMain"] [data-testid="stElementContainer"] {{
    animation: rise 300ms ease-out both;
}}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(1)  {{ animation-delay: 0ms;   }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(2)  {{ animation-delay: 30ms;  }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(3)  {{ animation-delay: 60ms;  }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(4)  {{ animation-delay: 90ms;  }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(5)  {{ animation-delay: 115ms; }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(6)  {{ animation-delay: 140ms; }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(7)  {{ animation-delay: 165ms; }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(8)  {{ animation-delay: 188ms; }}
.stMainBlockContainer .stVerticalBlock > .stElementContainer:nth-child(n+9) {{ animation-delay: 210ms; }}
/* ONLY opacity + translateY — NO filter:blur — this is the key perf fix */
@keyframes rise {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0);    }}
}}

/* prefers-reduced-motion -------------------------------------------------- */
@media (prefers-reduced-motion: reduce) {{
    [data-testid="stMain"] [data-testid="stElementContainer"] {{ animation: none; }}
    .risk-meter .fill  {{ animation: none; }}
    .kpi-dot.pulse     {{ animation: none; }}
    * {{ transition-duration: 1ms !important; }}
    [data-testid="stSidebar"],
    [data-testid="stSidebar"][aria-expanded="false"],
    [data-testid="stSidebar"][aria-expanded="true"],
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
        transition-duration: 1ms !important;
        transition-delay: 0ms !important;
    }}
}}
</style>
"""


# ---------------------------------------------------------------------------
# 3. Plotly template
# ---------------------------------------------------------------------------

def carbon_plotly_template() -> go.layout.Template:
    """Build the Aurora Glass Plotly template."""
    t = TOKENS
    axis = dict(
        showgrid=True,
        gridcolor=t["grid"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(family=FONT_STACK, size=12, color=t["tick"]),
        title=dict(font=dict(family=FONT_STACK, size=12, color=t["tick"])),
    )
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_STACK, size=13, color=t["text_primary"]),
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
                font=dict(family=FONT_STACK, size=12, color=t["text_secondary"]),
            ),
            hoverlabel=dict(
                bgcolor=t["hover_bg"],
                bordercolor=t["hover_border"],
                font=dict(family=FONT_STACK, size=12, color=t["text_primary"]),
            ),
            bargap=0.35,
        )
    )


def register_plotly_template() -> None:
    pio.templates["hr_insight"] = carbon_plotly_template()
    pio.templates.default = "hr_insight"


# ---------------------------------------------------------------------------
# 4. Injection + component helpers
# ---------------------------------------------------------------------------

def inject_theme() -> None:
    import streamlit as st
    st.markdown(build_css(), unsafe_allow_html=True)


def chart(fig, width="stretch", config=None, height=None, **kwargs):
    """Render a Plotly chart in the Aurora Glass palette."""
    import streamlit as st

    t = TOKENS
    fig.update_layout(
        template=carbon_plotly_template(),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": t["text_primary"]},
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if height is not None:
        fig.update_layout(height=height)
    if config is None:
        config = {"displayModeBar": False}
    selectable_kwargs = {
        k: kwargs.pop(k) for k in ("key", "on_select", "selection_mode") if k in kwargs
    }
    return st.plotly_chart(fig, width=width, config=config, theme=None, **selectable_kwargs, **kwargs)


def kpi_card(label: str, value: str, delta: str = "", delta_tone: str = "", accent: str = "") -> str:
    """Render an Aurora Glass KPI card."""
    top = ACCENT_COLORS.get(accent, accent) if accent else _DEFAULT_KPI_ACCENT
    dot = ""
    if top != _DEFAULT_KPI_ACCENT:
        pulse = " pulse" if accent in ("high",) else ""
        dot = f'<span class="kpi-dot{pulse}" style="color:{top};background:{top};"></span>'
    tone = ""
    if delta_tone in {"good", "bad"}:
        tone = f" {delta_tone}"
    delta_html = f'<div class="kpi-delta{tone}">{delta}</div>' if delta else ""
    # Long textual values (model versions, labels) switch to the compact style -
    # a 2.25rem unbreakable token would overflow the card otherwise.
    value_class = "kpi-value long" if "<" not in value and len(value) > 14 else "kpi-value"
    # Thin accent top-border (like DESIGN.md specular edge)
    border_style = f' style="border-top:3px solid {top};"' if top else ""
    return (
        f'<div class="kpi-card"{border_style}>'
        f'<div class="kpi-label">{dot}{label}</div>'
        f'<div class="{value_class}">{value}</div>'
        f"{delta_html}"
        "</div>"
    )


def kpi_row(cards: list[dict[str, Any]]) -> str:
    """Render a flex row of Aurora Glass KPI cards."""
    items = "".join(
        '<div style="flex:1 1 0; min-width:0;">'
        + kpi_card(
            c.get("label", ""),
            c.get("value", ""),
            c.get("delta", ""),
            c.get("delta_tone", ""),
            c.get("accent", ""),
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
    """Render a risk band as a glass pill."""
    text_color, bg_color = RISK_COLORS.get(band, (TOKENS["text_secondary"], TOKENS["card"]))
    return (
        f'<span class="badge" style="color:{text_color}; background:{bg_color};">{band}</span>'
    )


def risk_color(band: str) -> str:
    return RISK_COLORS.get(band, (TOKENS["text_secondary"], TOKENS["card"]))[0]


def risk_meter(probability: float, band: str) -> str:
    """Render the attrition risk as a CSS Aurora Glass meter bar."""
    pct = float(probability) * 100.0
    color = risk_color(band)
    zones = max(0.0, min(100.0, pct))
    badge = risk_badge(band)
    return (
        f'<div class="risk-meter">'
        f'<div class="track"><div class="fill" style="width:{zones:.0f}%; background:{color};"></div></div>'
        f'<div class="ticks"><span>0</span><span>25</span><span>55</span><span>100</span></div>'
        f"</div>"
        f'<div style="display:flex; align-items:center; gap:0.75rem; margin-top:0.6rem;">'
        f'<span style="font-size:2rem; font-weight:700; letter-spacing:-0.02em; '
        f'color:{TOKENS["text_primary"]}; font-variant-numeric:tabular-nums;">'
        f"{pct:.0f}%</span>{badge}"
        f"</div>"
        f'<div class="small-note" style="margin-top:0.5rem;">Bands: &lt;25% &middot; 25–55% &middot; &gt;55%</div>'
    )


def section_label(text: str) -> str:
    return f'<div class="section-label">{text}</div>'


def app_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="app-subtitle">{subtitle}</div>' if subtitle else ""
    return f'<div class="app-title">{title}</div>{sub}'


def heat_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return SEQUENTIAL[0]
    t = (value - vmin) / (vmax - vmin)
    return SEQUENTIAL[int(round(t * (len(SEQUENTIAL) - 1)))]


def heat_text_color(value: float, vmin: float, vmax: float) -> str:
    return TOKENS["text_primary"] if _mode == "day" else "#eef0ff"


def matrix_html(
    rows: list[str],
    cols: list[str],
    cells: dict[tuple[str, str], tuple[str, str, float]],
    vmin: float,
    vmax: float,
    row_caption: str = "",
    col_caption: str = "",
) -> str:
    """Render a 3x3 talent matrix as Aurora Glass tiles (matching DESIGN.md 9-box spec)."""
    t = TOKENS
    out: list[str] = [
        '<div style="display:grid; grid-template-columns: 150px repeat(%d, 1fr); '
        'gap:6px; margin-top:0.5rem;">' % len(cols)
    ]
    out.append(
        f'<div style="display:flex;align-items:flex-end;justify-content:flex-end;'
        f'padding:0 0.5rem 0.35rem 0;font-size:0.6875rem;font-weight:600;letter-spacing:0.06em;'
        f'text-transform:uppercase;color:{t["text_helper"]};">'
        f"{col_caption}</div>"
    )
    for col in cols:
        out.append(
            f'<div style="font-size:0.6875rem;font-weight:600;letter-spacing:0.06em;'
            f'text-transform:uppercase;color:{t["text_helper"]};'
            f'padding:0 0.5rem 0.35rem 0;text-align:center;">{col}</div>'
        )
    for row in rows:
        out.append(
            f'<div style="display:flex;align-items:center;justify-content:flex-end;'
            f'padding-right:0.75rem;font-size:0.6875rem;font-weight:600;letter-spacing:0.06em;'
            f'text-transform:uppercase;color:{t["text_helper"]};height:88px;">{row}</div>'
        )
        for col in cols:
            label, detail, value = cells.get((row, col), ("-", "", vmin))
            bg = heat_color(value, vmin, vmax)
            fg = heat_text_color(value, vmin, vmax)
            out.append(
                f'<div style="background:{bg};color:{fg};height:88px;'
                f'border-radius:8px;padding:0.65rem 0.75rem;'
                f'box-shadow:0 2px 8px rgba(14,165,233,0.08);'
                f'border:1px solid rgba(255,255,255,0.60);'
                f'display:flex;flex-direction:column;justify-content:center;gap:0.2rem;'
                f'transition:transform 160ms ease, box-shadow 160ms ease;">'
                f'<div style="font-size:0.8125rem;font-weight:600;line-height:1.25;">{label}</div>'
                f'<div style="font-size:0.6875rem;opacity:0.80;">{detail}</div>'
                f"</div>"
            )
    out.append("</div>")
    if row_caption:
        out.append(f'<div class="small-note" style="margin-top:0.6rem;">{row_caption}</div>')
    return "".join(out)
