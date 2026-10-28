# HR Insight — Design System (`design.md`)

**Structure source:** [IBM Carbon Design System](https://carbondesignsystem.com/) (open source, Apache-2.0).
The layout discipline — tokenised roles, the 8px mini-unit, IBM Plex type, layer separation,
audit-trail documentation — is Carbon. **Applied to:** Streamlit 1.65.0, Plotly 7.1.0, IBM Plex Sans.

> The original build used Carbon's White theme + Blue 60 brand. It was re-skinned as
> **Palette v2 (§0.5)** — dark canvas, lavender/chartreuse/tomato — after a hands-on review
> found the all-white version read as a wireframe. It has now been rebuilt as
> **Palette v3 (§0)** — "Aurora Glass" — after a design review asked for a light-blue,
> premium-but-simple glassmorphism look with a day/night toggle. The Carbon value sheet is
> retained in §1–§9 as the audit trail and fallback; where §0 supersedes a value below, §0 wins.

---

## 0. Palette v3 — Aurora Glass (current)

A light-blue glassmorphism layer on the same Carbon layout discipline: an ambient
gradient-orb canvas, frosted translucent cards, a deep-navy glass sidebar in **both**
modes, gradient-text numbers, and a real day/night toggle. Chartreuse and tomato are
retired from general use; the risk palette is a single red/amber/green family in both modes.

### 0.1 How the mode switch works (no CSS hacks)

* The toggle is a **real widget**: `st.segmented_control("Appearance", …)` at the top of
  the sidebar. Streamlit reruns the script, `theme.set_mode()` swaps the exported token
  sheets (`TOKENS`, `RISK_COLORS`, `SEQUENTIAL`, …), `theme.inject_theme()` re-injects the
  rebuilt stylesheet, and `theme.register_plotly_template()` rebuilds the Plotly template
  for the same mode. Everything on the page — including charts — follows in one run.
* **Day is the default**; the choice is persisted in the URL as `?mode=day|night` via
  `st.query_params`.
* `.streamlit/config.toml` keeps `base = "light"` fixed (Streamlit cannot switch its native
  base theme at runtime). Night mode is a full CSS overlay; the one place the base shows
  through under night styling — the data grid — intentionally keeps its light sheet, which
  reads as a bright data card on the dark canvas.

### 0.2 Day tokens (default)

| Role | Value | Contrast | Notes |
|---|---|---|---|
| Canvas | gradient `#eaf2ff → #f7fbff → #eef6ff` | — | with sky/cyan ambient orbs (animated) |
| Glass card fill | `rgba(255,255,255,0.58)` + blur 18px | body ink **~14:1** | nested surfaces raise to `0.82` |
| Card border / top highlight | `rgba(255,255,255,0.70)` / inset `0.90` | — | the "glass edge" |
| Text primary / secondary / helper | `#0b1b34` / `#3c5372` / `#5b6b8a` | 14.9 / 7.6 / 4.8 on card | navy ink family |
| Primary fill | `#2563eb` | white text **5.2:1** ✅ AA | hover deepens to `#1d4ed8` (6.7:1) |
| Primary text / links / chart lines | `#1d4ed8` | 7.4:1 on card ✅ | the deep end, never the neon |
| KPI value gradient | `#1d4ed8 → #60a5fa` (lighted by transparent-fill) | dark ends ≥ AA | on all five KPI cards |

### 0.3 Night tokens

| Role | Value | Contrast | Notes |
|---|---|---|---|
| Canvas | gradient `#0a1020 → #101a33 → #0b1226` | — | indigo/violet/teal ambient orbs (animated) |
| Glass card fill | `rgba(255,255,255,0.06)` + blur 18px | light ink **11:1** | border lifts to `0.22` |
| Text primary / secondary / helper | `#e6efff` / `#a7b8d8` / `#8296b8` | 14.9 / 7.9 / 5.2 on card | the same roles, inverted |
| Primary fill | `#2563eb` (shared with day) | white **5.2:1** ✅ | hover `#1d4ed8` (6.7:1) — night keeps AA by sharing the day blues |
| Accent text / lines | `#7dd3fc` | 10.9:1 on card ✅ | sky family owns "interactive text" |
| KPI value gradient | `#dbeafe → #7dd3fc` | — | bright-on-glass |

### 0.4 The glass recipe (load-bearing)

Glass only reads as glass over a **non-flat** background. Every card therefore needs all
five layers, in this order:

1. translucent fill (token above);
2. `backdrop-filter: blur(18px) saturate(1.35)` (sidebar uses 22px);
3. 1px light border (`rgba(255,255,255,…)` per mode);
4. 1px inset top highlight — `inset 0 1px 0 rgba(255,255,255,0.9)` day / `0.22` night;
5. one soft shadow — `0 8px 32px rgba(37,99,235,0.12)` day / `rgba(2,6,18,0.45)` night.

The orb layer lives on `body::before` (fixed, `z-index:-1`), so every card's
`backdrop-filter` picks it up. Radii: 16px cards, 10px buttons/inputs, 8px charts, 999px badges.

### 0.5 Risk palette + status accents

| Band | Marks & badges (day) | Marks & badges (night) | Badge text on tint/contrast |
|---|---|---|---|
| High | `#e11d48` | `#fb7185` | `#be123c` on 12% rose tint ✅ / `#fb7185` on 16% ✅ |
| Medium | `#f59e0b` | `#fbbf24` | `#b45309` on 16% amber ✅ / `#fbbf24` on 14% ✅ |
| Low | `#10b981` | `#34d399` | `#047857` on 14% mint ✅ / `#34d399` on 14% ✅ |

Badges are glass pills with a status dot. The Override stays the single danger-outline
button (`type="tertiary"` + the band-red). Chartreuse/tomato appear only inside the
retired v2 sheet (below).

### 0.6 Motion and micro-interaction (CSS only)

Everything is gated behind `@media (prefers-reduced-motion: reduce)`, which collapses to
no animation and 1ms transitions:

| Motion | Where | Spec |
|---|---|---|
| Rise + fade (entrance) | content elements, staggered ~45ms per element | `480ms cubic-bezier(.22,.61,.36,1)` |
| Card hover lift | glass cards, KPI cards | `translateY(-2px)` + deeper shadow, 200ms |
| Ambient orb drift | `body::before` | `90s ease-in-out infinite alternate` |
| Meter fill growth | employee-detail risk meter | 800ms ease-out from 0 width |
| Pulsing dot | KPI cards with a **high** accent | 2.2s soft pulse |
| Sidebar crop-out (minimize) | `[data-testid=stSidebar]` | `min-width/max-width/transform` 480ms eased; panel stays solid glass during the slide, content fades + slides `-20px`, background/border fade *after* the crop completes |
| Sidebar crop-in (maximize) | same element, `aria-expanded="true"` | panel reappears fully painted, content fades back in with a 60ms delay |

The sidebar minimize/maximize rides on Streamlit's own class-flip mechanism
(collapsed = `aria-expanded="false"` on the section, which retargets `min-width`,
`max-width` and `transform`). The built-in `300ms` transition is replaced with the
480ms drawer ease above; `overflow: hidden` while collapsed guarantees a clean crop
(no text paints past the shrinking edge). The **«** minimize control is forced always
visible (Streamlit hides it unless hovered), and while minimized a labeled
**» SIDEBAR** glass pill is the reopen control. Reduced-motion collapses all of this
to 1ms.

### 0.7 Interactivity (new in v3)

1. **Employees → detail jump**: rows are selectable (`selection_mode="single-row"`);
   selecting shows the employee chip and a primary *"Review this employee →"* button that
   pre-selects the employee on the detail page.
2. **Overview → Employees jump**: the department bar chart is click-selectable
   (`st.plotly_chart(on_select="rerun", selection_mode=["points"])`); a confirmation banner
   with *"Open … in Employees →"* jumps to the Employees page pre-filtered to that
   department.
3. Both cross-page moves run through callbacks (pre-rerun) writing the explicit
   `nav` widget key — the only pattern that cannot be modified-after-instantiation banned
   by Streamlit.

### 0.8 Phase-2 networking (documented, intentionally not built)

As decided in the design review, LAN/hosting is a *documented* future step, not code:
`streamlit run app/main.py --server.address 0.0.0.0` plus a firewall allow-rule for the
chosen port (and SmallNet/office LAN config). It is listed in `docs/report.md` §10.2.

---

## 0.5 Palette v2 — legacy (the dark "verdict" shell, retained for history)

A custom colour layer on Carbon's layout system, motivated by contrast maths:
chartreuse (even darkened) can never carry white text on a light canvas (~1.8:1), but
reads at ~13.7:1 on the dark card; tomato with white text fails AA (~2.9:1) but operates
as text/outline on dark (~5.8:1); lavender needs the dark end of its family for button
fills. Dark canvas also resolves the former dark-sidebar / white-content incoherence.

### 0.5.1 Core + shell tokens (`app/theme.py` → `TOKENS`)

| Role | Hex | vs surface `#1e1b23` | vs canvas `#141216` | Notes |
|---|---|---|---|---|
| Canvas `bg` | `#141216` | — | — | Charcoal, violet undertone |
| Card `surface` | `#1e1b23` | — | — | Layer 01 |
| Emphasis `surface_2` | `#282430` | — | — | Nested panels, hover fills |
| Border subtle / strong | `#332d40` / `#524a6b` | — | — | Hairlines / input & ghost borders |
| Text primary | `#f4f2fa` | **15.3:1** | 16.8:1 | Body |
| Text secondary | `#b8b3c8` | 8.3:1 | 9.2:1 | Notes, small-label text |
| Text helper | `#8a84a0` | 4.8:1 | 5.4:1 | Captions, tick labels (AA normal) |
| Sidebar bg | `#0e0c11` | — | — | Deeper than the canvas (shell) |

### 0.5.2 Accent trio + interactivity

| Role | Hex | Where it may appear | Contrast |
|---|---|---|---|
| Lavender fill (buttons, selected nav) | `#6d28d9` | Fill only | White text **7.1:1** ✅ |
| Lavender accent (links, chart lines, active-nav edge) | `#a78bfa` | Text/line on dark | **6.3:1** ✅ on card |
| Chartreuse (highlight) | `#c6f832` | Text/accents on dark **or fills with dark text only** | **13.7:1** on card ✅; dark-on-fill 15:1 ✅ |
| Tomato (danger, High risk, danger-outline button) | `#ff6347` | Text/outline on dark **or fills with dark text only** | **5.8:1** on card ✅; dark-on-fill 6.3:1 ✅ |
| Teal (success, Low risk) | `#3ddbd9` | Text/series on dark | **10:1** ✅ |

**Hard rules from the maths:** never white-on-chartreuse or white-on-tomato; chartreuse
High-value chart marks and badges keep the "Medium" slot; teal owns Low/success so
"Danger" stays red-family. The Override button is the single danger-outline
(`type="tertiary"` + tomato CSS); every other default button is a quiet ghost;
`type="primary"` carries the lavender fill.

### 0.5.3 Chart palettes

| Palette | Content |
|---|---|
| Categorical (14) | Seeded lavender → chartreuse → tomato → teal, then pink, indigo, amber, mint, fuchsia, blue, salmon, slate, peach, cyan |
| Sequential (9-box / confusion matrix) | Charcoal-violet → bright lavender (`#2a2536 … #cfc2fb`), light text except on the lightest step |
| Diverging (fairness gaps) | Lavender ⇄ teal around parity |
| Risk series (marks) | High `#ff6347` · Medium `#c6f832` · Low `#3ddbd9` |

### 0.5.4 Streamlit config (`​.streamlit/config.toml`, v2 — superseded by §0.1)

```toml
[theme]
base = "dark"
primaryColor = "#a78bfa"
backgroundColor = "#141216"
secondaryBackgroundColor = "#1e1b23"
textColor = "#f4f2fa"
font = "sans serif"
```

---

## 1. Colour tokens (legacy light sheet — superseded by §0, then §0.5)

Carbon separates **roles** (what a colour does) from **values** (the hex it resolves to).
We use the **White theme** for content and the **Gray 100 dark theme** for the navigation shell.

### 1.1 Core tokens — surfaces, borders, text

```css
:root{
  /* Backgrounds & layers */
  --bg:              #ffffff;   /* $background        White      */
  --surface:         #f4f4f4;   /* $layer-01          Gray 10    */
  --surface-2:       #ffffff;   /* $layer-02          White      */
  --surface-accent:  #e0e0e0;   /* $layer-accent-01   Gray 20    */
  --overlay:         rgba(0,0,0,0.60);  /* $overlay               */

  /* Borders */
  --border-subtle:   #e0e0e0;   /* $border-subtle-00  Gray 20    */
  --border-subtle-2: #c6c6c6;   /* $border-subtle-01  Gray 30    */
  --border-strong:   #8d8d8d;   /* $border-strong-01  Gray 50    */

  /* Text */
  --text-primary:    #161616;   /* $text-primary      Gray 100   */
  --text-secondary:  #525252;   /* $text-secondary    Gray 70    */
  --text-helper:     #6f6f6f;   /* $text-helper       Gray 60    */
  --text-placeholder:#a8a8a8;   /* $text-placeholder  Gray 40    */
  --text-inverse:    #ffffff;   /* $text-inverse      White      */
  --text-error:      #da1e28;   /* $text-error        Red 60     */
  --text-link:       #0f62fe;   /* $link-primary      Blue 60    */

  /* Skeleton / loading */
  --skeleton:        #c6c6c6;   /* $skeleton-element  Gray 30    */
  --skeleton-bg:     #e5e5e5;   /* $skeleton-background          */
}
```

### 1.2 Navigation shell — Gray 100 dark theme

```css
:root{
  --sidebar-bg:          #161616;  /* Gray 100 $background       */
  --sidebar-text:        #f4f4f4;  /* Gray 10                     */
  --sidebar-text-2:      #a8a8a8;  /* Gray 40                     */
  --sidebar-border:      #393939;  /* Gray 80                     */
  --sidebar-active-bg:   #393939;  /* Gray 80                     */
  --sidebar-active-text: #ffffff;  /* White                       */
  --sidebar-hover-bg:    #262626;  /* Gray 90                     */
}
```

### 1.3 Brand / interactive

```css
:root{
  --primary:          #0f62fe;   /* Blue 60  $interactive / $focus   */
  --primary-hover:    #0043ce;   /* Blue 70  $button-primary-hover   */
  --primary-active:   #0043ce;   /* Blue 70                          */
  --primary-focus:    #0f62fe;   /* Blue 60  $focus                  */
  --highlight:        #d0e2ff;   /* Blue 20  $highlight              */
  --focus-inset:      #ffffff;   /* White   $focus-inset            */
}
```

### 1.4 Status palette (Carbon **alert palette**)

Carbon's alert palette maps: red → danger, orange → serious warning, yellow → regular warning,
green → normal/success.

```css
:root{
  --support-error:   #da1e28;   /* Red 60    $support-error      */
  --support-success: #24a148;   /* Green 50  $support-success    */
  --support-warning: #f1c21b;   /* Yellow 30 $support-warning    */
  --support-info:    #0043ce;   /* Blue 70   $support-info       */
  --caution-major:   #ff832b;   /* Orange 40 $caution-major      */
}
```

**Application to attrition risk bands** (the one place status colour carries meaning):

| Band | Token | Hex | Notes |
|---|---|---|---|
| High risk | `--support-error` | `#da1e28` | Red 60 |
| Medium risk | `--caution-major` | `#ff832b` | Orange 40 |
| Low risk | `--support-success` | `#198038` | Green 60 |

> ⚠️ **Yellow 30 (`#f1c21b`) is a fill-only colour.** On white it fails WCAG AA for text
> (contrast ≈ 1.7:1). Never use it for type — use Orange 40 for anything text-bearing.

### 1.5 Categorical chart palette

Carbon's categorical sequence is **curated for maximum contrast between neighbouring colours**
and must be applied **in strict order**. Truncate from the left for fewer series.

```
 1  #6929c4   Purple 70
 2  #1192e8   Cyan 50
 3  #005d5d   Teal 70
 4  #9f1853   Magenta 70
 5  #fa4d56   Red 50
 6  #570408   Red 90
 7  #198038   Green 60
 8  #002d9c   Blue 80
 9  #ee538b   Magenta 50
10  #b28600   Yellow 50
11  #009d9a   Teal 50
12  #012749   Cyan 90
13  #8a3800   Orange 70
14  #a56eff   Purple 50
```

Single-hue overrides when the category count is known in advance:
`#6929c4` · `#002d9c` · `#1192e8` · `#005d5d`

### 1.6 Sequential palette — Blue ramp (9-box heatmap)

```
#edf5ff  #d0e2ff  #a6c8ff  #78a9ff  #4589ff
#0f62fe  #0043ce  #002d9c  #001d6c  #001141
```

### 1.7 Diverging palette — Purple ⇄ Teal (fairness gap, pre ⇄ post)

Use for values with a meaningful zero (a fairness gap of 0 = parity).

```
Purple side:  #1c0f30 #31135e #491d8b #6929c4 #8a3ffc
              #a56eff #be95ff #d4bbff #e8daff #f6f2ff
Teal side:    #d9fbfb #9ef0f0 #3ddbd9 #009d9a #007d79
              #005d5d #00413e #002d2c #001a19
```

> The Red ⇄ Cyan diverging palette is deliberately **not** used: red/green connotations would
> collide with the risk-band status colours.

---

## 2. Typography — IBM Plex

Carbon's typeface. Load from Google Fonts (SIL OFL 1.1).

```css
font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
font-family: 'IBM Plex Mono', 'Menlo', 'DejaVu Sans Mono', monospace;  /* code / metrics */
```

### Type scale

Carbon's scale is generated from a single equation with `y₀ = 12px`.

| Token | px | Used for |
|---|---|---|
| Plex 12 | 12 | micro-labels, axis ticks |
| Plex 14 | 14 | table body, helper text, button labels |
| Plex 16 | 16 | body copy (base) |
| Plex 18 | 18 | card titles, section headings |
| Plex 20 | 20 | page subtitles |
| Plex 24 | 24 | KPI values |
| Plex 28 | 28 | page titles |
| Plex 32 | 32 | hero / modal headings |

**Weights:** `300` Light · `400` Regular · `600` SemiBold.
SemiBold (600) is for section headers only — never for long-form text.

---

## 3. Spacing — Carbon 2x Grid

Mini unit = **8px**. Scale: `8 · 16 · 24 · 32 · 48 · 64 · 80`.

| Context | Value |
|---|---|
| Card padding | 16px (24px for KPI cards) |
| Grid gutter | 32px |
| Page content margin | 32px (16px < 672px) |
| Icon → label gap | 16px |
| Section separation | 48px |
| Button height | 40px (large) / 32px (sm) |

**Breakpoints**

| Name | px | Columns | Margin | Padding |
|---|---|---|---|---|
| Small | 320 | 4 | 0 | 16 |
| Medium | 672 | 8 | 16 | 16 |
| Large | 1056 | 16 | 16 | 16 |
| X-Large | 1312 | 16 | 16 | 16 |
| Max | 1584 | 16 | 24 | 16 |

**App content max-width:** 1440px, centred, 32px side padding.

---

## 4. Radius, elevation, borders

Carbon uses borders rather than shadows to separate surfaces.

```css
--radius-card:   8px;
--radius-input:  4px;
--radius-pill:   999px;
--border:        1px solid #e0e0e0;
--shadow-card:   0 1px 2px rgba(22,22,22,0.06);
--shadow-pop:    0 4px 12px rgba(22,22,22,0.12);
```

> **Documented deviation:** Carbon v11 ships square corners. We use an 8px card radius because
> modern SaaS dashboards read better with slight rounding, and the change is applied uniformly.

---

## 5. Component specs

### 5.1 Sidebar (dark shell)

| Element | Spec |
|---|---|
| Background | `#161616` |
| Nav item | 16px left padding, 40px height, 0 radius, hover `#262626` |
| Active item | left 3px `#0f62fe` rule + background `#393939` + text `#ffffff` |
| Nav item label | Plex 14, `#f4f4f4`; inactive `#a8a8a8` |
| Section label | Plex 12 uppercase, `#6f6f6f`, 24px top margin |
| Footer | Plex 12, `#6f6f6f` |

### 5.2 Cards

```
background: #ffffff
border:     1px solid #e0e0e0
radius:     8px
padding:    16px  (KPI cards: 24px)
shadow:     0 1px 2px rgba(22,22,22,0.06)
title:      Plex 14 #525252, then Plex 18 #161616 semibold
```

### 5.3 KPI card

```
label:    Plex 12 uppercase #6f6f6f, letter-spacing .04em
value:    Plex 24 #161616 (semibold for percentages)
delta:    Plex 12 — green #24a148 (down is good for attrition) / red #da1e28
icon:     16px, right-aligned, #0f62fe
padding:  24px
```

### 5.4 Buttons

| Type | Background | Text | Border |
|---|---|---|---|
| Primary | `#0f62fe` | `#ffffff` | none |
| Primary hover | `#0043ce` | `#ffffff` | none |
| Secondary | `#ffffff` | `#0f62fe` | 1px `#0f62fe` |
| Ghost | transparent | `#0f62fe` | none |
| Danger | `#da1e28` | `#ffffff` | none |

Radius 4px, height 40px, horizontal padding 16px, Plex 14.

### 5.5 Risk badge (pill)

```
padding: 4px 12px
radius:  999px
Plex 12 semibold
High:   bg #ffd7d9 (Red 20)   text #da1e28 (Red 60)
Medium: bg #ffdecf (Orange 20) text #8a3800 (Orange 70)
Low:    bg #a7f0ba (Green 20) text #198038 (Green 60)
```

### 5.6 Tables

- Header: background `#f4f4f4`, Plex 12 uppercase `#525252`, sticky
- Rows: 1px `#e0e0e0` bottom border only — **no vertical gridlines**
- Cell: Plex 14 `#161616`, 16px vertical padding
- Row hover: `#f4f4f4`

### 5.7 Chart rules (applies to every Plotly figure)

- Background **transparent** (card provides the surface)
- Gridlines `#e0e0e0`, 1px, **horizontal only**
- Axis labels `#6f6f6f` Plex 12; tick marks hidden
- No chart junk: no border, no legend box, legend horizontal above plot
- Bars: `marker_line_width = 0`, 2px gap between bars
- Lines: 2px width, no markers unless ≤ 20 points

---

## 6. The 11 Streamlit overrides

Streamlit's defaults are not Carbon-compliant. These are the required overrides, implemented in
`app/theme.py`.

| # | Fix | Selector / approach |
|---|---|---|
| 1 | Dark sidebar | `[data-testid="stSidebar"] { background-color:#161616 }` + inverted text rules |
| 2 | Remove top gradient bar | `[data-testid="stDecoration"] { display:none }` |
| 3 | Remove menu / footer / deploy chrome | `#MainMenu`, `footer`, `[data-testid="stStatusWidget"]` → `visibility:hidden` |
| 4 | Cards | `[data-testid="stVerticalBlockBorderWrapper"]` → white bg, 1px `#e0e0e0`, 8px radius, 16px padding |
| 5 | KPI cards | Custom HTML component, not `st.metric` |
| 6 | IBM Plex Sans | `@import url('https://fonts.googleapis.com/...IBM+Plex+Sans...')` |
| 7 | Buttons | `[data-testid="stBaseButton-primary"]` → `#0f62fe`, 4px radius, no shadow |
| 8 | Tabs | Active tab → 2px `#0f62fe` underline, transparent background |
| 9 | Data tables | `st.dataframe` → remove gridlines, sticky header, 14px font |
| 10 | Plotly template | Shared `carbon` template built from the tokens above |
| 11 | Content width | `.block-container { max-width:1440px; padding: 32px }` |

---

## 7. Accessibility requirements

Carbon's palette is designed against **WCAG 2.1**:

| Content | Minimum contrast |
|---|---|
| Body text (< 24px) | **4.5:1** |
| Large text (≥ 24px) | **3:1** |
| Graphical objects / chart elements | **3:1** |
| UI component boundaries | **3:1** |

Rules enforced in this project:

1. No body text is ever coloured with `--support-warning` (`#f1c21b`) — use Orange 70 `#8a3800`.
2. Chart series are never distinguished by hue alone — every chart also carries a label, pattern,
   or direct annotation.
3. Risk bands are encoded by **badge text as well as colour**, so the UI is readable in greyscale.
4. Focus states use a 2px `--primary` border and are never removed.
5. The categorical chart palette is applied in Carbon's curated order specifically to maximise
   perceptual distance between adjacent series.

---

## 8. Sources

| What | Source | URL |
|---|---|---|
| Colour tokens (exact hex) | Carbon — Color tokens | `https://carbondesignsystem.com/elements/color/tokens/` |
| Chart palettes | Carbon — Data-viz colour palettes | `https://carbondesignsystem.com/data-visualization/color-palettes/` |
| Colour principles, WCAG table | Carbon — Colour overview | `https://carbondesignsystem.com/elements/color/overview/` |
| Typeface, type scale | Carbon — Typography | `https://carbondesignsystem.com/elements/typography/overview/` |
| 8px grid, spacing, breakpoints | Carbon — 2x Grid | `https://carbondesignsystem.com/elements/2x-grid/overview/` |
| IBM Plex font | Google Fonts (SIL OFL 1.1) | `https://fonts.google.com/specimen/IBM+Plex+Sans` |

---

## 9. Theme switching

Not implemented in this version. If added later, the token map is already theme-scoped, so a
**Gray 100 / Gray 90** dark mode only requires swapping the `--bg` / `--text-*` / `--surface-*`
values plus the sidebar colours. The chart template must switch to the light-end of the
sequential ramp (Carbon: "in dark themes the lightest colour denotes the largest values").
