# HR Insight — Design System (`design.md`)

**Source of truth:** [IBM Carbon Design System](https://carbondesignsystem.com/) (open source, Apache-2.0).
Chosen because it is an enterprise-grade, fully-tokenised system with published WCAG contrast
tables, a purpose-built data-visualisation palette, and thematic coherence with the IBM HR
Analytics dataset used in this project.

**Applied to:** Streamlit 1.65.0 (single-page app), Plotly 7.1.0 charts, IBM Plex Sans type.

---

## 1. Colour tokens

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
