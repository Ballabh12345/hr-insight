# HR Insight — Smart HR Decision Support Platform

**Machine Learning (CCSML0301) · BTech AI · Semester 3**
**SDG 8 — Decent Work and Economic Growth**

A decision-support platform for HR: it estimates which employees are likely to leave,
places every employee on a talent matrix, explains *why* each recommendation was made,
audits the model for disparate impact, and records every accept/override decision a
manager makes.

---

## Quick start

```bash
pip install -r requirements.txt
python ml/train_all.py          # builds every model + dataset (~60s)
streamlit run app/main.py       # opens the application
```

No Kaggle account is needed — the dataset is fetched from a checksum-verified GitHub mirror.

---

## What it does

| MVP | Module | Method | Result |
|---|---|---|---|
| 1 | **Attrition risk** | Logistic Regression + sigmoid calibration | ROC-AUC **0.813**, PR-AUC **0.584**, log-loss **0.339** (vs 0.440 constant) |
| 2 | **Talent matrix** | K-Means (k=6, silhouette 0.278) + engagement/growth tertiles | 9/9 cells populated |
| 3 | **Fairness audit** | Fairlearn `MetricFrame`, reweighting, equalized-odds post-processing | Age-band parity gap **0.238**, EO gap **0.603** |
| 4 | **Explainability** | SHAP (`LinearExplainer`) | Overtime **0.79**, Job role **0.49**, Environment **0.43** |
| 5 | **Human-in-the-loop** | SQLite decision log | Every accept/override recorded with model version |
| 6 | **Assistant** | Deterministic local router + TinyFish live web | Free for internal questions, ~$0.08 for live answers |

**Seven application pages:** Overview · Employees · Employee detail · Fairness audit ·
Model performance · Decision log · Assistant.

---

## Three findings worth reading the report for

### 1. The manager's performance rating is not a usable target

`PerformanceRating` in the IBM dataset is a **deterministic function of
`PercentSalaryHike`** — every employee with the same raise percentage carries the same
rating. A model that includes the raise as a feature scores RMSE **0.0000** and R²
**1.0000**, which measures nothing.

Remove the proxy and the rating becomes **unpredictable**: cross-validated ROC-AUC of
**0.457** for distinguishing a rating of 4 from 3 — below chance. The same holds for an
engagement composite built from the satisfaction survey fields (R² negative).

So the talent matrix does **not** predict a performance rating. Its performance axis is an
*observed* engagement composite (standardised mean of six indicators), and its potential
axis is the tertile of an employee-level growth index. Both are honest, balanced, and
described in full in the model card.

> Regression test: `tests/test_features.py::test_percent_salary_hike_is_a_deterministic_proxy_for_the_rating`

### 2. The interpretable baseline won

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Logistic Regression** | **0.8411 ± 0.0311** | 0.8114 |
| Gradient Boosting | 0.8289 ± 0.0333 | 0.8140 |
| Hist Gradient Boosting | 0.8130 ± 0.0394 | 0.7665 |
| Random Forest | 0.8071 ± 0.0330 | 0.7836 |

On this dataset a regularised linear model beats every tree ensemble. That is a good
outcome for an HR product: the winner is inherently interpretable, and its SHAP values
are exact additive decompositions rather than approximations.

### 3. Calibration method was chosen on variance, not just the mean

| Method | CV log-loss | CV Brier |
|---|---|---|
| none | 0.3264 ± 0.0293 | 0.0922 ± 0.0071 |
| **sigmoid** | **0.3236 ± 0.0209** | 0.0938 ± 0.0055 |
| isotonic | 0.3241 ± 0.0471 | 0.0908 ± 0.0082 |

All three are statistically indistinguishable on the mean. Isotonic has **2.3× the
variance** and emits degenerate 0/1 probabilities on small samples. Sigmoid wins the
tie-break on stability — which matters when the UI tells a manager "risk 62%".

---

## Fairness

Selection-rate gaps in the unmitigated model (demographic parity difference):

| Attribute | Parity gap | Equal-opportunity gap |
|---|---|---|
| Gender | 0.018 | 0.085 |
| **AgeBand** | **0.238** | **0.603** |
| MaritalStatus | 0.155 | 0.440 |
| Department | 0.123 | 0.688 |

**Mitigation on Gender** — reweighting closes the parity gap (0.0093 → 0.0037) but leaves
the equal-opportunity gap untouched (0.1149 → 0.1149). Equalized-odds post-processing
addresses opportunity directly (0.1149 → 0.1129) but requires the sensitive attribute at
decision time.

**Does the detector actually work?** The IBM dataset has no ground truth about unfairness,
so `ml/generate_synthetic.py` bootstraps its feature rows and re-draws the label from an
explicit generative model with a *known* unfair component. The gap the audit reports grows
monotonically with the injected bias:

| Injected bias | Detected gap (Female vs Male, within R&D) |
|---|---|
| 0.0 | 0.051 |
| 0.5 | 0.193 |
| 1.0 | 0.273 |
| 2.0 | 0.777 |

---

## Architecture

```
Streamlit app  ──▶  service layer  ──▶  ML artifacts (committed .pkl)
   app/views/         app/services/      ml/artifacts/*.pkl
                        │
                        ├─▶ SQLite (employees, decisions, assistant cache)
                        └─▶ TinyFish (optional live web for the assistant)
```

Models are trained **offline** into committed artifacts. The application only loads and
serves them, so it starts instantly and cannot crash mid-demo on a retrain.

### Repository layout

```
hr-insight/
├── app/
│   ├── main.py                  # entry point, sidebar navigation
│   ├── theme.py                 # Carbon tokens + 11 Streamlit overrides + Plotly template
│   ├── views/                   # 7 pages
│   └── services/                # data · model · fairness · audit · assistant
├── ml/
│   ├── common/                  # features · metrics · fairness · explain · io
│   ├── generate_synthetic.py    # controlled-bias workforce generator
│   ├── train_attrition.py       # MVP-1
│   ├── train_ninebox.py         # MVP-2
│   ├── train_fairness.py        # MVP-3
│   ├── build_dataset.py         # assembles what the app serves
│   └── train_all.py             # runs everything in order
├── data/  raw · processed · db
├── docs/   design.md · report.md · model_cards/ · fairness_report.md
├── tests/  58 tests
└── ops/    Dockerfile · scripts/
```

---

## Design system

The UI is built on the **IBM Carbon Design System** (open source, Apache-2.0) — full token
sheet, chart palettes, typography, spacing and the 11 Streamlit-specific overrides are in
[`docs/design.md`](docs/design.md). Sources: [tokens](https://carbondesignsystem.com/elements/color/tokens/),
[data-viz palettes](https://carbondesignsystem.com/data-visualization/color-palettes/),
[typography](https://carbondesignsystem.com/elements/typography/overview/),
[2x grid](https://carbondesignsystem.com/elements/2x-grid/overview/).

---

## Budget

The application runs entirely on free tooling. The only paid component is the optional
live-web route in the assistant.

| Use | Tool | Cost |
|---|---|---|
| Literature review, dataset | GitHub raw | $0 |
| Internal assistant questions | local router | $0 |
| Live market questions | TinyFish Agent, ~5 steps | ~$0.08 each, cached |
| Planned spend | | **~$5.60 of $16** |

---

## Tests

```bash
python -m pytest tests/ -v      # 58 tests
```

The suite renders every application page through Streamlit's `AppTest` harness, so a
broken view fails the build rather than the demo. It also contains regression tests for
the leakage finding above.

---

## Data

**IBM HR Analytics Employee Attrition & Performance** — 1470 employees × 35 attributes,
released by IBM under CC0. The dataset is **fictional** (generated by IBM data
scientists), so it carries no real personal data; conclusions drawn from it are method
demonstrations, not findings about real employment decisions.

---

## Limitations

- The dataset is small (1470 rows, 237 attrition cases). Confidence intervals are wide.
- Only one genuinely predictive target exists in it; the talent matrix is therefore
  descriptive rather than predictive by design.
- Fairness is measured against the chosen metrics on this dataset only. Small groups
  (Human Resources has 63 employees) give unstable rates.
- The retention action rules are a transparent, deterministic table — not a validated
  intervention. Whether acting on them reduces attrition is untested.
- The assistant's live route depends on a third-party service and its wallet credit.

---

## License

Code: MIT. Dataset: CC0 (IBM). Design tokens: IBM Carbon, Apache-2.0.
