# HR Insight — Compact Summary

**Smart HR Decision Support Platform** · Machine Learning (CCSML0301) · BTech AI, Sem 3
SDG 8 — Decent Work and Economic Growth · Individual assignment

> Compacted one-page companion to [`README.md`](README.md) and
> [`docs/report.md`](docs/report.md). Full detail lives in the model cards and
> `docs/design.md`.

---

## What it is

A decision-support platform for HR. It estimates attrition risk, places every employee
on a talent matrix, explains each recommendation in plain language, audits the model for
disparate impact across four protected attributes, and records every accept/override
decision a manager makes.

**Core commitment: the model advises, the human decides, the decision is recorded.**

## Run it

```bash
pip install -r requirements.txt
python ml/train_all.py      # rebuild every model + dataset (~61s, deterministic)
streamlit run app/main.py   # http://localhost:8501
```

No Kaggle account needed — the dataset comes from a checksum-verified GitHub mirror.

---

## Modules and results

| MVP | Module | Method | Headline result |
|---|---|---|---|
| 1 | Attrition risk | Logistic Regression + sigmoid calibration | ROC-AUC **0.813**, PR-AUC **0.584**, log-loss **0.339** (constant baseline 0.440) |
| 2 | Talent matrix | K-Means (k=6, silhouette 0.278) + tertile axes | **9/9** cells populated, balanced axes |
| 3 | Fairness audit | Fairlearn `MetricFrame`, reweighting, equalized-odds post-processing | AgeBand parity gap **0.238**, EO gap **0.603**, MAE ratio **1.98** |
| 4 | Explainability | SHAP `LinearExplainer`, one-hot aggregated | Overtime **0.79**, Job role **0.49**, Environment **0.43** |
| 5 | Human-in-the-loop | SQLite decision log | Overrides require a written reason |
| 6 | Assistant | Deterministic local router + TinyFish live web | Internal = free/instant; live ≈ $0.08, cached |

**7 pages:** Overview · Employees · Employee detail · Fairness audit · Model performance ·
Decision log · Assistant

---

## Three findings that carry the report

**1. The manager's performance rating is not a usable target.**
`PerformanceRating` is a *deterministic* function of `PercentSalaryHike` — every employee
with the same raise carries the same rating. Including the raise gives RMSE 0.0000 and
R² 1.0000, which measures nothing. Remove it and the rating becomes *unpredictable*
(CV ROC-AUC **0.457**, below chance). The talent matrix is therefore **descriptive, not
predictive** — built on an observed engagement composite and an observed growth index.
Pinned by a regression test.

**2. The interpretable baseline won.**
Logistic Regression (CV AUC **0.8411**) beat Gradient Boosting (0.8289), HistGradient
Boosting (0.8130) and Random Forest (0.8071). Good news for an HR product: the winner is
inherently interpretable and its SHAP values are exact rather than approximate.

**3. Calibration was chosen on variance, not the mean.**
Repeated CV (5×5) over {none, sigmoid, isotonic}: all three tie on mean log-loss
(~0.324). Isotonic has **2.3× the variance** and emits degenerate 0/1 probabilities.
Sigmoid wins the tie-break — the UI tells a manager "risk 62%", so stability matters more
than a 0.003-nat mean difference.

**Bonus — the fairness detector is validated, not just reported.**
A synthetic generator injects a *known* unfair component. The reported gap tracks it
monotonically (0.051 → 0.777) and model AUC *rises* with it (0.710 → 0.778): a more
accurate model is not a fairer one.

---

## Layout

```
hr-insight/
├── app/         main.py · theme.py (Carbon tokens + 11 Streamlit overrides)
│                views/ (7 pages) · services/ (data·model·fairness·audit·assistant)
├── ml/          common/ (features·metrics·fairness·explain·io)
│                generate_synthetic · train_attrition · train_ninebox
│                train_fairness · build_dataset · train_all
├── data/        raw · processed · db
├── docs/        design.md · report.md · fairness_report.md · model_cards/ · screenshots/
├── notebooks/   01_EDA.ipynb
├── tests/       58 tests
└── ops/         Dockerfile · scripts/
```

Models train **offline** into committed `.pkl` artifacts; the app only loads and serves
them, so it starts instantly and cannot crash mid-demo.

---

## Design system

**IBM Carbon Design System** (open source, Apache-2.0) — chosen for enterprise
maturity, published WCAG contrast tables, a purpose-built accessible data-viz palette,
and thematic coherence with the IBM dataset.

| Element | Value |
|---|---|
| Surface / layer | `#ffffff` / `#f4f4f4` |
| Sidebar (Gray 100) | `#161616` |
| Primary | `#0f62fe` (Blue 60) |
| Risk High / Med / Low | `#da1e28` / `#ff832b` / `#198038` |
| Type | IBM Plex Sans (300/400/600) |
| Spacing | 8px mini unit |
| Charts | Carbon's curated categorical order, truncated from the left |

Token sheet, chart palettes, type scale, spacing grid and the 11 Streamlit overrides are
in [`docs/design.md`](docs/design.md), with source URLs for every value.

---

## Budget

| Use | Tool | Cost |
|---|---|---|
| Dataset, literature | GitHub raw | $0 |
| Internal assistant answers | local router | $0 |
| Live market answers | TinyFish Agent (~5 steps) | ~$0.08 each, cached |
| **Spent so far** | | **~$0.08 of $16** |

---

## Verification

```bash
python -m pytest tests/ -v     # 58 passed
python ml/train_all.py         # deterministic, ~61s
```

The suite renders **every page** through Streamlit's `AppTest` harness, so a broken view
fails the build rather than the demo.

---

## Remaining before submission

1. **Update the topic-approval form** — tick **Individual Assignment**, clear section D,
   re-sign.
2. **Fill `docs/report.md`** — every number is real and reproducible; `[write: …]` marks
   where your own analysis goes. Prioritise §4.3 (leakage) and §7 (fairness).
3. **Set OneDrive `hpc` to "Always keep on this device"** — Files On-Demand can break
   git and pytest.

---

## Honest limitations

- 1470 rows, 237 positives — confidence intervals are wide.
- Only attrition is genuinely predictable in this dataset; the talent matrix is
  descriptive by necessity.
- Fairness is measured against chosen metrics on one fictional dataset; HR has 63
  employees, so its rates are unstable.
- The retention action rules are a transparent table, not a validated intervention.
- The assistant's live route depends on a third-party service and its wallet credit.
