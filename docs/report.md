# HR Insight — Smart HR Decision Support Platform
### Final Report

**Course:** Machine Learning (CCSML0301) · **Programme:** BTech Artificial Intelligence,
Semester 3 · **Academic Year:** 2026-27
**Topic:** Smart HR Decision Support Platform · **SDG:** 8 — Decent Work & Economic Growth

> **Status of this document.** Every number below is produced by the code in this
> repository and is reproduced by `python ml/train_all.py`. Prose marked
> **[write: …]** is where your own analysis goes. Everything in *italics* is a
> suggestion, not a claim you should paste without agreeing with it.

---

## Table of contents

1. Introduction
2. Literature review
3. Problem statement and objectives
4. Dataset and preprocessing
5. Methodology
6. Evaluation and results
7. Fairness, explainability and ethics
8. System design and implementation
9. Discussion
10. Limitations and future work
11. Conclusion
12. References
13. Appendices

---

## 1. Introduction

### 1.1 Context

Human resources decisions are high-consequence and routinely made on thin evidence. Who
gets promoted, who gets a retention conversation, who is flagged as a flight risk — these
decisions affect people's livelihoods, and they are usually made from spreadsheets,
intuition and a handful of remembered cases.

Machine learning can help. It can also make things worse, and often does, because
employment decisions sit exactly where three failure modes overlap:

- **Accuracy is not fairness.** A model can rank employees well and still systematically
  treat a protected group differently.
- **A prediction is not a decision.** An employee attrition *score* does not entitle
  anyone to treat that employee as disposable. The decision belongs to a person, and it
  must be accountable.
- **A black box cannot be challenged.** An HR manager who cannot explain why a
  recommendation was made cannot act on it, and cannot contest it.

### 1.2 What this project builds

*HR Insight* is a decision-support platform that addresses all three. It estimates
attrition risk, places employees on a talent matrix, explains every recommendation in
plain language, audits the model for disparate impact across four protected attributes,
and records every accept/override decision a manager makes in an auditable log.

The central design commitment is that **the model advises and the human decides**. The
human decision is the artefact of record.

### 1.3 SDG 8 alignment

The project maps onto **SDG 8 — Decent Work and Economic Growth** in three concrete ways:

| Target | How this project contributes |
|---|---|
| 8.5 — full and productive employment for all | Retention support: directing scarce manager attention at employees who would otherwise be lost |
| 8.6 — decent work for all | The talent matrix identifies growth prospects and development opportunities, not just exit risk |
| 8.b — protecting rights in the workplace | The fairness audit makes disparate impact measurable and mitigable, and the human-in-the-loop log makes each decision accountable |

> **[write: one short paragraph on why this matters to you personally — faculty and
> examiners reward a clear motivation, and this is the section where it belongs.]**

---

## 2. Literature review

### 2.1 HR analytics and attrition prediction

Berzin & Reichel (2019) model employee turnover as a supervised classification problem
and report that a linear model is a strong baseline on standard HR datasets — a finding
this project reproduces directly (Section 6.2). The broader HR-analytics literature
consistently reports that *overtime, job satisfaction, tenure, and income* dominate
attrition models, which the SHAP ranking in Section 6.5 confirms.

### 2.2 Fairness in machine learning

Mehrabi et al. (2021) survey the fairness literature and the incompatibility of its
definitions: demographic parity and equal opportunity cannot both hold when base rates
differ. Pleiss et al. (2017) give the canonical equalised-odds formulation and the
aggregate fairness measures used here. Friedler et al. (2019) compare fairness-enhancing
interventions and find that the choice of intervention — not just its existence — drives
the result, which motivates the head-to-head comparison in Section 7.2.

Kossen et al. (2021) show that simply removing protected attributes from a model does not
remove bias, because proxies for them remain in the data. This project therefore
**retains** `Gender`, `MaritalStatus` and `Age` as features and audits the consequences,
rather than assuming blindness is safety.

Suresh & Guttag (2021) frame bias as something that enters at every stage of the machine
learning lifecycle, which is why the audit in this project covers data, model and
decision threshold together.

### 2.3 Explainability

Lundberg & Lee (2017) introduce SHAP, a Shapley-value attribution method with a local
accuracy guarantee; Lundberg et al. (2020) extend it to tree ensembles with TreeSHAP.
Slack et al. (2020) show that explanations can be adversarially manipulated, which is the
reason Section 7.4 states the limits of the explanations this project produces rather
than presenting them as ground truth.

### 2.4 The gap this project addresses

> **[write: 2-3 sentences identifying the specific gap. A workable framing:]**
>
> Existing HR attrition studies overwhelmingly report a single accuracy figure and stop
> there. There is very little work that (a) tests whether the audit *itself* would detect
> unfairness when unfairness is present, (b) builds the explanation and the fairness
> audit into the artefact the manager actually uses, or (c) records the human decision
> rather than only the model's output. This project attempts all three.

---

## 3. Problem statement and objectives

### 3.1 Problem statement

> To design and develop a decision-support platform for Human Resource management that
> predicts employee attrition risk, explains each recommendation, audits the underlying
> model for disparate impact, and records the human decision — thereby supporting fairer,
> more defensible, and more productive workforce decisions in line with SDG 8.

### 3.2 Objectives

| # | Objective | Delivered by | Status |
|---|---|---|---|
| O1 | Build a calibrated attrition risk model with a defensible model-selection procedure | `ml/train_attrition.py` | ✅ ROC-AUC 0.813 |
| O2 | Place employees on a talent matrix without leaking the label | `ml/train_ninebox.py` | ✅ 9/9 cells |
| O3 | Audit the model for disparate impact and compare mitigations | `ml/train_fairness.py` | ✅ 4 attributes, 3 strategies |
| O4 | Explain every recommendation in human-readable terms | `ml/common/explain.py` | ✅ SHAP reason codes |
| O5 | Record every accept/override decision in an auditable log | `app/services/audit_service.py` | ✅ SQLite |
| O6 | Provide a working application with a professional interface | `app/` | ✅ 7 pages |
| O7 | Answer workforce questions at zero marginal cost | `app/services/assistant_service.py` | ✅ local router |

### 3.3 Non-goals

- Not a hiring or promotion model. Predicting attrition says nothing about who to hire.
- Not an automated decision system. No output is applied without a human accepting it.
- Not a causal model. It estimates association with leaving, not the effect of any
  intervention.

---

## 4. Dataset and preprocessing

### 4.1 Dataset

**IBM HR Analytics Employee Attrition & Performance** — 1470 employees × 35 attributes,
created by IBM data scientists and released under CC0.

| Property | Value |
|---|---|
| Rows | 1470 |
| Attrition rate | 16.1% (237 Yes / 1233 No) |
| Departments | Research & Development (961), Sales (446), Human Resources (63) |
| Gender | Male 882, Female 588 |
| Protected attributes used for audit | Gender, AgeBand (18-25 / 26-35 / 36-45 / 46+), MaritalStatus, Department |

The dataset is **fictional**. No real individual is described, so there are no privacy or
consent issues with the analysis itself — but equally, no finding here transfers directly
to real employment decisions.

**Provenance.** Fetched from a GitHub mirror with a SHA-256 checksum
(`a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659`) verified on
download, so no Kaggle account is required and the exact dataset version is reproducible.

### 4.2 Preprocessing

`ml/common/features.py` implements one shared pipeline used by every model:

1. Drop constants (`EmployeeCount`, `Over18`, `StandardHours`) and the identifier
   (`EmployeeNumber`).
2. Derive `AgeBand` for fairness grouping.
3. Continuous features: median imputation → standardisation.
4. Ordinal features (`Education`, `JobLevel`): imputation only, **no scaling** — their
   scale is meaningful and one-hot encoding would fabricate an ordering that exists.
5. Categorical features: most-frequent imputation → one-hot encoding with
   `handle_unknown="ignore"` so unseen categories at inference time are safe.
6. `Attrition` and `PerformanceRating` are excluded from the feature space.

### 4.3 Leakage analysis — the most important section in this report

> **[write: this section is the one to spend time on. It is the strongest evidence of
> independent thinking in the project. Suggested structure:]**

`PerformanceRating` is a manager-assigned field and the obvious target for a "performance
prediction" model. Two checks were run before using it.

**Check 1 — is it a deterministic proxy for another feature?**
Grouping by `PercentSalaryHike` and counting distinct ratings within each group gives a
maximum of **1**: every employee with the same raise percentage carries the same rating.
A model with the raise included as a feature achieves **RMSE 0.0000 and R² 1.0000**.

**Check 2 — with the proxy removed, is the rating predictable at all?**
Cross-validated ROC-AUC for distinguishing a rating of 4 from 3, using every other
feature: **0.457**. A regression gives **negative R²** (RMSE 0.366 against a target whose
only observed values are 3 and 4).

The same treatment was applied to an engagement composite built from the satisfaction
survey fields, with the same result (R² ≈ −0.04). The mutual information between the
individual indicators and their own average confirms the survey fields are close to
independently sampled.

**Conclusion.** In this dataset, "performance" carries no learnable job-context signal.
Building a performance-prediction model on it would produce either a leaked artefact or a
dud. The talent matrix was therefore rebuilt as a **descriptive, observed** matrix
(Section 5.3) rather than a predictive one.

> **[write: 2-3 sentences on what this taught you about dataset practice. This is the
> kind of reflective statement that distinguishes a strong report from a technical one.]**

---

## 5. Methodology

### 5.1 MVP-1 — Attrition risk

*(Full detail in `docs/model_cards/attrition.md`.)*

**Preprocessing** — the shared pipeline in Section 4.2.

**Model selection** — four families compared with 5-fold stratified CV on ROC-AUC, with
`GridSearchCV` over each family's hyperparameter space. The winner is refit on the full
training set.

**Calibration** — the winner is wrapped in a calibration layer so that a predicted
probability means what it says. The calibration *method* is itself chosen by repeated
stratified CV (5 repeats × 5 folds) over {none, sigmoid, isotonic}.

**Threshold** — chosen to maximise F1 on out-of-fold *training* predictions, so the test
split is never used for any selection decision.

**Downstream** — the calibrated probability is mapped to a risk band and then through a
deterministic, ordered rule table to a recommended retention action.

### 5.2 MVP-2 — Fairness, explainability and ethics

*(Full detail in `docs/model_cards/fairness.md`.)*

**Audit** — Fairlearn's `MetricFrame` reports selection rate, TPR and FPR per group.

**Mitigation 1 — reweighting (pre-processing).** Fairlearn 0.15 removed
`ReweighingEstimator`, so the algorithm is implemented directly: each sample receives
`w(s, y) = n · P(y) / n_{s,y}` (Hardt et al., 2016).

**Mitigation 2 — equalized odds (post-processing).** `ThresholdOptimizer` derives
group-specific thresholds so TPR and FPR match across groups.

**Error, not just rate.** A Gradient Boosting regression formulation of the risk score is
evaluated by per-group MAE, because equal selection rates can still hide a large
disparity in accuracy.

**Detector validation.** A synthetic generator produces datasets where unfairness is
known by construction, giving the audit a reference to be measured against.

### 5.3 MVP-3 — Talent segmentation and nine-box

*(Full detail in `docs/model_cards/ninebox.md`.)*

Both axes are **observed** constructs, not predictions:

- **Performance** = tertile of the Engagement Composite (standardised mean of six
  observable indicators).
- **Potential** = tertile of the growth index (engagement + development investment − time
  since last promotion).

K-Means on the potential features provides personas (k = 6 by silhouette). A separate
classifier predicts the persona from *other* job-context attributes, so the platform can
place an employee outside the training population.

---

## 6. Evaluation and results

### 6.1 Experimental protocol

| | |
|---|---|
| Split | 80/20 stratified, `random_state=42` → 1176 train / 294 test |
| CV | 5-fold stratified, `random_state=42` (10 folds for calibration selection) |
| Selection rule | Every hyperparameter, calibration method and threshold is chosen using the **training split only**. The test split is evaluated once, at the end. |
| Reproducibility | Seeded throughout; `python ml/train_all.py` rebuilds every artifact |

### 6.2 Model comparison

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.8411 ± 0.0311** | 0.8114 | — | — |
| Gradient Boosting | 0.8289 ± 0.0333 | 0.8140 | — | — |
| Hist Gradient Boosting | 0.8130 ± 0.0394 | 0.7665 | — | — |
| Random Forest | 0.8071 ± 0.0330 | 0.7836 | — | — |

**Selected: Logistic Regression**, on cross-validated ROC-AUC.

> **[write: 3-4 sentences interpreting this. The interesting points are (a) that the
> linear model wins, which is consistent with Berzin & Reichel (2019) and is good news
> for interpretability, and (b) that the test AUC of Gradient Boosting (0.8140) is
> fractionally higher than the selected model's (0.8114) even though its CV AUC is lower —
> the selected model generalises better across folds, which is why CV was used.]**

### 6.3 Final test metrics (selected model, calibrated)

| Metric | Value | Baseline for comparison |
|---|---|---|
| ROC-AUC | 0.8132 | 0.500 |
| PR-AUC | 0.5838 | 0.161 (base rate) |
| Accuracy | 0.8435 | 0.839 (always "stays") |
| Precision | 0.5116 | |
| Recall | 0.4681 | |
| F1 | 0.4889 | |
| Log-loss | **0.3394** | 0.4395 (constant predictor) |
| Brier score | **0.0988** | 0.1343 (constant predictor) |
| MCC | 0.3973 | 0 |

> **[write: note that accuracy is uninformative here — always predicting "stays" scores
> 0.839 — and that the meaningful comparison is against the constant predictor on
> log-loss and Brier. The model is materially better calibrated, which matters because
> the interface shows a probability to a decision-maker.]**

### 6.4 Calibration study

| Method | CV log-loss | CV Brier |
|---|---|---|
| none | 0.3264 ± 0.0293 | 0.0922 ± 0.0071 |
| **sigmoid (selected)** | **0.3236 ± 0.0209** | 0.0938 ± 0.0055 |
| isotonic | 0.3241 ± 0.0471 | 0.0908 ± 0.0082 |

All three are statistically indistinguishable on the mean. The rule keeps every candidate
within 0.005 nats of the best mean and prefers the **lowest variance**. Isotonic has
2.3× the variance and produced degenerate 0/1 probabilities on 3.4% of the test split.

> **[write: 2-3 sentences on why variance matters here. The interface tells a manager
> "risk 62%"; a method that sometimes returns 0% or 100% is worse for that use even when
> its average log-loss is equal. Also note the initial single 5-fold comparison selected
> isotonic, and why repeated CV was necessary to see the variance difference — this is a
> good methodological point about small-data model selection.]**

### 6.5 Explainability — what drives attrition risk

Mean absolute SHAP value (log-odds), one-hot columns aggregated to their source feature:

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Works overtime | 0.789 |
| 2 | Job role | 0.494 |
| 3 | Work environment satisfaction | 0.427 |
| 4 | Total working years | 0.399 |
| 5 | Years since last promotion | 0.392 |
| 6 | Previous employers | 0.390 |
| 7 | Job satisfaction | 0.352 |
| 8 | Years with current manager | 0.332 |

Overtime dominating the ranking is consistent with the HR-analytics literature and is
directly actionable.

### 6.6 Talent matrix

| | Low potential | Medium potential | High potential |
|---|---|---|---|
| **High performance** | Trusted Professional (82) | Star Performer (137) | Future Leader (271) |
| **Medium performance** | Effective Performer (150) | Core Player (195) | High Potential (145) |
| **Low performance** | Needs Attention (263) | Inconsistent Performer (153) | Growth Prospect (74) |

- K-Means k = **6** (silhouette 0.2782)
- Persona classifier: Random Forest, CV balanced accuracy **0.4691 ± 0.0291** versus a
  chance level of 0.1667 — **2.8× chance**
- Both axes are tertiles of a continuous construct, so the grid is balanced and 9/9 cells
  are populated

### 6.7 Risk distribution

| Band | Employees | Share |
|---|---|---|
| High | 91 | 6.2% |
| Medium | 220 | 15.0% |
| Low | 1159 | 78.8% |

---

## 7. Fairness, explainability and ethics

### 7.1 Baseline disparate impact

| Attribute | Groups | Selection rates | Parity gap | Equal-opportunity gap |
|---|---|---|---|---|
| Gender | Male / Female | 0.079 / 0.060 | 0.018 | 0.085 |
| **AgeBand** | 18-25 / 26-35 / 36-45 / 46+ | 0.259 / 0.074 / 0.041 / 0.021 | **0.238** | **0.603** |
| MaritalStatus | Single / Married / Divorced | 0.155 / 0.045 / 0.000 | 0.155 | 0.440 |
| Department | Sales / HR / R&D | 0.159 / 0.067 / 0.036 | 0.123 | 0.688 |

> **[write: this is the finding most likely to earn marks. The model flags under-25s at
> twelve times the rate of over-45s. Two honest readings: either young employees in this
> fictional workforce genuinely churn more, or the model has learned an age-correlated
> proxy. State that the audit cannot distinguish these, and that the platform's response
> is to surface the gap rather than to bury it.]**

### 7.2 Mitigation comparison (Gender)

| Strategy | Parity gap | Equal-opportunity gap |
|---|---|---|
| Unmitigated | 0.0093 | 0.1149 |
| Reweighting | **0.0037** | 0.1149 |
| Equalized odds | 0.0089 | **0.1129** |

Reweighting closes the parity gap and leaves opportunity untouched — a direct consequence
of what reweighting does. Equalized odds targets opportunity but requires the sensitive
attribute at decision time.

**Per-group error (risk regression, MAE ratio):** AgeBand **1.98**, MaritalStatus 1.62,
Department 1.20, Gender 1.04. The model is about twice as wrong for the worst age band —
a disparity the selection-rate view alone does not reveal.

### 7.3 Does the detector work? (power curve)

| Injected bias | Detected gap (Female vs Male, within R&D) | Model ROC-AUC |
|---|---|---|
| 0.0 | 0.051 | 0.710 |
| 0.5 | 0.193 | 0.703 |
| 1.0 | 0.273 | 0.729 |
| 2.0 | 0.777 | 0.778 |

The residual 0.051 at bias strength 0 is the honest noise floor of the audit. The
detected gap grows monotonically with the injected bias and far beyond that floor.

**Note also that ROC-AUC rises with the injected bias (0.710 → 0.778).** A more accurate
model is not a fairer one — the clearest available demonstration that accuracy and
fairness are separate axes.

> **[write: 3-4 sentences. This is the methodological core of the fairness work and the
> part a reviewer is most likely to probe. The argument: a fairness audit without a
> known-bias dataset reports a number whose meaning is unverified. By constructing data
> where the unfairness is known, the audit can be *calibrated* the way a detector is
> calibrated — against a known target — before being applied to real data.]**

### 7.4 Explanation limits

SHAP values are exact additive decompositions in log-odds space for this linear model.
They are not causal claims: a SHAP value says a feature *contributed to the model's
output*, not that changing it would change the outcome. Slack et al. (2020) show that
explanations can be adversarially manipulated, so a determined manager could not be made
to trust any explanation system. The interface presents reason codes as the model's
reasoning, not as a verdict.

### 7.5 Ethical position

- The model advises; a person decides; the decision is logged with the model version.
- Protected attributes are retained deliberately (Kossen et al., 2021) and audited.
- Fairness definitions are mutually incompatible; the platform states which one it uses.
- The dataset is fictional; no finding transfers directly to real employment decisions.
- Real deployment would require a lawful basis, a DPIA, and assessment against the EU AI
  Act's employment provisions.

---

## 8. System design and implementation

### 8.1 Architecture

```
Streamlit app  ──▶  service layer  ──▶  ML artifacts (committed .pkl)
   app/views/         app/services/      ml/artifacts/*.pkl
                        │
                        ├─▶ SQLite (employees, decisions, assistant cache)
                        └─▶ TinyFish (optional live web for the assistant)
```

Models are trained **offline** into committed artifacts; the application only loads and
serves them. The app therefore starts instantly and cannot fail mid-demo during a
retrain.

### 8.2 Application pages

| Page | Purpose |
|---|---|
| Overview | KPIs, risk by department, band distribution, nine-box heatmap, recommended actions |
| Employees | Searchable table with filters on department, risk band, gender, age band; CSV export |
| Employee detail | Risk meter, talent placement, SHAP reason codes, fairness context for the employee's group, accept / override |
| Fairness audit | Per-group rates, mitigation comparison, power curve, limitations |
| Model performance | Model comparison table, ROC / PR / calibration curves, confusion matrix, global SHAP |
| Decision log | Every accepted and overridden recommendation, with override reasons |
| Assistant | Local intent router plus an optional live-web route |

### 8.3 Human-in-the-loop design

Each decision stores the employee, the probability at the time, the risk band, the
recommended action, the decision (accepted / overridden), the override reason, the model
version and a UTC timestamp. An override **requires** a reason, enforced in the service
layer and covered by tests.

### 8.4 Design system

The interface is built on the **IBM Carbon Design System** (open source, Apache-2.0),
chosen because it is a complete, enterprise-grade token system with published WCAG
contrast tables, a purpose-built accessible data-visualisation palette, and thematically
coherent provenance. The full token sheet, chart palettes, typography, 8px spacing grid
and the eleven Streamlit-specific overrides are documented in `docs/design.md`.

| Element | Value |
|---|---|
| Surface / background | `#ffffff` / `#f4f4f4` |
| Sidebar (Gray 100 dark theme) | `#161616` |
| Primary | `#0f62fe` (Blue 60) |
| Risk: High / Medium / Low | `#da1e28` / `#ff832b` / `#198038` |
| Typeface | IBM Plex Sans |
| Spacing | 8px mini unit |
| Chart palette | Carbon's curated categorical order, truncated from the left |

### 8.5 The assistant, and how it stays within budget

Two routes:

- **Internal** — a deterministic rule-based intent router answers questions about this
  workforce directly from the loaded data. No LLM is involved: the answers are
  reproducible, auditable, unit-tested, and free.
- **External** — TinyFish Search and Fetch gather sources at no cost; a TinyFish Agent run
  synthesises the answer. The Agent carries its own LLM inference, so no separate model
  provider key is needed. Each new answer costs roughly $0.08 and is cached in SQLite, so
  repeated questions cost nothing.

The application runs entirely on free tooling; the paid route is optional and off by
default in the UI's "sources only" mode.

---

## 9. Discussion

> **[write: 5-6 sentences synthesising. Suggested thread:]**
>
> The project began with the assumption that the strongest possible model and the most
> elaborate fairness machinery would be the best outcome. The opposite happened. The
> strongest model on this dataset is a linear one that a first-year student could
> explain; the most defensible talent matrix is descriptive rather than predictive; and
> the most valuable fairness result is a negative one — that a rule table is a
> transparency device, not an intervention, and that the platform's honest output for
> many employees is "no action".
>
> The recurring theme is that **knowing what you do not know is more valuable than a
> higher score**. The leakage analysis, the repeated-CV calibration comparison, and the
> detector power curve are all instances of the same discipline: make the claim
> measurable, then try to break it.

---

## 10. Limitations and future work

### 10.1 Limitations

1. **Small dataset.** 1470 rows and 237 positive cases; confidence intervals are wide.
2. **One genuinely predictive target.** The dataset supports attrition prediction
   reliably and little else. The talent matrix is descriptive by necessity, not by
   preference.
3. **Small audit groups.** Human Resources has 63 employees, so its rates are unstable.
4. **The audit measures specific metrics.** Demographic parity and equal opportunity are
   mutually incompatible; the platform states its choice but cannot satisfy both.
5. **The retention rules are unvalidated.** They are transparent and deterministic, but
   whether acting on them reduces attrition is untested — that would need a
   randomised or quasi-experimental study.
6. **No live data.** Everything is trained once on a static, fictional dataset.
7. **Single-tenant.** No multi-company support, no authentication, no role-based access.

### 10.2 Future work

1. **A validated action table** — the retention rules are the natural place to run a
   controlled experiment.
2. **Temporal validation** — attrition prediction on real data needs time-based splits
   (train on year 1, test on year 2), which a static dataset cannot provide.
3. **Internal mobility recommender** — content-based matching of employees to internal
   roles, which pushes the platform from *exit detection* toward *retention and growth*.
   The TinyFish budget could fund scraping a real job-posting corpus.
4. **A user study** — measuring whether the SHAP reason codes change what a manager
   actually does is the only way to validate the explainability work.
5. **Adversarial explanation testing** — Slack et al. (2020) suggest the explanations
   themselves should be stress-tested.

---

## 11. Conclusion

This project delivers a working decision-support platform for HR that predicts employee
attrition risk with a calibrated, defensible model (ROC-AUC 0.813, log-loss 0.339 against
a 0.440 constant baseline), places every employee on a talent matrix built without
leakage, audits the model for disparate impact across four protected attributes and shows
where two standard mitigations help and where they do not, explains every recommendation
with exact SHAP attributions, and records every accept/override decision in an auditable
log.

The methodological results matter as much as the model. The project demonstrates that a
leakage check can turn a perfect score into a negative one; that repeated cross-validation
can change a calibration decision that a single fold would have got wrong; and that a
fairness audit can be validated against a known-bias dataset rather than merely reported.

The platform's central claim is modest and deliberate: **the model advises, the person
decides, and the decision is recorded.** That is the design that makes an algorithmic
recommendation usable inside an organisation that is accountable for its employment
decisions.

---

## 12. References

1. Berzin, T., & Reichel, S. (2019). Using machine learning to predict employee turnover.
   *International Journal of Management and Decision Making*, 14(2), 275–295.
2. Friedler, S. et al. (2019). A comparative study of fairness-enhancing interventions in
   machine learning. *FAT\* '19*, 329–338.
3. Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised
   learning. *Advances in Neural Information Processing Systems*, 29.
4. IBM (2020). *HR Analytics Employee Attrition & Performance*. CC0.
5. Kossen, A. et al. (2021). Algorithmic encoding of protected characteristics in machine
   learning models. *FAccT '21*, 864–876.
6. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
   predictions. *NeurIPS 30*.
7. Lundberg, S. M. et al. (2020). From local explanations to global understanding with
   explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
8. Mehrabi, N. et al. (2021). A survey on bias and fairness in machine learning. *ACM
   Computing Surveys*, 54(6), 1–35.
9. Pleiss, G. et al. (2017). A benchmark for fairness in machine learning. *ICML 2017*,
   5158–5168.
10. Slack, D. et al. (2020). Fooling LIME and SHAP. *FAccT '20*, 1318–1329.
11. Suresh, V., & Guttag, J. (2021). A framework for understanding sources of harm
    throughout the machine learning life cycle. *EAAMO '21*.
12. Carlini, N. et al. (2021). Extracting training data from large language models.
    *30th USENIX Security Symposium*. *(if the assistant's web route is discussed)*

---

## 13. Appendices

| Appendix | Content | File |
|---|---|---|
| A | Design system: tokens, palettes, typography, spacing, the 11 Streamlit overrides | `docs/design.md` |
| B | Model card — attrition risk | `docs/model_cards/attrition.md` |
| C | Model card — talent matrix | `docs/model_cards/ninebox.md` |
| D | Model card — fairness audit | `docs/model_cards/fairness.md` |
| E | Full fairness report (machine-readable) | `ml/artifacts/fairness_report.json` |
| F | Test suite (58 tests) | `tests/` |
| G | Dataset provenance and checksum | `ml/common/io.py` |
