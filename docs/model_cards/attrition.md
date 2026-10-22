# Model Card — Attrition Risk Prediction (MVP-1)

## At a glance

| | |
|---|---|
| **Artifact** | `ml/artifacts/attrition_model.pkl` + `attrition_predictor.pkl` |
| **Task** | Binary classification — will this employee leave the company? |
| **Selected model** | Logistic Regression (`C=1.0`, `class_weight=None`) + sigmoid calibration |
| **Train / test** | 1176 / 294, stratified, `random_state=42` |
| **Input** | 31 features (1 numeric, 1 ordinal, 7 categorical before encoding) |
| **Output** | Calibrated probability, a risk band, and a retention action |

## Intended use

To surface employees who may leave, so a manager can spend a scarce retention budget on
the right people and can see *why* the system flagged them.

**Out of scope:** making automated employment decisions. The platform produces a
recommendation that a human accepts or overrides, and every decision is logged.

## Data

- **Source:** IBM HR Analytics Employee Attrition & Performance (fictional, CC0)
- **Rows:** 1470 employees · **Attrition rate:** 16.1% (237 Yes / 1233 No)
- **Target:** `Attrition` ∈ {Yes, No}
- **Split:** 80/20 stratified, `random_state=42`, held untouched until the final evaluation

### Features used

All columns except:

| Excluded | Reason |
|---|---|
| `EmployeeCount`, `Over18`, `StandardHours` | constant |
| `EmployeeNumber` | row identifier |
| `PerformanceRating` | manager-assigned outcome — would leak the label |
| `AgeBand` | derived from `Age`; used for fairness grouping only |

**Protected attributes (`Gender`, `MaritalStatus`, `Age`) are deliberately retained.**
Following Kossen et al. (2021), removing protected attributes does not remove bias —
proxies for them remain in the data. The project's response is *retain, audit, mitigate*,
not *remove and hope*.

## Method

1. Median-impute and z-score continuous features; impute and one-hot encode categorical
   features (ordinal `Education` and `JobLevel` pass through unscaled — their scale is
   meaningful and one-hot would fabricate an ordering).
2. Compare four model families with 5-fold stratified CV on ROC-AUC.
3. Calibrate the winner so a probability means what it says.
4. Pick the decision threshold on out-of-fold training predictions.

### Model comparison

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Logistic Regression** | **0.8411 ± 0.0311** | 0.8114 |
| Gradient Boosting | 0.8289 ± 0.0333 | 0.8140 |
| Hist Gradient Boosting | 0.8130 ± 0.0394 | 0.7665 |
| Random Forest | 0.8071 ± 0.0330 | 0.7836 |

Logistic Regression was selected on cross-validated ROC-AUC. Its advantage over the
tree ensembles is consistent with the literature on HR analytics (Berzin & Reichel, 2019)
and is a genuine benefit for an employment-facing model: a linear model is inherently
interpretable and its SHAP values are exact.

### Calibration

Three options were compared with **repeated stratified CV (5 repeats × 5 folds)** on the
training set. A single 5-fold run proved too noisy to decide.

| Method | CV log-loss | CV Brier |
|---|---|---|
| none | 0.3264 ± 0.0293 | 0.0922 ± 0.0071 |
| **sigmoid** | **0.3236 ± 0.0209** | 0.0938 ± 0.0055 |
| isotonic | 0.3241 ± 0.0471 | 0.0908 ± 0.0082 |

The three are statistically indistinguishable on the mean. The selection rule keeps every
candidate within 0.005 nats of the best mean and then prefers the **lowest variance**:
isotonic has 2.3× the variance and produced degenerate 0/1 probabilities on 3.4% of the
test split. Sigmoid never does.

### Decision threshold

Tuned for maximum F1 on **out-of-fold training predictions** (0.332), so the test split
was not used to choose it.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.332 (tuned) | 0.512 | 0.468 | **0.489** |
| 0.50 (default) | 0.714 | 0.319 | 0.441 |

## Evaluation (held-out test set, 294 rows)

| Metric | Value |
|---|---|
| ROC-AUC | 0.8132 |
| PR-AUC | 0.5838 |
| Accuracy | 0.8435 |
| Precision | 0.5116 |
| Recall | 0.4681 |
| F1 | 0.4889 |
| Log-loss | 0.3394 |
| Brier score | 0.0988 |
| MCC | 0.3973 |

**Baselines.** A constant base-rate predictor scores log-loss **0.4395** and Brier
**0.1343**; the model is materially better on both. Accuracy of 0.84 is *not* meaningful
here — a model that always predicts "stays" also scores 0.839 — which is why PR-AUC and
the calibration metrics are the headline.

## Downstream use

- **Risk band:** Low < 0.25, Medium 0.25–0.55, High > 0.55.
- **Retention action:** a deterministic, ordered rule table (see
  `ml/build_dataset.py::retention_action`). Low → no action; Medium → 90-day check-in;
  High → onboarding check-in if new, compensation review if high performance, workload
  rebalance if engagement is low, retention bonus if long-tenured, otherwise a career
  conversation.
- **Explanation:** SHAP `LinearExplainer` on the fitted logistic model. Values are exact
  additive contributions **in log-odds space**. One-hot columns are aggregated back to
  their source feature, so a user sees "Works overtime = Yes", not a dummy column.

Top global drivers (mean |SHAP|, log-odds): Overtime **0.79**, Job role **0.49**,
Work environment satisfaction **0.43**, Total working years **0.40**,
Years since last promotion **0.39**.

## Ethical considerations

- The model flags *risk*, not *cause*. Retention actions are a starting point for a human
  conversation, not a verdict on an employee.
- Protected attributes are in the model. That is deliberate (Kossen et al., 2021) and is
  the reason the fairness audit exists — see the Fairness model card.
- The dataset is fictional, so no conclusion here transfers to real employment decisions.
- Using this model on real data would require a lawful basis for processing, a DPIA, and
  an assessment against the EU AI Act's employment provisions.

## Limitations

- 237 positive cases is small; the confidence intervals on the metrics are wide.
- The rating/leakage analysis in the talent matrix card shows this dataset's "performance"
  fields are largely uninformative. The same caution applies to any composite.
- Selection rate and calibration are measured on the full population, not on the
  subgroup-conditional base rate. A group can be under-predicted and still show a low
  parity gap.
- The retention action rules have not been validated as an intervention. Whether acting
  on them reduces attrition is untested.
- Nothing here is retrained on new data. `python ml/train_all.py` reproduces the build.

## Reproducing

```bash
python ml/train_attrition.py     # ~22s
```

Seeded throughout (`random_state=42`); the output is reproducible byte-for-byte given the
same package versions in `requirements.txt`.
