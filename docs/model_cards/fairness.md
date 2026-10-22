# Model Card — Fairness Audit & Bias Mitigation (MVP-3)

## At a glance

| | |
|---|---|
| **Artifact** | `ml/artifacts/fairness_report.json` |
| **Tooling** | Fairlearn 0.15 (`MetricFrame`, `ThresholdOptimizer`) + a local reweighting implementation |
| **Sensitive attributes** | Gender, AgeBand, MaritalStatus, Department |
| **Output** | Per-group selection / TPR / FPR rates, parity and opportunity gaps, mitigation comparison, detector power curve |

## Why this module exists

A model that scores well can still treat groups differently. In employment decisions that
is not merely an accuracy question — it is an ethical and, in many jurisdictions, a legal
one. The platform therefore audits the attrition model rather than assuming it is fair.

This module is also the answer to the obvious criticism of a fairness audit: *"you
reported a number — how do you know the number means anything?"* Study C below answers
it.

## Study A — baseline audit

`MetricFrame` over the held-out test split (294 rows), reporting selection rate, true
positive rate and false positive rate per group.

| Attribute | Groups | Selection rates | Parity gap | Equal-opportunity gap |
|---|---|---|---|---|
| Gender | Male / Female | 0.079 / 0.060 | 0.018 | 0.085 |
| **AgeBand** | 18-25 / 26-35 / 36-45 / 46+ | 0.259 / 0.074 / 0.041 / 0.021 | **0.238** | **0.603** |
| MaritalStatus | Single / Married / Divorced | 0.155 / 0.045 / 0.000 | 0.155 | 0.440 |
| Department | Sales / HR / R&D | 0.159 / 0.067 / 0.036 | 0.123 | 0.688 |

**The headline is not gender — it is age.** The model flags employees under 25 at a rate
twelve times that of employees over 45. Whether that reflects real attrition dynamics in
this fictional workforce or an artefact of the model is exactly the kind of question a
fairness audit exists to raise.

The Human Resources department has only 63 employees in total, so its rate is unstable.
The audit reports group counts alongside every rate for this reason.

## Study B — mitigation comparison

Two standard interventions, both applied to the same Logistic Regression base model and
evaluated on the same test split.

| Strategy | Parity gap (Gender) | Equal-opportunity gap (Gender) |
|---|---|---|
| Unmitigated | 0.0093 | 0.1149 |
| **Reweighting** (pre-processing) | **0.0037** | 0.1149 |
| **Equalized odds** (post-processing) | 0.0089 | **0.1129** |

### Reweighting

`fairlearn.preprocessing.ReweighingEstimator` was removed in Fairlearn 0.15, so the
algorithm is implemented directly in `ml/common/fairness.py`: each sample receives
`w(s, y) = n · P(y) / n_{s,y}`, the standard scheme from Hardt et al. (2016). Under these
weights the sensitive attribute and the label are independent, so the estimator cannot
learn a group-dependent prior.

**It closes the parity gap and leaves the opportunity gap untouched.** That is not a bug —
it is the direct consequence of what reweighting does. It balances the label prior across
groups; it does not change the error rate at which each group's actual leavers are
caught. To move the opportunity gap you have to intervene on the decision rule.

### Equalized odds post-processing

`ThresholdOptimizer` derives group-specific thresholds from the training set so that
true-positive and false-positive rates match across groups.

**Its real limitation is operational:** it requires the sensitive attribute to be known at
decision time. For a retention recommendation that is often available (HR already holds
gender and age), but it is not universally so, and it would be a serious design flaw in a
setting where it is not.

### Group-level error, not just group-level rate

A model can have equal selection rates while still being much more accurate for one
group than another. A Gradient Boosting **regression** formulation of the risk score is
therefore also evaluated by per-group MAE.

| Attribute | Baseline MAE ratio | Reweighted MAE ratio |
|---|---|---|
| Gender | 1.041 | 1.006 |
| AgeBand | 1.984 | 1.932 |
| MaritalStatus | 1.621 | 1.659 |
| Department | 1.199 | 1.201 |

The AgeBand ratio of **1.98** means the risk score is about twice as wrong for the worst
age band as for the best — a discrepancy the selection-rate view alone would not reveal.

## Study C — does the detector actually work?

The IBM dataset contains no ground truth about unfairness, so a gap measured on it is a
number without a reference. `ml/generate_synthetic.py` therefore builds datasets where
the unfairness is **known by construction**.

Feature rows are bootstrapped from the real data (so realistic correlations survive), then
the attrition label is re-drawn from an explicit generative model:

```
logit(p) = b0 + b_overtime·OverTime + b_satisfaction·(JobSatisfaction ≤ 2)
         + b_distance·(DistanceFromHome > 15) + b_worklife·(WorkLifeBalance ≤ 2)
         + b_tenure·YearsAtCompany + b_income·z(MonthlyIncome)
         + BIAS_TERM

BIAS_TERM = bias_strength · 1[Gender = Female] · 1[Department = R&D]
```

Every term except `BIAS_TERM` is a legitimate driver of attrition. Sweeping
`bias_strength` from 0 to 2 produces a **power curve** for the detector:

| Injected bias | Female selection rate (R&D) | Male selection rate (R&D) | Detected gap | Model ROC-AUC |
|---|---|---|---|---|
| 0.0 | 0.124 | 0.073 | 0.051 | 0.710 |
| 0.5 | 0.278 | 0.085 | 0.193 | 0.703 |
| 1.0 | 0.366 | 0.094 | 0.273 | 0.729 |
| 2.0 | 0.859 | 0.083 | 0.777 | 0.778 |

The gap is measured *within* R&D so the injected effect is isolated from the different
base rates of the two genders.

**At bias strength 0 there is a residual gap of 0.051** — that is the honest baseline
noise of the audit, produced by real feature correlations rather than by injection. As the
injected bias grows the detected gap grows monotonically and far beyond that noise. The
detector is not reporting an arbitrary number; it is tracking the bias that is actually
there.

Note that model AUC also rises with the injected bias (0.710 → 0.778). A stronger
predictor is not a fairer one — accuracy and fairness are separate axes, and this is the
clearest possible demonstration of it.

## Ethical considerations

- **Fairness is not a property of the model alone.** It depends on the data, the decision
  rule, and the institutional context in which the output is used. This audit measures
  the first two on one fictional dataset.
- **Different fairness definitions are mutually incompatible** on this data. Demographic
  parity and equal opportunity cannot both be satisfied when the base rates differ. Any
  choice is a value judgement, and the platform should not hide which one it uses.
- **Mitigating a measured gap can itself be harmful.** Equalized-odds post-processing
  changes who gets flagged; if the original disparities reflected genuine differences in
  the workforce rather than bias, the "fix" introduces a new distortion.
- **The platform does not automate decisions.** Every recommendation is presented to a
  human who accepts or overrides it, and the decision is logged. The fairness audit
  exists to inform that human, not to be bypassed by an automated policy.
- The dataset is fictional. No finding here transfers directly to real employment
  decisions, and using such a model on real data would require a lawful basis, a DPIA,
  and assessment against the EU AI Act's employment provisions.

## Limitations

- Equalized-odds post-processing requires the sensitive attribute at decision time.
- Reweighting cannot remove bias encoded in the features themselves (Kossen et al., 2021);
  it only balances the label distribution across groups.
- Fairness is measured with respect to the chosen metrics on this dataset only.
- Groups with small counts produce unstable rates. Human Resources has 63 employees.
- The audit is a **snapshot**, not a continuous control. Re-running it after every
  retrain is necessary; `ml/train_fairness.py` is idempotent and designed to be run in
  the same pipeline as training.
- The synthetic experiment demonstrates that the *detector* works on the kind of bias it
  was built to detect. It does not establish that the detector would catch every form of
  bias present in a real dataset.

## Reproducing

```bash
python ml/generate_synthetic.py    # builds the four bias levels
python ml/train_fairness.py         # ~3s
```
