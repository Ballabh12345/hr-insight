# Model Card — Talent Segmentation & Nine-Box Matrix (MVP-4)

## At a glance

| | |
|---|---|
| **Artifacts** | `ninebox_kmeans.pkl`, `ninebox_persona_classifier.pkl`, `ninebox_config.pkl`, `ninebox_metrics.json`, `ninebox_pca.json` |
| **Tasks** | Unsupervised clustering + supervised multi-class classification |
| **Output** | Performance tier, potential tier, one of nine talent segments, a persona cluster |

## The finding that shaped this module

The classical nine-box grid is built on a manager's performance rating. In the IBM
dataset that target is unusable:

1. **`PerformanceRating` is a deterministic function of `PercentSalaryHike`.** Every
   employee with the same raise percentage carries the same rating. A model that includes
   the raise as a feature achieves RMSE **0.0000**, R² **1.0000** — a perfect score that
   measures nothing.
2. **With the proxy removed the rating is unpredictable.** Cross-validated ROC-AUC for
   distinguishing a rating of 4 from 3 is **0.457** — below chance. A regression on it
   gives **negative R²** (RMSE 0.366 against a target whose only values are 3 and 4).
3. **The same holds for an engagement composite.** Removing the satisfaction fields from
   the feature space and predicting their average gives R² ≈ **−0.04**.

Conclusion: in this dataset, "performance" carries no learnable job-context signal.
Building a performance-prediction model on it would produce either a leaked artefact or a
dud. This module therefore builds a **descriptive, observed** matrix, and says so.

## What the module builds

### Performance axis — observed

The **Engagement Composite**: the standardised mean of six observable indicators —
`JobInvolvement`, `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`,
`WorkLifeBalance`, `TrainingTimesLastYear`. Split into tertiles → Low / Medium / High.

It is observed, not predicted, so it needs no validation against an outcome.

### Potential axis — observed

The **growth index**: the standardised mean of `JobInvolvement` (+), `TrainingTimesLastYear`
(+) and `YearsSinceLastPromotion` (−, since a long time since promotion means less
growth). Split into tertiles → Low / Medium / High.

Tertiles of an employee-level construct are used rather than per-cluster tiers, because
per-cluster tiers inherit the very unequal cluster sizes (775 / 462 / 233) and leave the
3×3 grid lopsided.

### The nine-box

| | Low potential | Medium potential | High potential |
|---|---|---|---|
| **High performance** | Trusted Professional (82) | Star Performer (137) | Future Leader (271) |
| **Medium performance** | Effective Performer (150) | Core Player (195) | High Potential (145) |
| **Low performance** | Needs Attention (263) | Inconsistent Performer (153) | Growth Prospect (74) |

All **9 of 9 cells are populated**. The labels are defined by this project and documented
here so a reader can audit the mapping.

### Personas — unsupervised

K-Means on the potential features (`JobInvolvement`, `TrainingTimesLastYear`,
`YearsSinceLastPromotion`, `PromotionVelocity`, `YearsAtCompany`).

| k | Silhouette |
|---|---|
| 3 | 0.2491 |
| 4 | 0.2651 |
| 5 | 0.2701 |
| **6** | **0.2782** |
| 7 | 0.2377 |
| 8 | 0.2463 |

k = 6 is selected by silhouette. `PromotionVelocity` is a derived feature:
`YearsAtCompany / JobLevel` — years of service per level reached, where lower means faster.

### Persona classifier — supervised

K-Means assigns a persona only to the population it was trained on. A classifier is
therefore trained to predict the persona from attributes that were **not** used for
clustering, so it must infer the growth profile from surrounding job context. This is how
the platform would place an employee outside the training population.

| Model | CV balanced accuracy |
|---|---|
| **Random Forest** | **0.4691 ± 0.0291** |
| Logistic Regression | 0.4271 ± 0.0202 |
| Gradient Boosting | 0.4071 ± 0.0179 |

Chance for k = 6 is **0.1667**, so the classifier is **2.8× better than chance** at
inferring a persona from context. Test accuracy 0.4388, macro-F1 0.4294.

## Evaluation

There is no held-out ground truth for "potential", so the module is evaluated by:

| Criterion | Result |
|---|---|
| Cluster quality | silhouette 0.2782 at k = 6 |
| Axis balance | both axes within 5% of equal (490 / 490 / 490 for performance) |
| Grid coverage | 9 / 9 cells populated |
| Generalisation | persona classifier 2.8× chance on balanced accuracy |
| Leakage control | `PercentSalaryHike` excluded from the rating; regression test in `tests/test_features.py` |

## Ethical considerations

- The matrix is **descriptive**. It summarises observed engagement and growth indicators;
  it does not predict future performance or potential.
- The labels ("Needs Attention", "Growth Prospect") are a naming scheme defined by this
  project, not an established standard, and they can be read as judgements of people.
- Engagement indicators are partly self-reported and reflect the quality of the
  organisation as much as the individual. A low composite in a poorly managed team is not
  a statement about the employee.
- The `PerformanceRating` correlation with the engagement composite is **−0.041** in this
  dataset — effectively zero. Any story connecting "performance" and "engagement" here
  is a story about the dataset, not about people.

## Limitations

- The axes are constructs chosen by the author from available indicators; different
  reasonable choices would produce a different matrix.
- Clustering on 1470 rows with unequal cluster sizes is fragile. Silhouette of 0.278 is
  modest, and the k = 6 solution is not strongly separated from k = 4 or k = 5.
- `TrainingTimesLastYear` and `PercentSalaryHike` are partly consequences of management
  decisions, not of the employee.
- The persona classifier at 0.47 balanced accuracy is useful for triage, not for
  individual-level claims.

## Reproducing

```bash
python ml/train_ninebox.py       # ~20s
```

Also writes `data/processed/ninebox_assignments.csv`, which the dataset builder consumes
directly — so the labels the application serves are exactly the ones described here.
