"""Fairness auditing and bias mitigation.

Two mitigations are implemented and compared:

1. **Reweighting** (pre-processing) - training samples are re-weighted so the sensitive
   attribute becomes independent of the label, then the base estimator is refit.

   *Implementation note:* Fairlearn 0.15 removed ``fairlearn.preprocessing.ReweighingEstimator``,
   so the algorithm is implemented directly here. It is the standard reweighting scheme from
   Hardt et al. (2016), *Equality of Opportunity in Supervised Learning*, which assigns

   .. code-block:: text

       w(s, y) = n * P(y) / n_{s, y}

   to every sample with sensitive group ``s`` and label ``y``. Under these weights the
   sensitive attribute and the label are independent, so the estimator cannot learn a
   group-dependent prior.

2. **Equalized-odds post-processing** (``fairlearn.postprocessing.ThresholdOptimizer``) -
   a *post-processing* intervention. Group-specific decision thresholds are derived from the
   training set so that the true-positive rate and false-positive rate match across groups.

Both are evaluated against the unmitigated baseline, and both are reported with their
real limitations - reweighting can only partially remove bias when it is baked into the
features, and threshold post-processing requires the sensitive attribute to be known at
decision time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

#: Attributes the audit is reported over.
FAIRNESS_GROUPS: list[str] = ["Gender", "AgeBand", "MaritalStatus", "Department"]

#: Which protected attribute the mitigations operate on.
#: Gender is chosen as the primary axis because it is the attribute most commonly
#: challenged in employment decisions, and it has adequate group sizes in this dataset.
PRIMARY_SENSITIVE = "Gender"


def _metric_frame(y_true, y_pred, sensitive: pd.Series) -> MetricFrame:
    return MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
            "count": lambda y_true, y_pred: len(y_true),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )


def audit(
    y_true,
    y_pred,
    sensitive_features: dict[str, pd.Series],
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """Audit predictions for disparate impact across each protected group.

    Parameters
    ----------
    y_true, y_pred:
        Ground truth and model predictions (binary).
    sensitive_features:
        Mapping of attribute name -> pandas Series aligned with ``y_true``.
    groups:
        Which attributes to report on. Defaults to :data:`FAIRNESS_GROUPS`.

    Returns
    -------
    dict
        ``{"groups": {attribute: {...}}, "worst_gap": {...}}``
    """
    groups = groups or FAIRNESS_GROUPS
    report: dict[str, Any] = {"groups": {}, "worst_gap": {}}

    for attr in groups:
        if attr not in sensitive_features:
            continue
        sf = sensitive_features[attr]
        if sf.nunique() < 2:
            continue

        mf = _metric_frame(y_true, y_pred, sf)
        by_group = mf.by_group

        dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sf))
        dp_ratio = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sf))
        eo_diff = float(equal_opportunity_difference(y_true, y_pred, sensitive_features=sf))
        eo_ratio_gap = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sf))

        report["groups"][attr] = {
            "selection_rate": {str(k): float(v) for k, v in by_group["selection_rate"].items()},
            "true_positive_rate": {str(k): float(v) for k, v in by_group["true_positive_rate"].items()},
            "false_positive_rate": {str(k): float(v) for k, v in by_group["false_positive_rate"].items()},
            "count": {str(k): int(v) for k, v in by_group["count"].items()},
            "summary": {
                "demographic_parity_difference": dp_diff,
                "demographic_parity_ratio": dp_ratio,
                "equal_opportunity_difference": eo_diff,
                "equalized_odds_difference": eo_ratio_gap,
            },
        }
        report["worst_gap"][attr] = max(dp_diff, eo_diff)

    return report


def regression_group_mae(
    y_true,
    y_pred,
    sensitive_features: dict[str, pd.Series],
    groups: list[str] | None = None,
) -> dict[str, Any]:
    """Per-group MAE and MAE ratio for the nine-box regression model."""
    groups = groups or FAIRNESS_GROUPS
    out: dict[str, Any] = {}
    for attr in groups:
        if attr not in sensitive_features:
            continue
        sf = sensitive_features[attr]
        if sf.nunique() < 2:
            continue
        per_group = {}
        for value in sf.unique():
            mask = sf == value
            per_group[str(value)] = float(np.mean(np.abs(np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])))
        values = list(per_group.values())
        out[attr] = {
            "mae_by_group": per_group,
            "mae_ratio": float(max(values) / min(values)) if min(values) > 0 else float("inf"),
            "mean_absolute_error": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
        }
    return out


def _clean_sf(series: pd.Series) -> pd.Series:
    """Coerce a sensitive-feature Series to plain strings (fairlearn needs hashable keys)."""
    return series.astype(str).reset_index(drop=True)


def compute_reweighting_weights(y_true, sensitive: pd.Series) -> np.ndarray:
    """Reweighting weights after Hardt et al. (2016).

    Each sample receives ``w(s, y) = n * P(y) / n_{s,y}``, which makes the sensitive
    attribute and the label statistically independent in the reweighted sample.
    """
    y = np.asarray(y_true)
    s = np.asarray(sensitive).astype(str)
    n = len(y)

    weights = np.ones(n, dtype=float)
    for group in np.unique(s):
        in_group = s == group
        n_s = in_group.sum()
        if n_s == 0:
            continue
        for label in np.unique(y):
            cell = in_group & (y == label)
            n_cell = cell.sum()
            if n_cell == 0:
                continue
            p_y = (y == label).sum() / n
            weights[cell] = (n * p_y) / n_cell
    return weights


class Reweigher:
    """Fit an estimator with reweighting weights derived from a sensitive attribute.

    Handles both bare estimators (``sample_weight``) and scikit-learn Pipelines
    (``model__sample_weight``).
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.weights_: np.ndarray | None = None

    def _weight_params(self, weights: np.ndarray) -> dict:
        if hasattr(self.estimator, "named_steps") and "model" in self.estimator.named_steps:
            return {"model__sample_weight": weights}
        return {"sample_weight": weights}

    def fit(self, X, y, sensitive_features: pd.Series):
        self.weights_ = compute_reweighting_weights(y, sensitive_features)
        self.estimator.fit(X, y, **self._weight_params(self.weights_))
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def fit_reweighing(estimator, X_train, y_train, sensitive: pd.Series) -> Reweigher:
    """Fit a reweighted estimator. Returns the fitted :class:`Reweigher`."""
    rw = Reweigher(estimator)
    rw.fit(X_train, y_train, _clean_sf(sensitive))
    return rw


def fit_threshold_optimizer(
    estimator,
    X_train,
    y_train,
    sensitive: pd.Series,
    constraints: str = "equalized_odds",
    objective: str = "accuracy_score",
):
    """Fit a group-threshold post-processor. Returns the fitted ``ThresholdOptimizer``.

    Note that Fairlearn 0.15's ``ThresholdOptimizer`` takes no ``random_state``
    constructor argument. The randomised interpolation happens in ``predict``, which does
    accept one - pass it there or the mitigated results are not reproducible.
    """
    post = ThresholdOptimizer(
        estimator=estimator,
        constraints=constraints,
        objective=objective,
        predict_method="predict_proba",
        prefit=False,
    )
    post.fit(X_train, y_train, sensitive_features=_clean_sf(sensitive))
    return post
