"""SHAP-based explainability helpers.

The app surfaces two things from this module:

* **Global importance** - which features move the attrition risk across the workforce.
* **Local reason codes** - for one employee, the top contributing features *with direction*,
  phrased so a non-technical HR manager can act on them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Human-readable labels for the raw column names. Keeps the UI readable for an HR audience.
FEATURE_LABELS: dict[str, str] = {
    "OverTime": "Works overtime",
    "JobSatisfaction": "Job satisfaction",
    "EnvironmentSatisfaction": "Work environment satisfaction",
    "JobInvolvement": "Job involvement",
    "WorkLifeBalance": "Work-life balance",
    "YearsAtCompany": "Years at the company",
    "YearsInCurrentRole": "Years in the current role",
    "YearsSinceLastPromotion": "Years since last promotion",
    "YearsWithCurrManager": "Years with current manager",
    "DistanceFromHome": "Distance from home",
    "MonthlyIncome": "Monthly income",
    "NumCompaniesWorked": "Previous employers",
    "TrainingTimesLastYear": "Training sessions last year",
    "PercentSalaryHike": "Last salary hike",
    "TotalWorkingYears": "Total working years",
    "RelationshipSatisfaction": "Relationship satisfaction",
    "Age": "Age",
    "JobLevel": "Job level",
    "Education": "Education level",
    "BusinessTravel": "Business travel frequency",
    "Department": "Department",
    "JobRole": "Job role",
    "Gender": "Gender",
    "MaritalStatus": "Marital status",
    "StockOptionLevel": "Stock option level",
    "DailyRate": "Daily rate",
    "HourlyRate": "Hourly rate",
    "MonthlyRate": "Monthly rate",
    "JobInvolvementScore": "Engagement composite",
    "PromotionVelocity": "Promotion velocity",
    "IncomeRelJobLevel": "Income relative to job level",
    "PromotionGap": "Promotion gap",
    "TenureRatio_CurrentRole": "Tenure in current role (ratio)",
    "TenureRatio_Manager": "Tenure with manager (ratio)",
    "TenureRatio_TotalWorking": "Company tenure share of career",
    "OverTimeXDistance": "Overtime x distance from home",
}


def label_for(feature: str) -> str:
    """Return the human-readable label for a raw feature name."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    # One-hot encoded categories come through as e.g. "Department_Research & Development".
    if "_" in feature:
        base, _, value = feature.partition("_")
        if base in FEATURE_LABELS:
            return f"{FEATURE_LABELS[base]} = {value}"
    return feature.replace("_", " ").strip()


class ShapExplainer:
    """SHAP explainer for tree ensembles and for the linear (logistic) model.

    The selected attrition model is a Logistic Regression, so ``shap.LinearExplainer``
    is used for it. For a logistic model the SHAP values are exact additive
    decompositions *in log-odds space* - that is, each value is the change in log-odds
    attributable to that feature. For a tree model the values are in the model's raw
    output space (log-odds for a classifier with a log loss).
    """

    def __init__(
        self,
        model,
        feature_names: list[str],
        is_linear: bool = False,
        background=None,
    ):
        import shap

        self._shap = shap
        self.model = model
        self.feature_names = feature_names
        self.is_linear = is_linear
        if is_linear:
            # SHAP caps its background sample at 100 rows by default and warns about it.
            # Pass a deterministic subsample instead so the run is quiet and repeatable.
            if background is not None and len(background) > 100:
                background = np.asarray(background)[:100]
            self._explainer = shap.LinearExplainer(model, background)
        else:
            self._explainer = shap.TreeExplainer(model)

    def _values(self, X):
        values = self._explainer.shap_values(X)
        values = np.asarray(values)
        # Newer SHAP may return (n_samples, n_features, n_outputs); take the positive class.
        if values.ndim == 3:
            values = values[:, :, 1]
        if values.ndim == 2:
            # Some explainers return (n_features, n_samples) instead. Transpose only when
            # the feature axis is unambiguously the first one.
            if values.shape[1] != len(self.feature_names) and values.shape[0] == len(self.feature_names):
                values = values.T
        return values

    def global_importance(self, X, top_n: int = 15, group_onehot: bool = True) -> list[dict[str, Any]]:
        """Mean absolute SHAP value per feature, descending.

        With ``group_onehot=True`` (the default) one-hot encoded columns are aggregated
        back to their source feature by summing them. For a linear model the SHAP values
        of all dummy columns of a feature sum to that feature's total contribution, so
        summing them gives the correct, interpretable importance - without it, "Overtime
        = No" and "Overtime = Yes" would each appear to be a separate driver.
        """
        values = self._values(X)
        if not group_onehot:
            mean_abs = np.abs(values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:top_n]
            return [
                {
                    "feature": self.feature_names[i],
                    "label": label_for(self.feature_names[i]),
                    "importance": float(mean_abs[i]),
                }
                for i in order
            ]

        groups: dict[str, list[int]] = {}
        for i, name in enumerate(self.feature_names):
            base = name.split("_", 1)[0]
            groups.setdefault(base, []).append(i)

        aggregated = {
            base: np.abs(values[:, idxs].sum(axis=1)).mean() for base, idxs in groups.items()
        }
        order = sorted(aggregated, key=aggregated.get, reverse=True)[:top_n]
        return [
            {"feature": base, "label": label_for(base), "importance": float(aggregated[base])}
            for base in order
        ]

    def local_reasons(
        self,
        X_row,
        row_values: dict[str, Any] | None = None,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """Top contributing features for a single prediction, with direction.

        One-hot columns are aggregated back to their source feature, so an employee is
        told "Works overtime = Yes increases attrition risk" rather than being shown both
        the "Yes" and "No" dummy columns.

        Parameters
        ----------
        X_row:
            The transformed feature row (one sample).
        row_values:
            Optional mapping of source feature name -> the employee's actual value, used
            to render a readable reason such as ``Works overtime = Yes``.
        top_n:
            How many reasons to return.

        Returns
        -------
        list of dict
            ``{"feature", "label", "impact", "direction", "text"}``.
        """
        values = np.asarray(self._values(X_row)).ravel()
        row_values = row_values or {}

        groups: dict[str, list[int]] = {}
        for i, name in enumerate(self.feature_names):
            base = name.split("_", 1)[0]
            groups.setdefault(base, []).append(i)

        aggregated = {base: float(values[idxs].sum()) for base, idxs in groups.items()}
        order = sorted(aggregated, key=lambda b: abs(aggregated[b]), reverse=True)[:top_n]

        reasons: list[dict[str, Any]] = []
        for base in order:
            impact = aggregated[base]
            direction = "increases" if impact > 0 else "reduces"
            label = label_for(base)
            detail = row_values.get(base)
            if detail is not None:
                label = f"{label} = {detail}"
            reasons.append(
                {
                    "feature": base,
                    "label": label,
                    "impact": impact,
                    "direction": direction,
                    "text": f"{label} {direction} attrition risk by {abs(impact):.3f}",
                }
            )
        return reasons
