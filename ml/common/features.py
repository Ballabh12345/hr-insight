"""Feature engineering for the HR Insight models.

Design notes
------------
* The IBM dataset contains three constant columns and one identifier column; all four are
  dropped before modelling.
* ``PerformanceRating`` is a **manager-assigned** field. It is used as a regression *target*
  for the nine-box model and is never a feature, so the model does not leak the label.
* Ordinal fields (Education, JobLevel) are kept numeric - one-hot encoding them would
  fabricate an ordering that the data actually has.
* ``AgeBand`` is derived, not learned from the label, and is used only for fairness
  grouping, never as a model feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_ATTRITION = "Attrition"
TARGET_PERFORMANCE = "PerformanceRating"

#: Columns with a single constant value - carry no signal.
DROP_CONSTANTS: list[str] = ["EmployeeCount", "Over18", "StandardHours"]

#: Engineered features added by :func:`engineer_features`. Kept explicit so the
#: ablation study in ``ml/train_attrition.py`` can drop them as a block.
ENGINEERED_FEATURES: list[str] = [
    "PromotionVelocity",
    "IncomeRelJobLevel",
    "PromotionGap",
    "TenureRatio_CurrentRole",
    "TenureRatio_Manager",
    "TenureRatio_TotalWorking",
    "OverTimeXDistance",
]

#: Row identifier - must never enter a model.
DROP_ID: list[str] = ["EmployeeNumber"]

#: Explicit ordinal encodings (already 1..n in the source data).
ORDINAL_FEATURES: list[str] = ["Education", "JobLevel"]

#: Fields deliberately excluded from the attrition model, with the reason.
#: Keeping this list explicit is what makes the model card auditable.
#:
#: NOTE on protected attributes: ``Gender``, ``MaritalStatus`` and ``Age`` are **retained**
#: as model features. This follows Kossen et al. (2021), who showed that simply removing
#: protected attributes does not remove bias, because proxies for them remain in the data.
#: The project's response is *retain + audit + mitigate*, not *remove and hope*.
#: ``AgeBand`` is excluded because it is a deterministic function of ``Age`` and would
#: double-count age; it is used for fairness grouping only.
EXCLUDED_FROM_FEATURES: dict[str, str] = {
    "PerformanceRating": "manager-assigned outcome; leaking it makes attrition trivially predictable",
    "Attrition": "the label",
    "AgeBand": "deterministic function of Age; used for fairness grouping only",
}

AGE_BAND_EDGES = [18, 26, 36, 46, 100]
AGE_BAND_LABELS = ["18-25", "26-35", "36-45", "46+"]


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add a coarse ``AgeBand`` column used for fairness grouping."""
    out = df.copy()
    out["AgeBand"] = pd.cut(
        out["Age"], bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS, right=False
    )
    out["AgeBand"] = out["AgeBand"].astype(str)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived, leakage-free features used by the attrition and talent models.

    All features are functions of fields that are legitimately observable at decision
    time; none uses the attrition label or the manager-assigned performance rating.
    """
    out = df.copy()

    # Promotion velocity: years of service per job level (lower == faster promotion).
    if "YearsAtCompany" in out.columns and "JobLevel" in out.columns:
        out["PromotionVelocity"] = out["YearsAtCompany"] / out["JobLevel"].clip(lower=1)

    # Income relative to the median for the employee's job level.
    if "MonthlyIncome" in out.columns and "JobLevel" in out.columns:
        level_median = out.groupby("JobLevel")["MonthlyIncome"].transform("median")
        out["IncomeRelJobLevel"] = out["MonthlyIncome"] / level_median.replace(0, np.nan)
        out["IncomeRelJobLevel"] = out["IncomeRelJobLevel"].fillna(1.0)

    # Promotion gap: how long since the last promotion relative to tenure.
    if "YearsSinceLastPromotion" in out.columns and "YearsAtCompany" in out.columns:
        out["PromotionGap"] = out["YearsSinceLastPromotion"] / (out["YearsAtCompany"] + 1.0)

    # Tenure ratios.
    if "YearsInCurrentRole" in out.columns and "YearsAtCompany" in out.columns:
        out["TenureRatio_CurrentRole"] = out["YearsInCurrentRole"] / (out["YearsAtCompany"] + 1.0)
    if "YearsWithCurrManager" in out.columns and "YearsAtCompany" in out.columns:
        out["TenureRatio_Manager"] = out["YearsWithCurrManager"] / (out["YearsAtCompany"] + 1.0)
    if "YearsAtCompany" in out.columns and "TotalWorkingYears" in out.columns:
        out["TenureRatio_TotalWorking"] = out["YearsAtCompany"] / (out["TotalWorkingYears"] + 1.0)

    # Overtime x distance interaction.
    if "OverTime" in out.columns and "DistanceFromHome" in out.columns:
        overtime_flag = (out["OverTime"].astype(str).str.lower() == "yes").astype(float)
        out["OverTimeXDistance"] = overtime_flag * out["DistanceFromHome"]

    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop constants, identifiers, and any column not present in the source data."""
    out = df.copy()
    for col in DROP_CONSTANTS + DROP_ID:
        if col in out.columns:
            out = out.drop(columns=[col])
    out = engineer_features(out)
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` feature lists for a cleaned frame.

    Detection is dtype-based rather than ``dtype == object`` because pandas 3.x infers
    ``str`` dtype for text columns, which would otherwise be misread as numeric.
    """
    exclude = set(EXCLUDED_FROM_FEATURES)
    candidates = [
        c
        for c in df.columns
        if c not in exclude and c != TARGET_ATTRITION
    ]
    categorical = [c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in candidates if c not in categorical]
    # Ordinal fields stay numeric but must not be z-scored into meaninglessness;
    # they are handled by the passthrough branch below.
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build the scikit-learn ``ColumnTransformer`` for the cleaned frame.

    Numeric columns are median-imputed and z-scored. Ordinal columns are imputed and
    passed through unscaled (their scale is meaningful). Categorical columns are
    median-imputed, one-hot encoded and *not* scaled.
    """
    numeric, categorical = feature_columns(df)
    ordinal = [c for c in ORDINAL_FEATURES if c in numeric]
    continuous = [c for c in numeric if c not in ordinal]

    return ColumnTransformer(
        transformers=[
            (
                "continuous",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                continuous,
            ),
            (
                "ordinal",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                ordinal,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(df: pd.DataFrame, model) -> Pipeline:
    """Wrap a model with a preprocessor built from the schema of ``df``."""
    return Pipeline([("preprocessor", build_preprocessor(df)), ("model", model)])


def load_dataset(path) -> pd.DataFrame:
    """Load the raw CSV and attach the derived ``AgeBand`` column."""
    df = pd.read_csv(path)
    return add_age_band(df)


def load_cleaned(path) -> pd.DataFrame:
    """Load the raw CSV, clean it, and attach ``AgeBand``."""
    return clean(load_dataset(path))
