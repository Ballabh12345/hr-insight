"""Feature engineering tests - including regression tests for the leakage we found."""

from __future__ import annotations

import numpy as np


def test_constants_and_id_are_dropped(cleaned_df):
    from ml.common import features as F

    for col in F.DROP_CONSTANTS + F.DROP_ID:
        assert col not in cleaned_df.columns


def test_age_band_is_present_and_ordered(raw_df):
    from ml.common import features as F

    df = F.add_age_band(raw_df)
    assert "AgeBand" in df.columns
    assert set(df["AgeBand"].unique()) <= {"18-25", "26-35", "36-45", "46+"}


def test_categorical_detection_handles_pandas3_string_dtype(cleaned_df):
    """pandas 3.x infers `str` dtype, not `object`. A naive check misreads it as numeric."""
    from ml.common import features as F

    numeric, categorical = F.feature_columns(cleaned_df)
    assert "Department" in categorical
    assert "JobRole" in categorical
    assert "OverTime" in categorical
    assert "Age" in numeric
    assert "MonthlyIncome" in numeric


def test_label_and_manager_rating_are_excluded_from_features(cleaned_df):
    from ml.common import features as F

    numeric, categorical = F.feature_columns(cleaned_df)
    assert F.TARGET_ATTRITION not in numeric + categorical
    assert F.TARGET_PERFORMANCE not in numeric + categorical


def test_percent_salary_hike_is_a_deterministic_proxy_for_the_rating(cleaned_df):
    """Regression test for the leakage found in MVP-4.

    If this ever stops holding, the nine-box exclusion of ``PercentSalaryHike``
    would need revisiting - but the finding must be documented, not silently dropped.
    """
    groups = cleaned_df.groupby("PercentSalaryHike")["PerformanceRating"].nunique()
    assert groups.max() == 1, "PercentSalaryHike no longer proxies PerformanceRating"


def test_rating_is_not_predictable_without_the_proxy(cleaned_df):
    """The honest performance estimate: predicting rating 4 vs 3 is at chance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from ml.common import features as F

    frame = cleaned_df.drop(
        columns=[c for c in ("PerformanceRating", "PercentSalaryHike", "Attrition", "AgeBand")
                 if c in cleaned_df.columns]
    )
    y = (cleaned_df["PerformanceRating"] == 4).astype(int)
    pipeline = F.build_pipeline(frame, LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    auc = cross_val_score(pipeline, frame, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    assert auc < 0.55, f"rating unexpectedly became predictable (AUC {auc:.3f})"


def test_preprocessor_output_is_all_numeric(cleaned_df):
    from ml.common import features as F

    pre = F.build_preprocessor(cleaned_df)
    transformed = pre.fit_transform(cleaned_df.drop(columns=[F.TARGET_ATTRITION]))
    assert isinstance(transformed, np.ndarray)
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unseen_categories(cleaned_df):
    from ml.common import features as F

    pre = F.build_preprocessor(cleaned_df)
    pre.fit(cleaned_df.drop(columns=[F.TARGET_ATTRITION]))
    row = cleaned_df.drop(columns=[F.TARGET_ATTRITION]).head(1).copy()
    row["JobRole"] = "Astronaut"
    assert pre.transform(row).shape[1] == len(pre.get_feature_names_out())
