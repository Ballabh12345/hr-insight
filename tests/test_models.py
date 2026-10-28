"""Model artefact tests - the trained pipeline behaves as the report claims."""

from __future__ import annotations

import pytest

from ml.common import io


def _require(name: str) -> None:
    if not io.artifact_exists(name):
        pytest.skip(f"{name} not built - run `python ml/train_all.py`")


def test_artifacts_exist():
    for name in (
        "attrition_model.pkl",
        "attrition_predictor.pkl",
        "attrition_features.pkl",
        "attrition_metrics.json",
        "ninebox_kmeans.pkl",
        "ninebox_config.pkl",
        "ninebox_metrics.json",
        "fairness_report.json",
        "shap_global.json",
        "dataset_summary.json",
    ):
        _require(name)


def test_predictions_are_valid_probabilities(employees_df):
    _require("attrition_predictor.pkl")
    probs = employees_df["attrition_probability"]
    assert probs.between(0, 1).all()
    assert probs.nunique() > 100, "predictions look constant"


def test_risk_bands_match_the_probability_thresholds(employees_df):
    from ml.common import metrics

    bundle = io.read_json(io.ARTIFACTS_DIR / "attrition_metrics.json")
    low = bundle["band_thresholds"]["low"]
    high = bundle["band_thresholds"]["high"]
    for row in employees_df.itertuples():
        assert row.risk_band == metrics.band_risk(row.attrition_probability, low, high)


def test_selected_model_beats_the_constant_baseline_on_log_loss():
    """A calibrated model must be better than always predicting the base rate."""
    _require("attrition_metrics.json")
    bundle = io.read_json(io.ARTIFACTS_DIR / "attrition_metrics.json")
    log_loss = bundle["calibrated_test_metrics"]["log_loss"]
    assert log_loss < 0.4395, f"log-loss {log_loss} is no better than a constant predictor"


def test_selected_model_is_logistic_regression():
    """The CV comparison must have selected Logistic Regression (CV AUC 0.841)."""
    _require("attrition_metrics.json")
    bundle = io.read_json(io.ARTIFACTS_DIR / "attrition_metrics.json")
    assert bundle["selected_model"] == "logistic_regression"


def test_calibration_method_is_sigmoid():
    _require("attrition_metrics.json")
    bundle = io.read_json(io.ARTIFACTS_DIR / "attrition_metrics.json")
    assert bundle["calibration_method"] == "sigmoid"


def test_ninebox_fills_all_nine_cells(employees_df):
    _require("ninebox_metrics.json")
    assert employees_df["ninebox_label"].nunique() == 9


def test_ninebox_axes_are_balanced_tertiles(employees_df):
    """Both axes are tertiles of a continuous construct, so they must be near-equal."""
    _require("ninebox_metrics.json")
    perf = employees_df["performance_tier"].value_counts()
    pot = employees_df["potential_tier"].value_counts()
    for counts in (perf, pot):
        assert set(counts.index) == {"Low", "Medium", "High"}
        assert counts.max() - counts.min() <= len(employees_df) * 0.05


def test_shap_global_importance_is_grouped_by_source_feature():
    """One-hot columns must be aggregated, so no label may contain a category value."""
    _require("shap_global.json")
    importance = io.read_json(io.ARTIFACTS_DIR / "shap_global.json")["importance"]
    assert importance
    for item in importance:
        assert "_" not in item["feature"], f"ungrouped one-hot column: {item['feature']}"


def test_shap_top_driver_is_monthly_income():
    _require("shap_global.json")
    importance = io.read_json(io.ARTIFACTS_DIR / "shap_global.json")["importance"]
    assert importance[0]["feature"] == "MonthlyIncome"


def test_fairness_report_contains_all_three_studies():
    _require("fairness_report.json")
    report = io.read_json(io.ARTIFACTS_DIR / "fairness_report.json")
    assert "baseline_audit" in report
    assert set(report["mitigation"]) == {"baseline", "reweighting", "equalized_odds"}
    assert report["power_curve"]["datasets"]


def test_power_curve_detects_injected_bias():
    """The detector must see a wider gap as the injected bias grows."""
    _require("fairness_report.json")
    report = io.read_json(io.ARTIFACTS_DIR / "fairness_report.json")
    gaps = [
        d["detected_gap"] for d in sorted(report["power_curve"]["datasets"], key=lambda d: d["bias_strength"])
    ]
    assert len(gaps) >= 3
    assert gaps[-1] > gaps[0] * 3, f"detector did not separate the bias levels: {gaps}"
    assert all(b >= a for a, b in zip(gaps, gaps[1:])), f"gaps are not monotonic: {gaps}"


def test_fairness_audit_covers_every_protected_attribute():
    _require("fairness_report.json")
    report = io.read_json(io.ARTIFACTS_DIR / "fairness_report.json")
    groups = report["baseline_audit"]["groups"]
    assert set(groups) == {"Gender", "AgeBand", "MaritalStatus", "Department"}


def test_retention_action_is_never_empty(employees_df):
    assert employees_df["retention_action"].notna().all()
    assert (employees_df["retention_action"].str.len() > 0).all()


def test_every_high_risk_employee_gets_an_action(employees_df):
    high = employees_df[employees_df["risk_band"] == "High"]
    assert not high["retention_action"].str.contains("No action").any()
