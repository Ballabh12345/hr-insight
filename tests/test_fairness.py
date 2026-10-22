"""Fairness helpers - reweighting correctness and audit shape."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_reweighting_weights_make_group_and_label_independent():
    """After reweighting, the weighted P(label | group) must equal the overall P(label)."""
    from ml.common.fairness import compute_reweighting_weights

    rng = np.random.default_rng(0)
    n = 4000
    group = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
    label = rng.choice([0, 1], size=n, p=[0.8, 0.2])

    weights = compute_reweighting_weights(label, pd.Series(group))

    total = weights.sum()
    overall_positive = weights[label == 1].sum() / total
    for g in ("A", "B"):
        mask = group == g
        group_total = weights[mask].sum()
        conditional_positive = weights[mask & (label == 1)].sum() / group_total
        assert abs(conditional_positive - overall_positive) < 1e-9


def test_reweighting_weights_are_finite_and_positive():
    from ml.common.fairness import compute_reweighting_weights

    rng = np.random.default_rng(1)
    group = rng.choice(["A", "B"], size=500)
    label = rng.choice([0, 1], size=500)
    weights = compute_reweighting_weights(label, pd.Series(group))
    assert np.isfinite(weights).all()
    assert (weights > 0).all()


def test_reweighting_balances_a_deliberately_skewed_group():
    """A group with a 4x higher positive rate must end up with the same rate."""
    from ml.common.fairness import compute_reweighting_weights

    n = 1000
    group = np.array(["A"] * n + ["B"] * n)
    label = np.array([1] * 400 + [0] * 600 + [1] * 100 + [0] * 900)
    weights = compute_reweighting_weights(label, pd.Series(group))

    total = weights.sum()
    overall = weights[label == 1].sum() / total
    for g in ("A", "B"):
        mask = group == g
        conditional = weights[mask & (label == 1)].sum() / weights[mask].sum()
        assert abs(conditional - overall) < 1e-9


def test_audit_reports_the_expected_structure():
    from ml.common.fairness import audit

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    sensitive = {
        "Gender": pd.Series(["F", "M", "F", "M", "M", "F", "M", "F"]),
        "Constant": pd.Series(["X"] * 8),
    }
    report = audit(y_true, y_pred, sensitive, groups=["Gender", "Constant"])

    assert "Gender" in report["groups"]
    assert "Constant" not in report["groups"], "single-valued attributes must be skipped"
    block = report["groups"]["Gender"]
    for metric in ("selection_rate", "true_positive_rate", "false_positive_rate", "count"):
        assert metric in block
    assert 0.0 <= block["summary"]["demographic_parity_difference"] <= 1.0
    assert block["summary"]["demographic_parity_ratio"] >= 0.0


def test_regression_group_mae_ratio():
    from ml.common.fairness import regression_group_mae

    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.0, 0.5, 0.0, 1.0])
    sensitive = {"Gender": pd.Series(["F", "M", "F", "M"])}
    out = regression_group_mae(y_true, y_pred, sensitive, groups=["Gender"])
    assert out["Gender"]["mae_by_group"]["F"] == 0.0
    assert out["Gender"]["mae_by_group"]["M"] == 0.25
    assert out["Gender"]["mae_ratio"] > 1.0


def test_reweigher_fits_a_pipeline():
    from sklearn.linear_model import LogisticRegression

    from ml.common.fairness import Reweigher

    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=400) > 0).astype(int)
    sensitive = pd.Series(rng.choice(["A", "B"], size=400))

    pipeline = LogisticRegression()
    rw = Reweigher(pipeline)
    rw.fit(X, y, sensitive)
    assert rw.predict(X).shape == (400,)
    assert rw.weights_.shape == (400,)
