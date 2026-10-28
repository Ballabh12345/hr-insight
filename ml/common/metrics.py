"""Evaluation helpers: classification, regression, and curve data for plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    """Compute the standard binary classification metric set."""
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        out["log_loss"] = float(log_loss(y_true, y_proba, labels=[0, 1]))
        out["brier_score"] = float(brier_score_loss(y_true, y_proba))
    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the standard regression metric set."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def confusion_data(y_true, y_pred, labels=(0, 1)) -> dict[str, Any]:
    """Return the confusion matrix plus row/column-normalised views."""
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = cm / cm.sum(axis=1, keepdims=True)
    return {
        "matrix": cm.tolist(),
        "normalised": np.nan_to_num(normalised).tolist(),
        "labels": list(labels),
    }


def roc_curve_data(y_true, y_proba) -> dict[str, Any]:
    """Return ROC curve points and the AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc_score(y_true, y_proba)),
    }


def pr_curve_data(y_true, y_proba) -> dict[str, Any]:
    """Return precision-recall curve points and the average precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "ap": float(average_precision_score(y_true, y_proba)),
    }


def calibration_curve_data(y_true, y_proba, n_bins: int = 10) -> dict[str, Any]:
    """Return reliability-curve bin centres, observed rates, and mean predictions."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def band_risk(probability: float, low: float = 0.25, high: float = 0.55) -> str:
    """Map a calibrated attrition probability onto a risk band."""
    if probability >= high:
        return "High"
    if probability >= low:
        return "Medium"
    return "Low"


def summarize_cv(fold_scores: list[float]) -> dict[str, float]:
    """Mean and standard deviation across cross-validation folds."""
    arr = np.asarray(fold_scores, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def bootstrap_ci(
    y_true,
    y_score,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Stratified bootstrap confidence interval for a scalar metric.

    Resamples within each class so the class balance is preserved in every
    bootstrap replicate. Returns ``(lower, upper)`` percentiles.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    stats: list[float] = []
    alpha = (1.0 - ci) / 2.0
    for _ in range(n_boot):
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        try:
            value = float(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        return (float("nan"), float("nan"))
    arr = np.asarray(stats, dtype=float)
    return (float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha)))


def corrected_paired_ttest(
    scores_a: list[float],
    scores_b: list[float],
    n_train: int,
    n_test_fold: int,
) -> tuple[float, float]:
    """Nadeau-Bengio corrected paired t-test for cross-validated scores.

    The correction accounts for the fact that the training sets of different
    folds overlap, which the ordinary paired t-test ignores. Returns
    ``(t_statistic, p_value)`` with ``len(scores) - 1`` degrees of freedom.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        return (float("nan"), float("nan"))
    diff = a - b
    mean_diff = float(diff.mean())
    var_diff = float(diff.var(ddof=1))
    # Nadeau-Bengio variance correction.
    se = np.sqrt((1.0 / diff.size + n_test_fold / max(n_train, 1)) * var_diff)
    if se == 0 or not np.isfinite(se):
        return (float("nan"), float("nan"))
    t_stat = mean_diff / se
    df = diff.size - 1
    p_value = float(2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df)))
    return (float(t_stat), p_value)


def cost_optimal_threshold(
    y_true,
    proba,
    cost_leaver: float = 50000.0,
    cost_action: float = 5000.0,
    success_prob: float = 0.30,
) -> dict[str, float]:
    """Threshold minimising expected cost for a cost matrix.

    ``cost_leaver`` is the cost of losing an employee who leaves, ``cost_action``
    is the cost of a retention action, and ``success_prob`` is the probability
    that an action prevents a departure. Returns the optimal threshold plus the
    expected cost at that threshold and at the 0.5 baseline.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)

    def expected_cost(threshold: float) -> float:
        pred = (proba >= threshold).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        # Acting on a leaver costs cost_action and still loses (1-success_prob) of them.
        return (
            (tp + fp) * cost_action
            + tp * (1.0 - success_prob) * cost_leaver
            + fn * cost_leaver
        )

    candidates = sorted(set([0.0, 0.5, 1.0] + [float(p) for p in proba]))
    best_t, best_cost = 0.5, expected_cost(0.5)
    for t in candidates:
        c = expected_cost(t)
        if c < best_cost:
            best_cost = c
            best_t = t
    # Analytic threshold from the per-person decision rule.
    analytic = cost_action / (cost_action + success_prob * cost_leaver) if success_prob > 0 else 0.5
    return {
        "threshold": float(best_t),
        "analytic_threshold": float(analytic),
        "expected_cost": float(best_cost),
        "baseline_cost_at_0.5": float(expected_cost(0.5)),
        "cost_leaver": float(cost_leaver),
        "cost_action": float(cost_action),
        "success_prob": float(success_prob),
    }
