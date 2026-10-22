"""Evaluation helpers: classification, regression, and curve data for plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
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
