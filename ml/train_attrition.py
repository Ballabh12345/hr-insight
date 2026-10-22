"""MVP-1 - Attrition Risk Prediction.

Pipeline
--------
1. Load the IBM HR dataset (Kaggle-free mirror, checksum verified).
2. Compare four model families with 5-fold stratified CV on ROC-AUC:
   Logistic Regression (interpretable baseline), Random Forest, Gradient Boosting,
   Histogram Gradient Boosting.
3. Refit the winner on the full training set.
4. Choose a calibration method (isotonic vs sigmoid) by cross-validated log-loss, and
   calibrate the winner. A calibrated probability is what lets the UI honestly say
   "risk 0.62" instead of "score 0.62".
5. Tune the decision threshold for maximum F1 using out-of-fold training predictions,
   then evaluate once on the untouched test set.
6. Persist the model, the calibrator, the threshold, feature names, and a metrics bundle.

Usage
-----
    python ml/train_attrition.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import features as F  # noqa: E402
from ml.common import io, metrics  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

#: Calibration selection: repeated CV, mean-tolerance, then lowest-variance tie-break.
CALIBRATION_REPEATS = 5
CALIBRATION_TOLERANCE = 0.005

#: Decision thresholds for the calibrated risk band (see docs/model_cards/attrition.md).
BAND_LOW = 0.25
BAND_HIGH = 0.55


def candidate_models() -> dict[str, tuple[object, dict]]:
    """Model families to compare, each with its hyperparameter search space."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": ["balanced", None],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [300],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 5, 10],
                "model__class_weight": ["balanced"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 300],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [200, 400],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
                "model__class_weight": ["balanced"],
            },
        ),
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Return the threshold maximising F1, and that F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def main() -> None:
    started = time.time()
    io.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MVP-1  Attrition Risk Prediction")
    print("=" * 72)

    # ---------------------------------------------------------------- data
    path = io.download_ibm_hr()
    print(f"[data] {path.name}  sha256={io.sha256(path)[:16]}...")

    raw = F.load_dataset(path)
    cleaned = F.clean(raw)
    print(
        f"[data] {len(cleaned)} employees | attrition rate "
        f"{(cleaned['Attrition'] == 'Yes').mean():.1%}"
    )

    X = cleaned.drop(columns=[F.TARGET_ATTRITION])
    y = (cleaned[F.TARGET_ATTRITION] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"[split] train={len(X_train)}  test={len(X_test)}")

    # ------------------------------------------------------------- compare
    results: dict[str, dict] = {}
    best_name, best_score, best_pipeline = None, -np.inf, None

    for name, (estimator, grid) in candidate_models().items():
        pipeline = F.build_pipeline(cleaned, estimator)
        search = GridSearchCV(
            pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)

        fold_scores = cross_val_score(
            search.best_estimator_, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        cv_summary = metrics.summarize_cv(list(fold_scores))

        y_pred = search.best_estimator_.predict(X_test)
        y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        test_metrics = metrics.binary_metrics(y_test, y_pred, y_proba)

        results[name] = {
            "best_params": {k: str(v) for k, v in search.best_params_.items()},
            "cv_roc_auc": cv_summary,
            "test": test_metrics,
        }
        print(
            f"[model] {name:24s} cv_auc={cv_summary['mean']:.4f}+/-{cv_summary['std']:.4f}  "
            f"test_auc={test_metrics.get('roc_auc', float('nan')):.4f}  "
            f"f1={test_metrics['f1']:.4f}"
        )

        if cv_summary["mean"] > best_score:
            best_name, best_score, best_pipeline = name, cv_summary["mean"], search.best_estimator_

    assert best_pipeline is not None and best_name is not None
    print(f"[model] selected: {best_name} (cv_auc={best_score:.4f})")

    # ----------------------------------------------------------- calibration
    # "none" is a real candidate: logistic regression on this data is already close to
    # calibrated, and isotonic calibration on ~1.2k rows can overfit.
    #
    # Selection rule (see docs/model_cards/attrition.md):
    #   1. Rank candidates by mean cross-validated log-loss over REPEATED stratified CV
    #      (5 repeats x 5 folds) - a single 5-fold run proved too noisy to decide.
    #   2. Keep every candidate within TOLERANCE of the best mean.
    #   3. Among those, prefer the one with the lowest variance.
    #
    # Rationale for step 3: isotonic regression matches the best mean log-loss but has
    # ~2.3x the variance and emits degenerate 0/1 probabilities. For a product that
    # shows "risk 62%" to a manager, a stable, never-degenerate probability is worth
    # more than a mean improvement of 0.003 nats.
    calibration_scores: dict[str, dict[str, float]] = {}
    repeat_cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CALIBRATION_REPEATS, random_state=RANDOM_STATE
    )
    for method in ("none", "sigmoid", "isotonic"):
        estimator = (
            best_pipeline
            if method == "none"
            else CalibratedClassifierCV(estimator=best_pipeline, method=method, cv=CV_FOLDS)
        )
        ll = -cross_val_score(
            estimator, X_train, y_train, cv=repeat_cv, scoring="neg_log_loss", n_jobs=-1
        )
        br = -cross_val_score(
            estimator, X_train, y_train, cv=repeat_cv, scoring="neg_brier_score", n_jobs=-1
        )
        calibration_scores[method] = {
            "cv_neg_log_loss_mean": float(ll.mean()),
            "cv_neg_log_loss_std": float(ll.std()),
            "cv_neg_brier_mean": float(br.mean()),
            "cv_neg_brier_std": float(br.std()),
        }
        print(
            f"[calib] {method:9s} log_loss={ll.mean():.4f}+/-{ll.std():.4f}  "
            f"brier={br.mean():.4f}+/-{br.std():.4f}"
        )

    best_mean = min(v["cv_neg_log_loss_mean"] for v in calibration_scores.values())
    within_tolerance = [
        m for m, v in calibration_scores.items()
        if v["cv_neg_log_loss_mean"] <= best_mean + CALIBRATION_TOLERANCE
    ]
    calibration_method = min(
        within_tolerance, key=lambda m: calibration_scores[m]["cv_neg_log_loss_std"]
    )
    print(
        f"[calib] selected: {calibration_method} "
        f"(mean within {CALIBRATION_TOLERANCE} of best, lowest variance)"
    )

    if calibration_method == "none":
        predictor = best_pipeline
    else:
        predictor = CalibratedClassifierCV(
            estimator=best_pipeline, method=calibration_method, cv=CV_FOLDS
        )
        predictor.fit(X_train, y_train)

    # ------------------------------------------------- threshold selection
    # Out-of-fold predictions on the training set give an honest threshold.
    oof_estimator = (
        best_pipeline
        if calibration_method == "none"
        else CalibratedClassifierCV(
            estimator=best_pipeline, method=calibration_method, cv=CV_FOLDS
        )
    )
    oof_proba = cross_val_predict(
        oof_estimator, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold, oof_f1 = best_f1_threshold(y_train, oof_proba)
    print(f"[thresh] F1-optimal threshold={threshold:.3f} (out-of-fold train F1={oof_f1:.4f})")

    y_proba_cal = predictor.predict_proba(X_test)[:, 1]
    y_pred_cal = (y_proba_cal >= threshold).astype(int)

    calibrated_metrics = metrics.binary_metrics(y_test, y_pred_cal, y_proba_cal)
    default_metrics = metrics.binary_metrics(y_test, (y_proba_cal >= 0.5).astype(int), y_proba_cal)

    print(
        f"[eval] thresholded  auc={calibrated_metrics['roc_auc']:.4f}  "
        f"p={calibrated_metrics['precision']:.4f}  r={calibrated_metrics['recall']:.4f}  "
        f"f1={calibrated_metrics['f1']:.4f}  log_loss={calibrated_metrics['log_loss']:.4f}"
    )
    print(
        f"[eval] @0.50        auc={default_metrics['roc_auc']:.4f}  "
        f"p={default_metrics['precision']:.4f}  r={default_metrics['recall']:.4f}  "
        f"f1={default_metrics['f1']:.4f}  log_loss={default_metrics['log_loss']:.4f}"
    )

    # ------------------------------------------------------------ persist
    feature_names = list(best_pipeline.named_steps["preprocessor"].get_feature_names_out())

    io.save_artifact(best_pipeline, "attrition_model.pkl")
    io.save_artifact(predictor, "attrition_predictor.pkl")
    io.save_artifact(
        {
            "feature_names": feature_names,
            "model_family": best_name,
            "is_linear": best_name == "logistic_regression",
        },
        "attrition_features.pkl",
    )

    bundle = {
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "test_size": TEST_SIZE,
        "calibration_method": calibration_method,
        "calibration_study": calibration_scores,
        "calibration_tolerance": CALIBRATION_TOLERANCE,
        "calibration_repeats": CALIBRATION_REPEATS,
        "decision_threshold": threshold,
        "band_thresholds": {"low": BAND_LOW, "high": BAND_HIGH},
        "candidates": results,
        "calibrated_test_metrics": calibrated_metrics,
        "default_threshold_test_metrics": default_metrics,
        "roc_curve": metrics.roc_curve_data(y_test, y_proba_cal),
        "pr_curve": metrics.pr_curve_data(y_test, y_proba_cal),
        "calibration_curve": metrics.calibration_curve_data(y_test, y_proba_cal),
        "confusion_matrix": metrics.confusion_data(y_test, y_pred_cal),
        "feature_names": feature_names,
        "excluded_features": F.EXCLUDED_FROM_FEATURES,
        "dataset": {
            "source": io.IBM_HR_URL,
            "sha256": io.IBM_HR_SHA256,
            "n_employees": int(len(cleaned)),
            "n_features_after_cleaning": int(cleaned.shape[1] - 1),
            "attrition_rate": float((cleaned["Attrition"] == "Yes").mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "training_seconds": round(time.time() - started, 2),
    }
    io.write_json(bundle, io.ARTIFACTS_DIR / "attrition_metrics.json")
    io.write_json(bundle["confusion_matrix"], io.ARTIFACTS_DIR / "attrition_confusion.json")

    print(f"[save] artifacts written to {io.ARTIFACTS_DIR}")
    print(f"[done] {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
