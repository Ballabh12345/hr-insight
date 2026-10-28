"""MVP-3 - Fairness Audit, Mitigation, and Explainability.

Three studies are run and persisted:

**Study A - Real-data audit.** The trained attrition model is audited across four
protected attributes (Gender, AgeBand, MaritalStatus, Department) using Fairlearn's
``MetricFrame``: selection rate, true-positive rate, false-positive rate, and the headline
demographic-parity and equal-opportunity gaps.

**Study B - Mitigation comparison.** Two standard interventions are applied and compared
against the unmitigated baseline:

* **Reweighting** (pre-processing) - training samples are re-weighted so that the sensitive
  attribute and the label become independent.
* **Equalized-odds threshold post-processing** - group-specific thresholds are derived so
  that true-positive and false-positive rates match across groups.

**Study C - Detector power curve.** The synthetic datasets from ``ml/generate_synthetic.py``
carry a *known* unfair component. Sweeping the bias strength from 0 to 2 shows whether the
detector actually finds it, and whether the mitigations remove it. Without a ground-truth
dataset, a fairness audit can only report a number; this study shows the detector works.

A **risk regression** (Gradient Boosting on the binary label) is included so that
group-level error can be measured as well as group-level rate - a model can have equal
selection rates while still being much more accurate for one group than another.

Usage
-----
    python ml/train_fairness.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import fairness as FAIR  # noqa: E402
from ml.common import features as F  # noqa: E402
from ml.common import io  # noqa: E402

RANDOM_STATE = 42
TEST_SIZE = 0.20


def _split(raw: pd.DataFrame):
    """Reproduce the exact train/test split used by ``ml/train_attrition.py``."""
    cleaned = F.clean(raw)
    X = cleaned.drop(columns=[F.TARGET_ATTRITION])
    y = (cleaned[F.TARGET_ATTRITION] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    cleaned["GenderXAgeBand"] = cleaned["Gender"].astype(str) + " / " + cleaned["AgeBand"].astype(str)
    cleaned["GenderXDepartment"] = cleaned["Gender"].astype(str) + " / " + cleaned["Department"].astype(str)
    sensitive = {
        "Gender": cleaned["Gender"],
        "AgeBand": cleaned["AgeBand"],
        "MaritalStatus": cleaned["MaritalStatus"],
        "Department": cleaned["Department"],
        "GenderXAgeBand": cleaned["GenderXAgeBand"],
        "GenderXDepartment": cleaned["GenderXDepartment"],
    }
    return cleaned, X, y, X_train, X_test, y_train, y_test, sensitive


def study_a_baseline_audit(X_test, y_test, predictor, sensitive) -> dict:
    """Audit the production model across every protected attribute."""
    y_pred = predictor.predict(X_test)
    sf = {k: v.loc[X_test.index] for k, v in sensitive.items()}
    report = FAIR.audit(y_test, y_pred, sf)

    # Bootstrap CIs on every gap + per-group calibration.
    bootstrap: dict[str, tuple[float, float]] = {}
    calibration: dict[str, dict[str, dict[str, float]]] = {}
    for attr in report["groups"]:
        if attr not in sf:
            continue
        bootstrap[attr] = FAIR.bootstrap_gap_ci(
            y_test, y_pred, sf[attr], seed=RANDOM_STATE
        )
    # Calibration uses the model's calibrated probabilities where available.
    try:
        proba = predictor.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None
    if proba is not None:
        for attr in report["groups"]:
            if attr not in sf:
                continue
            calibration[attr] = FAIR.calibration_by_group(y_test, proba, sf[attr])

    report["bootstrap_gap_ci"] = {k: list(v) for k, v in bootstrap.items()}
    report["calibration_by_group"] = calibration

    print("[studyA] disparate impact (selection rate) by attribute:")
    for attr, block in report["groups"].items():
        rates = block["selection_rate"]
        spread = max(rates.values()) - min(rates.values())
        ci = report["bootstrap_gap_ci"].get(attr, [float("nan"), float("nan")])
        detail = ", ".join(f"{g}={r:.3f}" for g, r in rates.items())
        print(f"           {attr:15s} gap={spread:.3f}  CI=[{ci[0]:.3f},{ci[1]:.3f}]  ({detail})")
    return report


def study_b_mitigation(X_train, y_train, X_test, y_test, sensitive) -> dict:
    """Compare reweighting and equalized-odds post-processing against the baseline."""
    base = F.build_pipeline(
        F.clean(F.load_dataset(io.download_ibm_hr())),
        LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
    )
    base.fit(X_train, y_train)
    base_pred = base.predict(X_test)
    baseline = FAIR.audit(
        y_test, base_pred, {k: v.loc[X_test.index] for k, v in sensitive.items()}
    )

    gender_train = sensitive["Gender"].loc[X_train.index].reset_index(drop=True)
    gender_test = sensitive["Gender"].loc[X_test.index].reset_index(drop=True)

    # --- Mitigation 1: reweighting (pre-processing) ---
    reweighed = FAIR.fit_reweighing(base, X_train, y_train, gender_train)
    reweighed_pred = reweighed.predict(X_test)
    reweighed_report = FAIR.audit(
        y_test, reweighed_pred, {k: v.loc[X_test.index] for k, v in sensitive.items()}
    )

    # --- Mitigation 2: equalized-odds threshold post-processing ---
    post = FAIR.fit_threshold_optimizer(base, X_train, y_train, gender_train)
    # The randomised interpolation happens at predict time, so the seed goes here.
    post_pred = post.predict(X_test, sensitive_features=gender_test, random_state=RANDOM_STATE)
    post_report = FAIR.audit(
        y_test, post_pred, {k: v.loc[X_test.index] for k, v in sensitive.items()}
    )

    def gap(report, attr):
        return report["groups"][attr]["summary"]["demographic_parity_difference"]

    print("[studyB] demographic parity gap on Gender:")
    print(f"           baseline            {gap(baseline, 'Gender'):.4f}")
    print(f"           reweighting         {gap(reweighed_report, 'Gender'):.4f}")
    print(f"           equalized-odds      {gap(post_report, 'Gender'):.4f}")
    print("[studyB] equal opportunity gap on Gender:")
    eo = "equal_opportunity_difference"
    print(f"           baseline            {baseline['groups']['Gender']['summary'][eo]:.4f}")
    print(f"           reweighting         {reweighed_report['groups']['Gender']['summary'][eo]:.4f}")
    print(f"           equalized-odds      {post_report['groups']['Gender']['summary'][eo]:.4f}")

    return {
        "baseline": baseline,
        "reweighting": reweighed_report,
        "equalized_odds": post_report,
    }


def study_b2_risk_regression(cleaned, X_train, y_train, X_test, y_test, sensitive) -> dict:
    """Group-level error analysis with a regression formulation of the risk score."""
    model_frame = cleaned.drop(columns=["GenderXAgeBand", "GenderXDepartment"], errors="ignore")
    reg = F.build_pipeline(
        model_frame, GradientBoostingRegressor(random_state=RANDOM_STATE)
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    baseline = FAIR.regression_group_mae(
        y_test, preds, {k: v.loc[X_test.index] for k, v in sensitive.items()}
    )

    gender_train = sensitive["Gender"].loc[X_train.index].reset_index(drop=True)
    reweighted = F.build_pipeline(
        model_frame, GradientBoostingRegressor(random_state=RANDOM_STATE)
    )
    rw = FAIR.fit_reweighing(reweighted, X_train, y_train, gender_train)
    rw_preds = rw.predict(X_test)
    mitigated = FAIR.regression_group_mae(
        y_test, rw_preds, {k: v.loc[X_test.index] for k, v in sensitive.items()}
    )

    print("[studyB2] risk-regression MAE ratio by attribute (lower is fairer):")
    for attr in baseline:
        print(
            f"           {attr:15s} baseline={baseline[attr]['mae_ratio']:.4f}  "
            f"reweighted={mitigated[attr]['mae_ratio']:.4f}"
        )

    return {
        "baseline": baseline,
        "reweighted": mitigated,
        "test_mae": float(np.mean(np.abs(np.asarray(y_test) - preds))),
    }


def study_c_power_curve() -> dict:
    """Does the detector find bias we injected on purpose?"""
    manifest = io.read_json(io.ARTIFACTS_DIR / "synthetic_manifest.json")
    results = []

    print("[studyC] detector power curve (gap within R&D, Female vs Male):")
    for entry in manifest["files"]:
        path = io.PROCESSED_DIR / entry["path"]
        if not path.exists():
            continue
        df = F.load_dataset(path)
        X = F.clean(df).drop(columns=[F.TARGET_ATTRITION])
        y = (df["Attrition"] == "Yes").astype(int)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        model = F.build_pipeline(
            F.clean(df),
            LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        )
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # Compare selection rates *within* R&D so the injected effect is isolated from
        # the different base rates of the two genders.
        dept = df["Department"].loc[Xte.index]
        gender = df["Gender"].loc[Xte.index]
        rnd = dept == "Research & Development"
        rates = {}
        for g in ("Female", "Male"):
            mask = rnd & (gender == g)
            rates[g] = float(pred[mask].mean()) if mask.sum() else float("nan")
        gap = abs(rates["Female"] - rates["Male"])

        auc = float(roc_auc_score(yte, proba))
        results.append(
            {
                "bias_strength": entry["bias_strength"],
                "file": entry["path"],
                "attrition_rate": entry["attrition_rate"],
                "selection_rate_female_rnd": rates["Female"],
                "selection_rate_male_rnd": rates["Male"],
                "detected_gap": gap,
                "model_roc_auc": auc,
            }
        )
        print(
            f"           bias={entry['bias_strength']:<4g} "
            f"female_rnd={rates['Female']:.3f} male_rnd={rates['Male']:.3f} "
            f"gap={gap:.4f}  auc={auc:.3f}"
        )

    return {"datasets": results}


def main() -> None:
    started = time.time()
    print("=" * 72)
    print("MVP-3  Fairness Audit, Mitigation & Explainability")
    print("=" * 72)

    raw = F.load_dataset(io.download_ibm_hr())
    cleaned, X, y, X_train, X_test, y_train, y_test, sensitive = _split(raw)

    predictor = io.load_artifact("attrition_predictor.pkl")

    baseline_audit = study_a_baseline_audit(X_test, y_test, predictor, sensitive)

    intersectional_audit = FAIR.audit(
        y_test,
        predictor.predict(X_test),
        {k: v.loc[X_test.index] for k, v in sensitive.items() if k in ("GenderXAgeBand", "GenderXDepartment")},
        groups=["GenderXAgeBand", "GenderXDepartment"],
    )

    mitigation = study_b_mitigation(X_train, y_train, X_test, y_test, sensitive)
    regression = study_b2_risk_regression(cleaned, X_train, y_train, X_test, y_test, sensitive)
    power_curve = study_c_power_curve()

    bundle = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "baseline_audit": baseline_audit,
        "intersectional_audit": intersectional_audit,
        "mitigation": mitigation,
        "risk_regression": regression,
        "power_curve": power_curve,
        "sensitive_attributes": FAIR.FAIRNESS_GROUPS,
        "primary_sensitive_attribute": FAIR.PRIMARY_SENSITIVE,
        "limitations": [
            "Equalized-odds post-processing requires the sensitive attribute to be known "
            "at decision time, which is not always operationally available.",
            "Reweighting cannot remove bias that is encoded in features (Kossen et al., 2021); "
            "it only balances the label distribution across groups.",
            "Fairness is measured with respect to the chosen metrics on this dataset only. "
            "The IBM dataset is fictional, so no conclusion is drawn about real employment decisions.",
            "Groups with small counts produce unstable rates; the Human Resources "
            "department has only 63 employees in total.",
        ],
        "training_seconds": round(time.time() - started, 2),
    }
    io.write_json(bundle, io.ARTIFACTS_DIR / "fairness_report.json")
    print(f"[save] {io.ARTIFACTS_DIR / 'fairness_report.json'}")
    print(f"[done] {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
