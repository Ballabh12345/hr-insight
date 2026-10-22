"""Assemble the dataset the application serves.

Reads the trained artifacts and the raw IBM data, then produces:

* ``data/processed/employees.csv`` - one row per employee with the attrition probability,
  risk band, retention action, nine-box placement, persona and derived fields.
* ``ml/artifacts/shap_global.json`` - global SHAP feature importance.
* ``ml/artifacts/dataset_summary.json`` - headline numbers for the Overview page.

Usage
-----
    python ml/build_dataset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import explain as EX  # noqa: E402
from ml.common import features as F  # noqa: E402
from ml.common import io, metrics  # noqa: E402

RANDOM_STATE = 42


def retention_action(
    band: str,
    performance_tier: str,
    years_at_company: float,
    engagement_z: float,
) -> str:
    """Map a risk assessment onto a recommended HR action.

    The rules are ordered and deterministic so that a manager can audit exactly why a
    given action was proposed. This table is reproduced in the model card.
    """
    if band == "Low":
        return "No action required - development opportunity"
    if band == "Medium":
        return "Schedule a 90-day check-in"
    # High risk band from here on.
    if years_at_company <= 2:
        return "Onboarding check-in within 2 weeks"
    if performance_tier == "High":
        return "Compensation review + high-visibility project assignment"
    if engagement_z < -0.25:
        return "Workload rebalance or role change conversation"
    if years_at_company >= 8:
        return "Retention bonus discussion"
    return "Structured career conversation"


def main() -> None:
    started = time.time()
    print("=" * 72)
    print("BUILD  Assembling the application dataset")
    print("=" * 72)

    raw = F.load_dataset(io.download_ibm_hr())
    cleaned = F.clean(raw)
    cleaned["PromotionVelocity"] = cleaned["YearsAtCompany"] / cleaned["JobLevel"].clip(lower=1)
    # EmployeeNumber is dropped from the *model* feature space but kept as the row id.
    employee_number = raw["EmployeeNumber"]

    # ------------------------------------------------------- attrition model
    predictor = io.load_artifact("attrition_predictor.pkl")
    feature_meta = io.load_artifact("attrition_features.pkl")
    bundle = io.read_json(io.ARTIFACTS_DIR / "attrition_metrics.json")

    X = cleaned.drop(columns=[F.TARGET_ATTRITION])
    proba = predictor.predict_proba(X)[:, 1]
    threshold = bundle["decision_threshold"]
    predicted_attrition = (proba >= threshold).astype(int)

    bands = [metrics.band_risk(p, bundle["band_thresholds"]["low"], bundle["band_thresholds"]["high"])
             for p in proba]

    # ---------------------------------------------------------- nine-box data
    # Assignments come straight from the training script, so the labels the app serves
    # are byte-for-byte the ones the model card describes.
    assignments_path = io.PROCESSED_DIR / "ninebox_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"{assignments_path} not found. Run `python ml/train_ninebox.py` first."
        )
    assignments = pd.read_csv(assignments_path).set_index("EmployeeNumber")
    assignments = assignments.reindex(employee_number.values)
    assignments.index = employee_number.index

    engagement_z = assignments["engagement_composite"]
    performance_tier = assignments["performance_tier"]
    potential_tier = assignments["potential_tier"]
    ninebox_label = assignments["ninebox_label"]
    cluster_labels = assignments["persona_cluster"].to_numpy()

    # --------------------------------------------------------- assemble frame
    out = pd.DataFrame(
        {
            "EmployeeNumber": employee_number,
            "Age": cleaned["Age"],
            "AgeBand": cleaned["AgeBand"],
            "Gender": cleaned["Gender"],
            "MaritalStatus": cleaned["MaritalStatus"],
            "Department": cleaned["Department"],
            "JobRole": cleaned["JobRole"],
            "JobLevel": cleaned["JobLevel"],
            "BusinessTravel": cleaned["BusinessTravel"],
            "OverTime": cleaned["OverTime"],
            "YearsAtCompany": cleaned["YearsAtCompany"],
            "YearsInCurrentRole": cleaned["YearsInCurrentRole"],
            "YearsSinceLastPromotion": cleaned["YearsSinceLastPromotion"],
            "MonthlyIncome": cleaned["MonthlyIncome"],
            "JobSatisfaction": cleaned["JobSatisfaction"],
            "EngagementComposite": engagement_z.round(4),
            "attrition_probability": proba.round(6),
            "risk_band": bands,
            "predicted_attrition": predicted_attrition,
            "actual_attrition": cleaned["Attrition"],
            "performance_tier": performance_tier,
            "potential_tier": potential_tier,
            "ninebox_label": ninebox_label,
            "persona_cluster": cluster_labels,
        }
    )
    out["retention_action"] = [
        retention_action(
            band,
            perf,
            years,
            eng,
        )
        for band, perf, years, eng in zip(
            out["risk_band"], out["performance_tier"], out["YearsAtCompany"], engagement_z
        )
    ]

    path = io.PROCESSED_DIR / "employees.csv"
    out.to_csv(path, index=False)
    print(f"[save] {path}  ({len(out)} rows x {out.shape[1]} cols)")

    # ------------------------------------------------------------ SHAP global
    base_pipeline = io.load_artifact("attrition_model.pkl")
    pre = base_pipeline.named_steps["preprocessor"]
    linear_model = base_pipeline.named_steps["model"]
    feature_names = list(pre.get_feature_names_out())

    Xt = pre.transform(X)
    is_linear = feature_meta.get("is_linear", False)
    explainer = EX.ShapExplainer(
        linear_model, feature_names, is_linear=is_linear, background=Xt
    )
    importance = explainer.global_importance(Xt, top_n=20)
    io.write_json(
        {"importance": importance, "unit": "mean absolute SHAP value (log-odds)"},
        io.ARTIFACTS_DIR / "shap_global.json",
    )
    print("[shap ] top features:")
    for item in importance[:8]:
        print(f"          {item['label']:38s} {item['importance']:.4f}")

    # -------------------------------------------------------------- summary
    summary = {
        "n_employees": int(len(out)),
        "attrition_rate": float((out["actual_attrition"] == "Yes").mean()),
        "high_risk_count": int((out["risk_band"] == "High").sum()),
        "high_risk_rate": float((out["risk_band"] == "High").mean()),
        "medium_risk_count": int((out["risk_band"] == "Medium").sum()),
        "low_risk_count": int((out["risk_band"] == "Low").sum()),
        "mean_probability": float(out["attrition_probability"].mean()),
        "departments": out["Department"].value_counts().to_dict(),
        "risk_by_department": out.groupby("Department")["attrition_probability"]
        .agg(["mean", "count", "sum"])
        .round(4)
        .to_dict("index"),
        "ninebox_counts": out["ninebox_label"].value_counts().to_dict(),
        "ninebox_risk": out.groupby("ninebox_label")["attrition_probability"]
        .agg(["mean", "count"])
        .round(4)
        .to_dict("index"),
        "risk_band_counts": out["risk_band"].value_counts().to_dict(),
        "model": {
            "selected": bundle["selected_model"],
            "calibration": bundle["calibration_method"],
            "decision_threshold": bundle["decision_threshold"],
            "test_metrics": bundle["calibrated_test_metrics"],
        },
        "built_seconds": round(time.time() - started, 2),
    }
    io.write_json(summary, io.ARTIFACTS_DIR / "dataset_summary.json")
    print(f"[save] {io.ARTIFACTS_DIR / 'dataset_summary.json'}")
    print(f"[done] {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
