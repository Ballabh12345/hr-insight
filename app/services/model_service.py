"""Model access: predictions, explanations, and the training reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml" / "artifacts"
RAW_CSV = ROOT / "data" / "raw" / "ibm_hr.csv"


def _artifact_path(name: str) -> Path:
    return ARTIFACTS / name


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the fitted attrition predictor."""
    path = _artifact_path("attrition_predictor.pkl")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python ml/train_all.py` first."
        )
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_explainer():
    """Load the base model and build a SHAP explainer over the transformed space."""
    import numpy as np

    from ml.common import explain as EX

    pipeline = joblib.load(_artifact_path("attrition_model.pkl"))
    meta = joblib.load(_artifact_path("attrition_features.pkl"))
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = list(pre.get_feature_names_out())

    background = None
    if meta.get("is_linear"):
        # A small, deterministic background sample is enough for LinearExplainer.
        from ml.common import features as F

        cleaned = F.clean(F.load_dataset(RAW_CSV))
        X = cleaned.drop(columns=[F.TARGET_ATTRITION])
        background = np.asarray(pre.transform(X))[:100]

    return EX.ShapExplainer(
        model, feature_names, is_linear=bool(meta.get("is_linear")), background=background
    ), pre, meta


@st.cache_data(show_spinner=False)
def load_attrition_metrics() -> dict:
    """Load the attrition training report."""
    path = _artifact_path("attrition_metrics.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_shap_global() -> list[dict[str, Any]]:
    """Load the global SHAP importance ranking."""
    path = _artifact_path("shap_global.json")
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("importance", [])


@st.cache_data(show_spinner=False)
def load_fairness_report() -> dict:
    """Load the fairness audit report."""
    path = _artifact_path("fairness_report.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_ninebox_pca() -> dict:
    """Load the 2-D PCA projection of the potential space."""
    path = _artifact_path("ninebox_pca.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_survival_metrics() -> dict:
    """Load the survival analysis report."""
    path = _artifact_path("survival_metrics.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def explain_employee(employee_number: int) -> list[dict[str, Any]]:
    """Return the top SHAP reason codes for one employee.

    The employee's row is rebuilt from the raw dataset, transformed with the fitted
    preprocessor, and explained with the same explainer used for global importance.
    """
    from ml.common import features as F

    explainer, pre, meta = load_explainer()

    cleaned = F.clean(F.load_dataset(RAW_CSV))
    X = cleaned.drop(columns=[F.TARGET_ATTRITION])

    # EmployeeNumber was dropped from the modelling frame, so re-attach it to locate the row.
    numbers = pd.read_csv(RAW_CSV)["EmployeeNumber"]
    mask = numbers.values == employee_number
    if not mask.any():
        return []
    row = X.loc[mask].iloc[[0]]

    transformed = pre.transform(row)
    values = {
        col: row.iloc[0][col]
        for col in row.columns
        if col not in ("Attrition",)
    }
    return explainer.local_reasons(transformed, row_values=values, top_n=3)


@st.cache_data(show_spinner=False)
def get_employee_feature_values(employee_number: int) -> dict[str, Any]:
    """Return a small set of editable feature values for one employee."""
    from ml.common import features as F

    raw = pd.read_csv(RAW_CSV)
    cleaned = F.clean(F.load_dataset(RAW_CSV))
    numbers = raw["EmployeeNumber"]
    mask = numbers.values == employee_number
    if not mask.any():
        return {}
    row = cleaned.loc[mask].iloc[0]
    keys = [
        "OverTime",
        "DistanceFromHome",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "MonthlyIncome",
        "JobInvolvement",
        "TrainingTimesLastYear",
    ]
    return {k: row.get(k) for k in keys if k in cleaned.columns}


def predict_with_overrides(employee_number: int, overrides: dict[str, Any]) -> float | None:
    """Re-score one employee after applying feature overrides.

    Returns the new attrition probability, or ``None`` if the employee is unknown.
    Engineered features are recomputed so interactions stay consistent.
    """
    from ml.common import features as F

    predictor = load_predictor()
    raw = pd.read_csv(RAW_CSV)
    cleaned = F.clean(F.load_dataset(RAW_CSV))
    numbers = raw["EmployeeNumber"]
    mask = numbers.values == employee_number
    if not mask.any():
        return None

    for key, value in (overrides or {}).items():
        if key in cleaned.columns:
            cleaned.loc[mask, key] = value

    cleaned = F.engineer_features(cleaned)
    X = cleaned.drop(columns=[F.TARGET_ATTRITION])
    row = X.loc[mask].iloc[[0]]
    return float(predictor.predict_proba(row)[:, 1][0])
