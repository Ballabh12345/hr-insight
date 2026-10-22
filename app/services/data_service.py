"""Dataset access for the application.

Reads the assembled ``data/processed/employees.csv`` and provides the filtering and
aggregation helpers the views need. Everything is cached, so the file is parsed once
per session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
EMPLOYEES_PATH = ROOT / "data" / "processed" / "employees.csv"
ARTIFACTS = ROOT / "ml" / "artifacts"

RISK_ORDER = ["High", "Medium", "Low"]


@st.cache_data(show_spinner=False)
def load_employees() -> pd.DataFrame:
    """Load the assembled employee dataset."""
    if not EMPLOYEES_PATH.exists():
        raise FileNotFoundError(
            f"{EMPLOYEES_PATH} not found.\n\n"
            "Run `python ml/train_all.py` to build the models and the dataset."
        )
    df = pd.read_csv(EMPLOYEES_PATH)
    df["risk_band"] = pd.Categorical(df["risk_band"], categories=RISK_ORDER, ordered=True)
    df["persona_cluster"] = df["persona_cluster"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    """Load the headline summary produced by ``ml/build_dataset.py``."""
    import json

    path = ARTIFACTS / "dataset_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_ninebox_metrics() -> dict:
    """Load the nine-box training/clustering report."""
    import json

    path = ARTIFACTS / "ninebox_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def filter_employees(
    df: pd.DataFrame,
    department: list[str] | None = None,
    risk_band: list[str] | None = None,
    gender: list[str] | None = None,
    age_band: list[str] | None = None,
    ninebox: list[str] | None = None,
    search: str = "",
) -> pd.DataFrame:
    """Apply the employee-table filters."""
    out = df
    if department:
        out = out[out["Department"].isin(department)]
    if risk_band:
        out = out[out["risk_band"].isin(risk_band)]
    if gender:
        out = out[out["Gender"].isin(gender)]
    if age_band:
        out = out[out["AgeBand"].isin(age_band)]
    if ninebox:
        out = out[out["ninebox_label"].isin(ninebox)]
    if search:
        needle = search.strip().lower()
        out = out[
            out["EmployeeNumber"].astype(str).str.contains(needle, na=False)
            | out["JobRole"].str.lower().str.contains(needle, na=False)
            | out["Department"].str.lower().str.contains(needle, na=False)
        ]
    return out


def employee_options(df: pd.DataFrame) -> dict[int, str]:
    """Build an ``{employee_number: display_label}`` map for the picker."""
    return {
        int(row.EmployeeNumber): f"#{int(row.EmployeeNumber)}  {row.JobRole}  -  {row.Department}"
        for row in df.itertuples()
    }


def risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count employees per risk band, ordered High -> Low."""
    counts = df["risk_band"].value_counts()
    return pd.DataFrame(
        {"Risk band": [str(i) for i in counts.index], "Employees": counts.values}
    ).reindex(columns=["Risk band", "Employees"])
