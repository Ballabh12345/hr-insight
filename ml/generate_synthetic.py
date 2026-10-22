"""Synthetic workforce generator with **controlled, known** bias.

The IBM HR Analytics dataset contains only 237 attrition cases and has no ground truth
about *why* any model is unfair. To test the fairness detector properly we need data
where the unfairness is known by construction.

This generator bootstraps the real feature rows from the IBM dataset (so all realistic
feature correlations are preserved) and then **re-draws the attrition label** from an
explicit generative model:

.. code-block:: text

    logit(p) = b0
             + b_over_time   * OverTime
             + b_satisfaction* (JobSatisfaction <= 2)
             + b_distance    * (DistanceFromHome > 15)
             + b_worklife    * (WorkLifeBalance <= 2)
             + b_tenure      * YearsAtCompany
             + b_income      * z(MonthlyIncome)
             + BIAS_TERM

    BIAS_TERM = bias_strength * 1[Gender == 'Female'] * 1[Department == 'Research & Development']

``BIAS_TERM`` is the only unfair component. Everything else is a legitimate driver of
attrition. Sweeping ``bias_strength`` from 0 to 2 therefore produces a *power curve* for
the fairness detector: at 0 there is nothing to find, and at higher values the detector
should see the gap widen.

Usage
-----
    python ml/generate_synthetic.py --n 20000 --sweep 0,0.5,1.0,2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import features as F  # noqa: E402
from ml.common import io  # noqa: E402

RANDOM_STATE = 42

#: Legitimate drivers of attrition (true coefficients on the logit scale).
TRUE_COEFFICIENTS: dict[str, float] = {
    "intercept": -2.30,
    "over_time": 1.30,
    "low_job_satisfaction": 0.85,
    "far_from_home": 0.45,
    "low_work_life_balance": 0.60,
    "years_at_company": 0.020,
    "income_z": -0.25,
}

#: The unfair component. Applied only to this group intersection.
BIAS_GROUP = {"gender": "Female", "department": "Research & Development"}

#: Department whose employees carry the injected bias (matches the IBM data, where
#: Research & Development is the largest department at 961 of 1470 rows).
BIAS_DEPARTMENT = "Research & Development"


def _logit_terms(df: pd.DataFrame) -> np.ndarray:
    """Compute the legitimate part of the attrition logit for each row."""
    c = TRUE_COEFFICIENTS
    income = (df["MonthlyIncome"] - df["MonthlyIncome"].mean()) / (
        df["MonthlyIncome"].std() or 1.0
    )
    return (
        c["intercept"]
        + c["over_time"] * (df["OverTime"] == "Yes").astype(float)
        + c["low_job_satisfaction"] * (df["JobSatisfaction"] <= 2).astype(float)
        + c["far_from_home"] * (df["DistanceFromHome"] > 15).astype(float)
        + c["low_work_life_balance"] * (df["WorkLifeBalance"] <= 2).astype(float)
        + c["years_at_company"] * df["YearsAtCompany"]
        + c["income_z"] * income
    )


def bias_term(df: pd.DataFrame, bias_strength: float) -> np.ndarray:
    """Return the injected unfair component for each row."""
    if bias_strength == 0:
        return np.zeros(len(df))
    in_group = (
        (df["Gender"] == BIAS_GROUP["gender"]) & (df["Department"] == BIAS_DEPARTMENT)
    )
    return bias_strength * in_group.astype(float).values


def generate(
    n: int = 20_000,
    bias_strength: float = 0.0,
    seed: int = RANDOM_STATE,
    source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate ``n`` employees with a controlled level of injected bias."""
    if source is None:
        source = F.load_dataset(io.download_ibm_hr())

    rng = np.random.default_rng(seed)

    # Bootstrap real feature rows so realistic correlations survive.
    idx = rng.choice(len(source), size=n, replace=True)
    out = source.iloc[idx].reset_index(drop=True).copy()

    # Unique identifiers.
    out["EmployeeNumber"] = np.arange(1, n + 1)

    # Re-draw the label from the generative model.
    logits = _logit_terms(out) + bias_term(out, bias_strength)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    out["Attrition"] = np.where(rng.random(n) < probabilities, "Yes", "No")

    # Record the ground truth for the experiment.
    out["bias_strength"] = bias_strength
    out["injected_bias_group"] = (
        (out["Gender"] == BIAS_GROUP["gender"]) & (out["Department"] == BIAS_DEPARTMENT)
    ).astype(int)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20_000, help="rows to generate")
    parser.add_argument(
        "--sweep",
        default="0,0.5,1.0,2.0",
        help="comma-separated bias_strength values to generate",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    source = F.load_dataset(io.download_ibm_hr())
    print(f"[source] {len(source)} rows bootstrapped from {io.IBM_HR_PATH.name}")

    strengths = [float(s) for s in args.sweep.split(",")]
    written = []
    for strength in strengths:
        df = generate(n=args.n, bias_strength=strength, seed=args.seed, source=source)
        name = f"synthetic_employees_b{strength:g}.csv"
        path = io.PROCESSED_DIR / name
        df.to_csv(path, index=False)
        rate = (df["Attrition"] == "Yes").mean()
        print(
            f"[gen] bias_strength={strength:<4g} n={len(df):<6d} "
            f"attrition_rate={rate:.3f}  -> {path.name}"
        )
        written.append({"bias_strength": strength, "path": name, "attrition_rate": float(rate)})

    io.write_json(
        {"n": args.n, "seed": args.seed, "files": written, "bias_group": BIAS_GROUP},
        io.ARTIFACTS_DIR / "synthetic_manifest.json",
    )
    print(f"[done] {len(written)} synthetic datasets written to {io.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
