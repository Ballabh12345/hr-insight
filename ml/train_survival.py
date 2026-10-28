"""Survival analysis - time to attrition.

Uses ``YearsAtCompany`` as the duration and ``Attrition`` as the event, fitting a
Cox proportional hazards model. Reports the concordance index on a held-out test
set and the largest hazard ratios.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import features as F
from ml.common import io, tracking

RANDOM_STATE = 42


def main() -> None:
    started = time.time()
    io.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Survival analysis - Cox proportional hazards")
    print("=" * 72)

    try:
        from lifelines import CoxPHFitter
    except Exception as exc:
        print(f"lifelines not available: {exc}")
        return

    raw = F.load_dataset(io.download_ibm_hr())
    cleaned = F.clean(raw)
    cleaned["event"] = (cleaned["Attrition"] == "Yes").astype(int)
    # Avoid zero durations.
    cleaned["duration"] = cleaned["YearsAtCompany"].clip(lower=0.5)

    # Feature frame: drop label-derived / id columns.
    drop_cols = ["Attrition", "AgeBand", "event", "duration"]
    X = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])
    X = pd.get_dummies(X, drop_first=True)

    from sklearn.model_selection import train_test_split

    X_train, X_test, dur_train, dur_test, ev_train, ev_test = train_test_split(
        X, cleaned["duration"], cleaned["event"], test_size=0.20, stratify=cleaned["event"], random_state=RANDOM_STATE
    )

    train_df = X_train.copy()
    train_df["duration"] = dur_train.values
    train_df["event"] = ev_train.values
    test_df = X_test.copy()
    test_df["duration"] = dur_test.values
    test_df["event"] = ev_test.values

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col="duration", event_col="event")

    train_ci = float(cph.concordance_index_)
    try:
        test_ci = float(cph.score(test_df, scoring_method="concordance_index"))
    except Exception:
        test_ci = float("nan")

    # Top hazard ratios.
    summary = cph.summary
    hr = np.exp(summary["coef"])
    top = hr.sort_values(ascending=False).head(10)
    hazard_ratios = {str(k): float(v) for k, v in top.items()}

    bundle = {
        "model": "CoxPH",
        "penalizer": 0.1,
        "train_concordance": train_ci,
        "test_concordance": test_ci,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "event_rate": float(cleaned["event"].mean()),
        "top_hazard_ratios": hazard_ratios,
        "training_seconds": round(time.time() - started, 2),
    }
    io.write_json(bundle, io.ARTIFACTS_DIR / "survival_metrics.json")
    print(f"[save] {io.ARTIFACTS_DIR / 'survival_metrics.json'}")
    print(f"[done] test concordance={test_ci:.4f}  {time.time() - started:.1f}s")

    tracking.log_run(
        "hr-insight-survival",
        "coxph",
        params={"penalizer": 0.1},
        metrics={"train_concordance": train_ci, "test_concordance": test_ci},
        artifacts=[io.ARTIFACTS_DIR / "survival_metrics.json"],
    )


if __name__ == "__main__":
    main()