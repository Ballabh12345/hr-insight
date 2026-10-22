"""Human-in-the-loop decision log backed by SQLite.

Every recommendation an HR manager accepts or overrides is recorded with the model
version, the probability at the time, and - for overrides - the stated reason. This is
what turns a prediction dashboard into a decision-support system: the human decision is
the artefact of record, and it is auditable.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "hr_insight.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_number     INTEGER NOT NULL,
    job_role            TEXT,
    department          TEXT,
    risk_probability    REAL NOT NULL,
    risk_band           TEXT NOT NULL,
    recommended_action  TEXT NOT NULL,
    decision            TEXT NOT NULL CHECK (decision IN ('accepted','overridden')),
    override_reason     TEXT,
    model_version       TEXT NOT NULL,
    decided_by          TEXT NOT NULL DEFAULT 'hr_manager',
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decisions_employee ON decisions(employee_number);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def model_version() -> str:
    """A stable identifier for the current model build."""
    import json

    path = ROOT / "ml" / "artifacts" / "attrition_metrics.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            f"{data.get('selected_model', 'unknown')}"
            f"-{data.get('calibration_method', 'none')}"
            f"-{data.get('random_state', 0)}"
        )
    return "unknown"


def record_decision(
    employee_number: int,
    job_role: str,
    department: str,
    risk_probability: float,
    risk_band: str,
    recommended_action: str,
    decision: str,
    override_reason: str = "",
    decided_by: str = "hr_manager",
) -> None:
    """Append one accept/override decision to the log."""
    if decision not in ("accepted", "overridden"):
        raise ValueError("decision must be 'accepted' or 'overridden'")
    if decision == "overridden" and not override_reason.strip():
        raise ValueError("an override requires a reason")

    conn = _connect()
    with conn:
        conn.execute(
            """
            INSERT INTO decisions
                (employee_number, job_role, department, risk_probability, risk_band,
                 recommended_action, decision, override_reason, model_version,
                 decided_by, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(employee_number),
                job_role,
                department,
                float(risk_probability),
                risk_band,
                recommended_action,
                decision,
                override_reason.strip() or None,
                model_version(),
                decided_by,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    conn.close()


def load_decisions() -> pd.DataFrame:
    """Return the full decision log, newest first."""
    if not DB_PATH.exists():
        return pd.DataFrame(
            columns=[
                "id", "employee_number", "job_role", "department", "risk_probability",
                "risk_band", "recommended_action", "decision", "override_reason",
                "model_version", "decided_by", "created_at",
            ]
        )
    conn = _connect()
    df = pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC", conn)
    conn.close()
    return df


def decision_counts() -> dict[str, int]:
    """Counts of accepted vs overridden decisions."""
    df = load_decisions()
    if df.empty:
        return {"accepted": 0, "overridden": 0}
    return df["decision"].value_counts().to_dict()
