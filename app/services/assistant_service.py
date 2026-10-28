"""The HR Insight Assistant.

Two routes, chosen by whether the question is about *this* workforce or the wider world:

**Internal route (free, instant, deterministic).** A rule-based intent router matches the
question against a set of known intents and answers it directly from a built-in
project-knowledge base ("what is this?", "what is its use?", "how does it work?"),
the loaded dataset, the training report, and the fairness report. No LLM is involved, so the answer is
reproducible, auditable, and costs nothing. It is also unit-testable - see
``tests/test_assistant.py``.

**External route (TinyFish).** For anything that needs the live web, TinyFish Search and
Fetch gather sources for free, and a TinyFish Agent run synthesises the answer. The agent
includes its own LLM inference, so no separate model provider key is needed. Each run
costs roughly $0.08 (about five agent steps at $0.016 each), and every answer is cached
in SQLite, so repeated questions cost nothing. If the agent run fails, a free Fetch pass
still excerpts the most relevant passages from the top pages - the live route never
silently fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DB = ROOT / "data" / "db" / "assistant_cache.db"


def _load_env() -> None:
    """Load project-local environment variables from ``.env`` if present."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


_load_env()

#: Cost of one live-web answer, in USD, for display in the UI.
LIVE_ANSWER_COST_USD = 0.08

#: Example questions shown in the UI.
EXAMPLE_QUESTIONS = [
    "How many employees are high risk?",
    "What is this project?",
    "What is its use?",
    "How does it work?",
    "Which talent segment has the highest attrition risk?",
    "How many high risk employees are in Sales?",
    "What is our attrition rate?",
    "What is the fairness gap on age?",
    "How accurate is the attrition model?",
    "What pages does the app have?",
    "What is the average salary for a Data Scientist in India in 2026?",
]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    question_hash TEXT PRIMARY KEY,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    sources       TEXT,
    created_at    TEXT NOT NULL
);
"""


def _cache_key(question: str) -> str:
    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()


def _cache_get(question: str) -> dict[str, Any] | None:
    if not CACHE_DB.exists():
        return None
    conn = sqlite3.connect(CACHE_DB)
    try:
        row = conn.execute(
            "SELECT answer, sources, created_at FROM answer_cache WHERE question_hash = ?",
            (_cache_key(question),),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "answer": row[0],
        "sources": json.loads(row[1]) if row[1] else [],
        "created_at": row[2],
        "cached": True,
    }


def _cache_put(question: str, answer: str, sources: list[str]) -> None:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute(_CACHE_SCHEMA)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO answer_cache "
                "(question_hash, question, answer, sources, created_at) VALUES (?,?,?,?,?)",
                (
                    _cache_key(question),
                    question,
                    answer,
                    json.dumps(sources),
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
    finally:
        conn.close()


def cache_stats() -> dict[str, Any]:
    """How many live answers are cached, and the credits they saved."""
    if not CACHE_DB.exists():
        return {"cached_answers": 0, "credits_saved_usd": 0.0}
    conn = sqlite3.connect(CACHE_DB)
    try:
        count = conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]
    except sqlite3.Error:
        count = 0
    finally:
        conn.close()
    return {"cached_answers": count, "credits_saved_usd": count * LIVE_ANSWER_COST_USD}


# ---------------------------------------------------------------------------
# Internal route - deterministic intent router
# ---------------------------------------------------------------------------

DEPARTMENTS = ["Research & Development", "Sales", "Human Resources"]
RISK_BANDS = ["high", "medium", "low"]


def _contains(text: str, *options: str) -> bool:
    return any(o in text for o in options)


def _match_department(text: str) -> str | None:
    mapping = {
        "research & development": "Research & Development",
        "research and development": "Research & Development",
        "r&d": "Research & Development",
        "rnd": "Research & Development",
        "sales": "Sales",
        "human resources": "Human Resources",
        "hr": "Human Resources",
    }
    for needle, value in mapping.items():
        if needle in text:
            return value
    return None


def _match_risk_band(text: str) -> str | None:
    for band in RISK_BANDS:
        if f"{band} risk" in text or f"{band}-risk" in text or f"risk {band}" in text:
            return band.capitalize()
    return None


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _read_artifact(name: str) -> dict[str, Any]:
    """Read one artifact JSON file; return ``{}`` when it is missing or unreadable."""
    path = ROOT / "ml" / "artifacts" / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _project_context() -> dict[str, Any]:
    """Cached artifact payloads behind the project-knowledge answers."""
    return {
        "metrics": _read_artifact("attrition_metrics.json"),
        "ninebox": _read_artifact("ninebox_metrics.json"),
        "survival": _read_artifact("survival_metrics.json"),
        "shap": _read_artifact("shap_global.json"),
    }


def _looks_like_workforce_question(q: str) -> bool:
    """True when the question is about people in *this* dataset, not the project."""
    return _contains(
        q,
        "how many",
        "count",
        "number of",
        "headcount",
        "who is at risk",
        "who should",
        "top risk",
        "riskiest",
    )


def build_project_handlers(
    summary: dict[str, Any],
    fairness: dict[str, Any],
) -> list[tuple[str, Callable[[str], str | None], list[str]]]:
    """Project-knowledge intents - answer questions *about HR Insight itself*.

    These run before the workforce handlers so that meta-questions ("what is
    this?", "what is its use?", "how does it work?") are answered locally and
    never fall through to the live web.
    """
    art = _project_context()
    metrics = art["metrics"]
    ninebox_m = art["ninebox"]
    survival_m = art["survival"]
    shap_importance = art["shap"].get("importance", [])

    model = summary.get("model", {})
    test = model.get("test_metrics", {})
    selected = str(model.get("selected", "logistic_regression")).replace("_", " ")
    conformal = metrics.get("conformal", {})
    candidates = [str(k).replace("_", " ") for k in (metrics.get("candidates") or {})]
    n_employees = int(summary.get("n_employees", 0) or 0)
    attr_rate = float(summary.get("attrition_rate", 0.0) or 0.0)
    groups = fairness.get("baseline_audit", {}).get("groups", {})
    age_eo = groups.get("AgeBand", {}).get("summary", {}).get("equal_opportunity_difference")

    def who_are_you(q: str) -> str | None:
        if not _contains(
            q,
            "who are you",
            "what are you",
            "who made you",
            "how do you work",
            "how do you answer",
            "what are you trained",
            "tinyfish",
            "what is the assistant",
            "what is tinyfish",
        ):
            return None
        return (
            "I'm the **HR Insight assistant**. I answer in two ways: **locally and free** "
            "from this project's knowledge base (what the app is, its models, pages, "
            "fairness audit) and the live workforce data; or via **TinyFish** live web "
            f"search for labour-market questions, at about ${LIVE_ANSWER_COST_USD:.2f} per "
            "new answer (cached in SQLite, so repeat questions cost nothing)."
        )

    def privacy(q: str) -> str | None:
        if not _contains(
            q,
            "privacy",
            "is my data safe",
            "is the data safe",
            "confidential",
            "gdpr",
            "data leave",
            "data stay",
            "sent anywhere",
            "uploaded",
        ):
            return None
        return (
            "**Local by default.** Employee data never leaves this machine - it lives in "
            "local CSVs and SQLite databases (`data/db/`). Only a *question* typed here that "
            "needs the live web is sent to TinyFish; its answers are cached locally so "
            "repeating a question costs nothing. The TinyFish key lives in `.env`."
        )

    def who_built(q: str) -> str | None:
        if not _contains(q, "who built", "who made", "who created", "who developed", "who wrote", "whose project"):
            return None
        return (
            "**HR Insight** is a capstone project for **Machine Learning (CCSML0301) - "
            "BTech AI, Semester 3**, aligned to **UN SDG 8 - Decent Work and Economic "
            "Growth**. It demonstrates an end-to-end ML system: data, model, fairness, "
            "explainability, and human-in-the-loop decisions."
        )

    def what_is_this(q: str) -> str | None:
        if not _contains(
            q,
            "what is this",
            "what's this",
            "whats this",
            "what is hr insight",
            "what is the project",
            "about the project",
            "about this project",
            "about this app",
            "about the app",
            "describe the project",
            "describe this project",
            "explain this project",
            "what does this project do",
            "what does this app do",
            "what does this do",
            "introduce",
            "what have you built",
            "what did you build",
        ):
            return None
        return (
            "This is **HR Insight - a smart HR decision-support platform**. In one line: it "
            "estimates which employees are likely to resign, explains *why*, audits itself "
            f"for fairness, and records what HR decided. Built on {n_employees:,} employee "
            "records, it provides:\n\n"
            f"- **Attrition risk** for every employee (calibrated {selected}, cost-optimal "
            "alert threshold, High/Medium/Low bands)\n"
            "- **Nine-box talent matrix** (K-Means + tertiles) with retention actions\n"
            "- **SHAP reason codes** for every prediction, plus what-if overrides\n"
            "- **Fairness audit** with mitigation comparison and a bias-detector power curve\n"
            "- **Human-in-the-loop decision log** in SQLite (accept/override + reason)\n"
            "- **This assistant** - project Q&A locally, labour-market questions via live web\n"
            "Delivered as a **Streamlit** app with seven pages."
        )

    def features_used(q: str) -> str | None:
        if not _contains(
            q,
            "what features",
            "which features",
            "feature importan",
            "top predictors",
            "top driver",
            "what drives",
            "what inputs",
            "what columns",
            "which columns",
            "feature list",
            "variables used",
            "most important",
            "does the model use",
        ):
            return None
        top = [
            f"- **{item.get('label') or item.get('feature')}** ({item.get('importance', 0):.2f})"
            for item in shap_importance[:6]
        ]
        lines = "\n".join(top) if top else "- (run `python ml/train_all.py` to regenerate global SHAP)"
        return (
            "Inputs are the cleaned IBM-HR columns (demographics, job, compensation, "
            "satisfaction, tenure) plus engineered features such as `TenureRatio` and "
            "`AgeBand`. The top global drivers (mean SHAP) are:\n\n"
            f"{lines}\n\n"
            "Per-employee reason codes are on the **Employee detail** page."
        )

    def model_overview(q: str) -> str | None:
        if _contains(q, "accuracy", "accurate", "auc", "f1", "how good", "performance"):
            return None  # "how accurate..." belongs to the model-accuracy handler
        if not _contains(
            q,
            "what model",
            "which model",
            "what algorithm",
            "which algorithm",
            "what machine learning",
            "what ml",
            "which ml",
            "classifier",
            "how does the model work",
            "what are the models",
        ):
            return None
        return (
            f"**Model:** `{selected}`, wrapped in **CalibratedClassifierCV (sigmoid)** so the "
            "probabilities are trustworthy. It won a 3-fold-cross-validated ROC-AUC bake-off "
            f"against: {', '.join(candidates) or 'several tree-based candidates'}.\n\n"
            f"On the held-out test split: **ROC-AUC {test.get('roc_auc', 0):.3f}**, "
            f"**PR-AUC {test.get('pr_auc', 0):.3f}**, **F1 {test.get('f1', 0):.3f}** and "
            f"log-loss **{test.get('log_loss', 0):.3f}** (a constant predictor scores 0.439). "
            f"The decision threshold is **{model.get('decision_threshold', 0.5):.3f}** - "
            "chosen to minimize expected cost, not to maximize accuracy."
        )

    def dataset(q: str) -> str | None:
        if not _contains(
            q,
            "what dataset",
            "which dataset",
            "what is the dataset",
            "what data is used",
            "which data",
            "data source",
            "where does the data",
            "ibm",
            "how many rows",
            "how many columns",
            "data comes",
            "data used",
        ):
            return None
        return (
            f"The platform runs on the **IBM HR Analytics** employee-attrition dataset: "
            f"{n_employees:,} rows x 35 columns, of which **{round(attr_rate * n_employees)}** "
            f"employees ({attr_rate:.1%}) left. It is fetched automatically from a "
            "checksum-verified GitHub mirror (no Kaggle key needed), then cleaned and "
            "feature-engineered (`ml/common/features.py`) and split 80/20 stratified with "
            "seed 42."
        )

    def risk_bands_def(q: str) -> str | None:
        if not _contains(
            q,
            "what are risk bands",
            "what is a risk band",
            "risk band",
            "what are the bands",
            "how are bands",
            "what does high risk mean",
            "what does medium risk mean",
            "band thresholds",
            "band meaning",
            "band system",
        ):
            return None
        return (
            "**Risk bands** classify the calibrated probability into action levels: "
            "**High >= 0.55**, **Medium >= 0.25**, **Low < 0.25**. Right now: "
            f"{int(summary.get('high_risk_count', 0) or 0)} high, "
            f"{int(summary.get('medium_risk_count', 0) or 0)} medium, "
            f"{int(summary.get('low_risk_count', 0) or 0)} low "
            f"(of {n_employees:,}). Ask *how many employees are high risk?* for details."
        )

    def ninebox_def(q: str) -> str | None:
        if not _contains(
            q, "nine-box", "nine box", "9-box", "talent matrix", "talent grid", "what is the nine"
        ):
            return None
        k = ninebox_m.get("n_clusters", 6)
        sil = ninebox_m.get("selected_silhouette", 0)
        ari = ninebox_m.get("kmeans_stability", {}).get("ari_mean", 0)
        return (
            "The **nine-box** is the classic talent grid: performance x potential across nine "
            "cells (*Star Performer*, *Future Leader*, *Core Player*, *Needs Attention*, ...). "
            "In this project:\n\n"
            f"- **Potential** comes from K-Means clustering (k={k}, silhouette {sil:.3f}; it "
            "beat GMM and agglomerative clustering) mapped to Low/Medium/High tiers, stable "
            f"across 20 bootstrap resamples (mean ARI {ari:.3f})\n"
            "- **Performance** comes from tertiles of engagement and performance signals\n"
            f"- All 9 cells are populated across {n_employees:,} employees; mean risk per cell "
            "is on the **Overview** page, and each employee also gets a persona and a "
            "retention action"
        )

    def shap_def(q: str) -> str | None:
        if _contains(q, "shape"):
            return None  # avoid matching "shap" inside "shape"
        if not _contains(
            q, "shap", "reason codes", "how are predictions explained", "explainability",
            "explain the prediction", "why did the model",
        ):
            return None
        return (
            "**SHAP (SHapley Additive exPlanations)** credits each feature with its share of "
            "a prediction. A `LinearExplainer` runs on the calibrated model and returns "
            "per-employee **reason codes** in log-odds:\n\n"
            "- In red when it pushes risk up, green when it pulls risk down (see "
            "**Employee detail**)\n"
            "- The global mean-importance ranking is on **Model performance**\n"
            "- The same values power the what-if overrides: change a feature, watch the risk move"
        )

    def fairness_about(q: str) -> str | None:
        if _contains(q, "gap"):
            return None  # numeric gap questions belong to the fairness-gap handler
        if not _contains(
            q,
            "fairness audit",
            "what is fairness",
            "programmatic fairness",
            "disparate impact",
            "equal opportunity",
            "equalized odds",
            "fairlearn",
            "demographic parity",
        ):
            return None
        tail = (
            f"Current headline: age-band equal-opportunity gap **{age_eo:.3f}**."
            if age_eo is not None
            else "Ask *what is the fairness gap on age?* for the current numbers."
        )
        return (
            "**Fairness audit** - the model is checked for disparate impact across Gender, "
            "AgeBand, MaritalStatus, Department and the intersections Gender x AgeBand and "
            "Gender x Department.\n\n"
            "- fairlearn's **MetricFrame** computes selection-rate (demographic parity) and "
            "equal-opportunity gaps per group, with **bootstrap confidence intervals** and "
            "per-group calibration\n"
            "- A **mitigation study** compares reweighting and equalized-odds post-processing "
            "against the baseline\n"
            "- A **power curve** on synthetic high-bias cohorts shows how strong a signal the "
            "detector can catch\n"
            f"- {tail}"
        )

    def survival_def(q: str) -> str | None:
        if not _contains(
            q, "survival", "cox", "hazard", "time to attrition", "how long until",
            "how long will", "when will someone",
        ):
            return None
        hazards = list(survival_m.get("top_hazard_ratios", {}).items())[:5]
        lines = "\n".join(f"- **{k.replace('_', ' ')} x{v:.2f}**" for k, v in hazards)
        return (
            "**Survival analysis** answers *how long until a departure?*: a Cox proportional "
            f"hazards model on `YearsAtCompany` (event = attrition), held-out **concordance "
            f"{survival_m.get('test_concordance', 0):.3f}**. The strongest hazard "
            f"multipliers:\n\n{lines}"
        )

    def conformal_def(q: str) -> str | None:
        if not _contains(q, "conformal", "prediction set", "uncertainty", "how sure"):
            return None
        qhat = float(conformal.get("q_hat", 0.0) or 0.0)
        return (
            "**Conformal prediction** gives the risk probability an honest uncertainty band: "
            f"with alpha={conformal.get('alpha', 0.1):.2f} the split-conformal radius is "
            f"**q^={qhat:.3f}** (about +/-{qhat / 2:.3f}). Empirical coverage on the test "
            f"split is {conformal.get('coverage_test', 0):.3f} with mean prediction-set size "
            f"{conformal.get('mean_set_size_test', 0):.2f}. Risk numbers ship with a margin, "
            "not a naked decimal."
        )

    def decision_log_def(q: str) -> str | None:
        if not _contains(
            q, "decision log", "human-in-the-loop", "human in the loop", "override",
            "audit trail", "who decides", "accept", "accepted",
        ):
            return None
        return (
            "**Decision log** - the human-in-the-loop record. Every recommendation an HR "
            "manager **accepts or overrides** (on the Employee detail page) is written to "
            "an SQLite database (`data/db/hr_insight.db`) with the employee, risk probability "
            "and band, the "
            "recommended action, the model version and - for overrides - the stated reason. "
            "The human decision is the auditable artefact of record; browse it on the "
            "**Decision log** page."
        )

    def drift_def(q: str) -> str | None:
        if not _contains(q, "drift", "psi", "population stability"):
            return None
        return (
            "**Data drift check** - the Overview page computes the Population Stability Index "
            "(PSI) between the IBM reference data and a high-bias synthetic cohort for Age, "
            "MonthlyIncome, DistanceFromHome and YearsAtCompany. PSI > 0.2 flags meaningful "
            "drift, i.e. the live population no longer looks like the training population."
        )

    def what_is_attrition(q: str) -> str | None:
        if not _contains(
            q, "what is attrition", "what does attrition mean", "define attrition",
            "meaning of attrition", "what is churn", "what does churn mean", "define churn",
        ):
            return None
        return (
            f"**Attrition** = an employee leaving the organisation. In this dataset the "
            f"historic rate is **{attr_rate:.1%}** ({round(attr_rate * n_employees)} of "
            f"{n_employees:,} employees). The model predicts each person's probability of "
            "leaving; High/Medium/Low bands turn that probability into an action level, and "
            "a retention action is recommended per employee."
        )

    def threshold_about(q: str) -> str | None:
        if not _contains(
            q, "threshold", "cost-optimal", "cost optimal", "expected cost",
            "cost of attrition", "cost of a departure", "roi", "business value",
            "cost of turnover",
        ):
            return None
        t = float(model.get("decision_threshold", 0.5) or 0.5)
        return (
            f"**Cost-optimal threshold** - the alert threshold ({t:.3f}) is not the default "
            "0.5. It minimizes *expected cost*: (people flagged) x (cost of an action) + "
            "(true leavers missed) x (cost of a departure), adjusted by the probability an "
            "action actually succeeds. Tune the sliders on the **Overview** page to explore "
            "the break-even point."
        )

    def sdg_about(q: str) -> str | None:
        if not _contains(q, "sdg", "decent work", "sustainable development"):
            return None
        return (
            "The project is aligned with **UN SDG 8 - Decent Work and Economic Growth**: "
            "predict, explain and mitigate avoidable employee departures instead of reacting "
            "to them, and give HR information-based, fair, auditable tools for workforce "
            "decisions."
        )

    def tech_stack(q: str) -> str | None:
        if not _contains(
            q, "tech stack", "technology", "what framework", "what language", "streamlit",
            "libraries", "packages", "requirements", "what tools", "built with",
            "dependencies", "python", "database", "sqlite", "mlflow",
        ):
            return None
        return (
            "**Stack:**\n\n"
            "- **Language**: Python 3\n"
            "- **UI**: Streamlit + Plotly with a custom day/night theme (`app/theme.py`)\n"
            "- **ML**: scikit-learn (logit + calibration, K-Means, clustering), SHAP, "
            "lifelines (Cox), fairlearn for the audit\n"
            "- **Tracking**: MLflow (`mlruns/`, `mlflow.db`)\n"
            "- **Storage**: SQLite - decision log (`data/db/hr_insight.db`) and assistant cache\n"
            "- **Live web**: the TinyFish SDK (search, fetch, agent)\n"
            "- **Quality**: pytest (`tests/`, 58 tests) and ruff; layout: `app/`, `ml/`, "
            "`data/`, `docs/`, `notebooks/`, `ops/`"
        )

    def pages(q: str) -> str | None:
        if not _contains(
            q, "what pages", "which pages", "pages does", "how many pages", "screens",
            "sections", "tabs", "navigation", "what views", "features of the app",
            "what can i see", "user journey",
        ):
            return None
        return (
            "**HR Insight** has seven pages (sidebar navigation):\n\n"
            "1. **Overview** - workforce KPIs, risk by department, the talent matrix, the "
            "cost-optimal threshold simulator and a drift check\n"
            "2. **Employees** - search and filter the workforce table\n"
            "3. **Employee detail** - risk meter, SHAP reason codes, fairness context, "
            "what-if overrides, and the accept/override decision\n"
            "4. **Fairness audit** - disparate impact by group, mitigation comparison, "
            "bias-detector power curve\n"
            "5. **Model performance** - candidate comparison, ROC/PR/calibration curves, "
            "global SHAP, survival and conformal notes\n"
            "6. **Decision log** - every accepted/overridden recommendation with reason, "
            "model version and timestamp\n"
            "7. **Assistant** - this chat: project + workforce answers locally, live web "
            "via TinyFish"
        )

    def how_to_run(q: str) -> str | None:
        if not _contains(
            q, "how to run", "how do i run", "how do i start", "run the app", "start the app",
            "how do i install", "how to install", "setup", "how to retrain", "retrain",
            "train the model", "how long does training take", "how do i test",
        ):
            return None
        return (
            "To run it:\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "python ml/train_all.py   # trains and calibrates everything, about a minute\n"
            "streamlit run app/main.py\n"
            "```\n\n"
            "There are also `run.bat` / `retrain.bat` helpers on Windows, individual trainers "
            "(`ml/train_attrition.py`, `ml/train_ninebox.py`, `ml/train_fairness.py`, "
            "`ml/train_survival.py`), and `python -m pytest tests/` for the test suite."
        )

    def results_overview(q: str) -> str | None:
        if _looks_like_workforce_question(q):
            return None
        if not _contains(
            q, "results", "findings", "conclusion", "how well did", "outcome",
            "summary of", "headline",
        ):
            return None
        parts = []
        if test:
            parts.append(
                f"- **Attrition model** ({selected}): test ROC-AUC "
                f"{test.get('roc_auc', 0):.3f}, PR-AUC {test.get('pr_auc', 0):.3f}, "
                f"calibrated log-loss {test.get('log_loss', 0):.3f}"
            )
        if ninebox_m:
            parts.append(
                f"- **Nine-box**: all 9 cells populated; K-Means k={ninebox_m.get('n_clusters', 6)} "
                f"with silhouette {ninebox_m.get('selected_silhouette', 0):.3f} and cluster "
                f"stability ARI {ninebox_m.get('kmeans_stability', {}).get('ari_mean', 0):.3f}"
            )
        if age_eo is not None:
            parts.append(
                f"- **Fairness**: age-band equal-opportunity gap {age_eo:.3f} - quantified, "
                "with the mitigation study showing what helps"
            )
        if survival_m:
            parts.append(
                f"- **Survival**: Cox model, test concordance {survival_m.get('test_concordance', 0):.3f}; "
                "working overtime roughly doubles the hazard"
            )
        parts.append(
            f"- **Decisions**: threshold {model.get('decision_threshold', 0.5):.3f} minimizes "
            f"expected cost; {int(summary.get('high_risk_count', 0) or 0)} employees are flagged "
            "high risk today"
        )
        return "**Headline results:**\n\n" + "\n".join(parts)

    def limitations(q: str) -> str | None:
        if not _contains(
            q, "limitations", "limitation", "weakness", "caveat", "what can't",
            "what cant", "downside", "not perfect", "drawback", "honest assessment",
            "risks of the model",
        ):
            return None
        return (
            "**Honest limitations:**\n\n"
            f"- One company, {n_employees:,} rows: the model learns this population, not the world\n"
            f"- Imbalanced classes ({attr_rate:.1%} leavers): accuracy flatters; PR-AUC and "
            "log-loss are the honest metrics\n"
            "- Historic bias can hide in the labels - the fairness audit is a guardrail, not a fix\n"
            "- Survival uses `YearsAtCompany` from one snapshot (right-censoring approximated)\n"
            "- Live assistant answers depend on third-party web sources\n"
            "- Predictions support decisions; the retention decision itself must stay human"
        )

    def real_life_use(q: str) -> str | None:
        if not _contains(
            q, "real life", "real-life", "real world", "real-world", "practical use",
            "practically", "in practice", "how would we use", "how do we use",
            "how will we use", "how to use in", "use scenario", "business scenario",
            "daily use", "day to day", "day-to-day",
        ):
            return None
        return (
            "**A working day with HR Insight (real life):**\n\n"
            "- **Monday triage** - HR opens the Overview: "
            f"{int(summary.get('high_risk_count', 0) or 0)} high-risk employees are flagged "
            "across departments. The risk is visible *before* the resignation email arrives.\n"
            "- **Investigate** - the manager opens an employee's detail page: the risk meter, "
            "the SHAP reasons (e.g. *works overtime*, *long commute*), and the fairness "
            "context for that person.\n"
            "- **Act with a what-if** - \"if overtime were NO, risk drops from 71% to ~45%\" - "
            "that becomes the retention plan: rebalance workload, a 90-day check-in, a role "
            "move. Actions are ranked by expected cost saved.\n"
            "- **Decide with accountability** - the manager **accepts or overrides** with a "
            "reason; every decision is logged (SQLite) with probability at decision time and "
            "model version - so when leadership HR asks \"why did you keep/let go X?\", there "
            "is an auditable answer.\n"
            "- **Stay compliant** - monthly, the Fairness audit page shows whether the model "
            "disadvantages any demographic group; this is exactly the disparate-impact "
            "analysis an equal-opportunity review needs.\n"
            "- **Benchmark the market** - in hiring or comp reviews, ask the assistant live "
            "web questions (\"salary for a Data Scientist in India 2026?\").\n"
            "- **In production** - the same pipeline would rescore nightly from an HRIS feed "
            "and push alerts to owners, replacing ad-hoc spreadsheet analysis and gut "
            "decisions with evidence-based ones."
        )

    def purpose(q: str) -> str | None:
        if not _contains(
            q, "what is its use", "what is the use", "what is it used for", "what is it for",
            "why was it built", "why was this built", "what problem", "purpose",
            "why is it useful", "who is it for", "who uses", "benefit", "use case",
            "why should i use", "why do we need", "why does it exist", "why is this needed",
            "why build", "what's its use", "what's the use", "what's this used for",
            "what's it for", "used for",
        ):
            return None
        return (
            "**Its use:** HR normally discovers resignations on the day they happen. This "
            "platform moves that moment earlier:\n\n"
            f"- **Flag who is at risk** - calibrated probability + High/Medium/Low band for "
            f"all {n_employees:,} employees\n"
            "- **Explain why** - SHAP reason codes for every employee, plus what-if scenarios "
            "(e.g. move off overtime) to see the risk move\n"
            "- **Decide with a paper trail** - a recommended retention action, then the "
            "manager accepts or overrides, logged with version and reason\n"
            "- **Stay accountable** - the fairness audit keeps the model honest across "
            "gender, age, marital status and department"
        )

    def how_it_works(q: str) -> str | None:
        if not _contains(
            q, "how does it work", "how does this work", "how does it all work",
            "how was it built", "how is it built", "architecture", "pipeline",
            "how does the app work", "how does the system work",
            "how does the project work", "workflow", "steps involved",
        ):
            return None
        return (
            "The pipeline has five layers, rebuilt end-to-end by `python ml/train_all.py`:\n\n"
            "1. **Data** - fetch the IBM HR dataset (checksum mirror), clean and engineer "
            "features (`ml/common/features.py`), stratified 80/20 split (seed 42)\n"
            "2. **Model** - train + calibrate a logistic regression, benchmark candidates, "
            "pick a cost-optimal decision threshold (`ml/train_attrition.py`)\n"
            "3. **Talent and risk views** - score the whole workforce, K-Means nine-box and "
            "personas (`ml/train_ninebox.py`), Cox survival (`ml/train_survival.py`)\n"
            "4. **Accountability** - fairness audit with mitigation study and bias power "
            "curve (`ml/train_fairness.py`), SHAP explanations\n"
            "5. **App** - `streamlit run app/main.py` serves seven pages over those "
            "artifacts; SQLite holds the decision log and this assistant's answer cache"
        )

    return [
        ("project_who_are_you", who_are_you, ["who are you", "tinyfish"]),
        ("project_privacy", privacy, ["privacy", "data safe"]),
        ("project_who_built", who_built, ["who built", "who made"]),
        ("project_what_is_this", what_is_this, ["what is this", "about the project"]),
        ("project_features", features_used, ["features", "columns", "drivers"]),
        ("project_model", model_overview, ["what model", "which model", "algorithm"]),
        ("project_dataset", dataset, ["dataset", "ibm", "rows", "columns"]),
        ("project_risk_bands", risk_bands_def, ["risk band", "bands"]),
        ("project_ninebox", ninebox_def, ["nine-box", "talent matrix"]),
        ("project_shap", shap_def, ["shap", "reason codes"]),
        ("project_fairness", fairness_about, ["fairness audit", "disparate impact"]),
        ("project_survival", survival_def, ["survival", "cox", "hazard"]),
        ("project_conformal", conformal_def, ["conformal", "uncertainty"]),
        ("project_decision_log", decision_log_def, ["decision log", "override"]),
        ("project_drift", drift_def, ["drift", "psi"]),
        ("project_attrition_def", what_is_attrition, ["what is attrition", "churn"]),
        ("project_threshold", threshold_about, ["threshold", "cost", "roi"]),
        ("project_sdg", sdg_about, ["sdg", "decent work"]),
        ("project_tech_stack", tech_stack, ["tech stack", "framework", "streamlit", "libraries"]),
        ("project_pages", pages, ["pages", "screens", "sections"]),
        ("project_how_to_run", how_to_run, ["how to run", "retrain", "install"]),
        ("project_results", results_overview, ["results", "findings", "conclusion"]),
        ("project_limitations", limitations, ["limitations", "weakness", "caveat"]),
        ("project_purpose", purpose, ["use", "purpose", "why"]),
        ("project_real_life_use", real_life_use, ["real life", "real world", "practical"]),
        ("project_how_it_works", how_it_works, ["how does it work", "architecture", "pipeline"]),
    ]


def build_internal_handlers(
    employees: pd.DataFrame,
    summary: dict[str, Any],
    fairness: dict[str, Any],
) -> list[tuple[str, Callable[[str], str | None], list[str]]]:
    """Build the intent table. Each entry is ``(name, handler, trigger keywords)``."""

    def count_risk_band(q: str) -> str | None:
        band = _match_risk_band(q)
        if not band or not _contains(q, "how many", "count", "number of"):
            return None
        # A department in the question makes this a department question - defer.
        if _match_department(q):
            return None
        n = int((employees["risk_band"].astype(str) == band).sum())
        return f"There are **{n}** employees in the **{band}** risk band ({n / len(employees):.1%} of the workforce)."

    def count_by_department(q: str) -> str | None:
        dept = _match_department(q)
        if not dept or not _contains(q, "how many", "count", "number of", "headcount"):
            return None
        sub = employees[employees["Department"] == dept]
        band = _match_risk_band(q)
        if band:
            n = int((sub["risk_band"].astype(str) == band).sum())
            return f"There are **{n}** {band.lower()}-risk employees in **{dept}** (of {len(sub)} in the department)."
        return f"**{dept}** has **{len(sub)}** employees, with a mean attrition probability of **{_fmt_pct(sub['attrition_probability'].mean())}**."

    def attrition_rate(q: str) -> str | None:
        if not _contains(q, "attrition rate", "turnover rate", "churn"):
            return None
        return (
            f"The historical attrition rate in this dataset is **{_fmt_pct(summary.get('attrition_rate', 0))}** "
            f"(237 of 1470 employees). The model's mean predicted risk across the workforce is "
            f"**{_fmt_pct(summary.get('mean_probability', 0))}**."
        )

    def worst_segment(q: str) -> str | None:
        if not _contains(q, "segment", "nine-box", "nine box", "talent", "group with the highest", "highest risk group"):
            return None
        cell = summary.get("ninebox_risk", {})
        if not cell:
            return None
        ranked = sorted(cell.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        lines = [
            f"**{label}** - mean risk {_fmt_pct(vals['mean'])} ({int(vals['count'])} employees)"
            for label, vals in ranked[:5]
        ]
        return "Talent segments ranked by mean predicted attrition risk:\n\n" + "\n".join(
            f"- {line}" for line in lines
        )

    def fairness_gap(q: str) -> str | None:
        if not _contains(q, "fair", "bias", "disparate", "parity gap", "unfair"):
            return None
        groups = fairness.get("baseline_audit", {}).get("groups", {})
        if not groups:
            return None
        lines = []
        for attribute, block in groups.items():
            summary_block = block["summary"]
            lines.append(
                f"- **{attribute}**: demographic parity gap {summary_block['demographic_parity_difference']:.3f}, "
                f"equal opportunity gap {summary_block['equal_opportunity_difference']:.3f}"
            )
        return "Current fairness audit of the attrition model:\n\n" + "\n".join(lines)

    def model_accuracy(q: str) -> str | None:
        if not _contains(q, "accurate", "accuracy", "auc", "how good", "performance of the model", "f1"):
            return None
        test = summary.get("model", {}).get("test_metrics", {})
        if not test:
            return None
        return (
            f"The attrition model is a **{summary.get('model', {}).get('selected', 'model')}** with "
            f"sigmoid calibration. On the held-out test split it reaches "
            f"**ROC-AUC {test.get('roc_auc', 0):.3f}**, **PR-AUC {test.get('pr_auc', 0):.3f}**, "
            f"**F1 {test.get('f1', 0):.3f}** and a log-loss of **{test.get('log_loss', 0):.3f}** "
            f"(a constant base-rate predictor scores {0.439:.3f})."
        )

    def top_risk(q: str) -> str | None:
        if not _contains(q, "highest risk", "top risk", "riskiest", "who should", "who is at risk", "highest probability"):
            return None
        top = employees.nlargest(5, "attrition_probability")
        lines = [
            f"- **#{int(r.EmployeeNumber)}** - {r.JobRole}, {r.Department} - risk "
            f"**{_fmt_pct(r.attrition_probability)}** ({r.ninebox_label})"
            for r in top.itertuples()
        ]
        return "The five highest predicted attrition risks:\n\n" + "\n".join(lines)

    def recommended_actions(q: str) -> str | None:
        if not _contains(q, "action", "recommend", "what should", "retention plan"):
            return None
        counts = (
            employees[employees["risk_band"].astype(str) != "Low"]
            .groupby("retention_action")
            .size()
            .sort_values(ascending=False)
        )
        if counts.empty:
            return "No employees currently need a retention action."
        lines = [f"- **{count}** - {action}" for action, count in counts.items()]
        return "Recommended retention actions:\n\n" + "\n".join(lines)

    def band_only(q: str) -> str | None:
        """Answer a bare risk-band keyword (e.g. "high") with the band count."""
        cleaned = q.strip().strip("?").strip()
        if cleaned not in {"high", "low", "medium", "high risk", "low risk", "medium risk"}:
            return None
        label = cleaned.title()
        n = int((employees["risk_band"].astype(str) == label).sum())
        return f"There are **{n}** employees in the **{label}** risk band ({n / len(employees):.1%} of the workforce)."

    def greeting(q: str) -> str | None:
        """Answer a bare greeting locally, with a short project intro."""
        cleaned = q.strip().strip("!.?").strip()
        if cleaned.lower() not in {
            "hi",
            "hi there",
            "hello",
            "hey",
            "hey there",
            "yo",
            "sup",
            "good morning",
            "good evening",
            "good afternoon",
        }:
            return None
        return (
            "Hi! I'm the **HR Insight** assistant. Ask me about this workforce "
            "(attrition risk, the nine-box talent matrix, the fairness audit, or the "
            "model), for example *\"how many high risk employees are there?\"* or "
            "*\"what is our attrition rate?\"*. For anything about the wider labour "
            "market I can search the live web."
        )

    _SMALLTALK = {
        "praise": {
            "thanks", "thank you", "thank u", "thx", "ty", "thank you so much",
            "well done", "good job", "great job", "nice", "nice work", "nice one",
            "awesome", "perfect", "amazing", "excellent", "superb", "brilliant",
            "love it", "that's great", "thats great", "very good", "good",
        },
        "neutral": {
            "ok", "okay", "okk", "k", "kk", "done", "fine", "alright", "cool",
            "yeah", "yes ok", "hmm", "got it", "understood", "i see",
        },
        "farewell": {
            "bye", "goodbye", "see you", "see ya", "cya", "tata", "thats all",
            "that's all", "nothing", "byee",
        },
    }
    _SMALLTALK_REPLY = {
        "praise": "Thank you! Happy to help. Anything else - about this project "
        "(*what is its use?*), the workforce (*how many high risk employees?*), or "
        "the labour market (live web)?",
        "neutral": "Great. Anything else - a project question (*what is this?*), a "
        "workforce question (*who is at risk?*), or a live-web search?",
        "farewell": "Goodbye! Come back anytime - the dashboard keeps recomputing, "
        "and every decision you log stays auditable.",
    }

    def smalltalk(q: str) -> str | None:
        """Answer compliments, acknowledgements and farewells locally - never the web."""
        cleaned = q.strip().strip("!.?").strip().lower()
        bucket = None
        for name, options in _SMALLTALK.items():
            if cleaned in options:
                bucket = name
                break
        if bucket is None and any(t in cleaned for t in ("thank",)):
            bucket = "praise"
        if bucket is None and cleaned.endswith("bye"):
            bucket = "farewell"
        if bucket is None:
            return None
        return _SMALLTALK_REPLY[bucket]

    def help(q: str) -> str | None:
        if not _contains(q, "help", "what can you", "examples"):
            return None
        return (
            "I can answer questions about **this project** (what it is, what it's used for, "
            "how it works, its pages, models, fairness audit, limitations) and about **this "
            "workforce** (counts by risk band, department or talent segment, the attrition "
            "rate, model accuracy, recommended actions) - instantly and for free. For "
            "anything about the wider labour market I search the live web, which uses a "
            "small amount of TinyFish wallet credit."
        )

    project = build_project_handlers(summary, fairness)
    return project + [
        ("greeting", greeting, ["hi", "hello", "hey"]),
        ("smalltalk", smalltalk, ["thanks", "well done", "bye"]),
        ("band_only", band_only, ["high", "low", "medium", "risk"]),
        ("count_risk_band", count_risk_band, ["how many", "count", "risk band"]),
        ("count_by_department", count_by_department, ["department", "sales", "r&d", "headcount"]),
        ("attrition_rate", attrition_rate, ["attrition rate", "turnover"]),
        ("worst_segment", worst_segment, ["segment", "talent", "nine-box"]),
        ("fairness_gap", fairness_gap, ["fair", "bias", "parity"]),
        ("model_accuracy", model_accuracy, ["accurate", "auc", "f1"]),
        ("top_risk", top_risk, ["highest risk", "riskiest"]),
        ("recommended_actions", recommended_actions, ["action", "recommend"]),
        ("help", help, ["help", "what can you"]),
    ]


def answer_internal(
    question: str,
    employees: pd.DataFrame,
    summary: dict[str, Any],
    fairness: dict[str, Any],
) -> str | None:
    """Answer a question about this project or workforce, or return ``None`` if it is not internal."""
    q = question.strip().lower()
    if not q:
        return None
    for _name, handler, _keywords in build_internal_handlers(employees, summary, fairness):
        try:
            result = handler(q)
        except Exception:  # a broken handler must never break the chat
            result = None
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# External route - TinyFish
# ---------------------------------------------------------------------------


class TinyFishUnavailable(RuntimeError):
    """Raised when the TinyFish API key is missing or the search call fails."""


def _client():
    """Build a TinyFish client (a helper so tests can monkeypatch it)."""
    from tinyfish import TinyFish

    return TinyFish()


def search_sources(question: str, limit: int = 4) -> list[dict[str, str]]:
    """Free web search via TinyFish. Returns ranked ``{title, url, snippet}`` dicts."""
    if not os.environ.get("TINYFISH_API_KEY"):
        raise TinyFishUnavailable(
            "TINYFISH_API_KEY is not set, so live web answers are unavailable."
        )

    try:
        response = _client().search.query(question)
    except Exception as exc:  # pragma: no cover - network dependent
        raise TinyFishUnavailable(f"Search failed: {exc}") from exc

    results = []
    for item in getattr(response, "results", [])[:limit]:
        results.append(
            {
                "title": getattr(item, "title", "") or "",
                "url": getattr(item, "url", "") or "",
                "snippet": getattr(item, "snippet", "") or "",
            }
        )
    return results


def _clean_live_answer(text: str) -> str:
    """Strip stray blockquote/code-fence junk some agent runs emit."""
    cleaned = str(text).strip()
    cleaned = re.sub(r"^[\s>]+", "", cleaned)
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


#: Words ignored when scoring fetched passages against a question.
_STOPWORDS = frozenset(
    "a an and any are ask about can could does do for from has have how i in is it its me my "
    "of on or our so tell that the their there these they this to was were what when where "
    "which who why will with you your".split()
)

#: Split a fetched page into sentence-like segments: at terminal punctuation,
#: at newlines, and at markdown headings/links that would confuse reading.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _question_tokens(question: str) -> list[str]:
    """Meaning-bearing lowercase words from the question (drop stopwords)."""
    words = re.findall(r"[a-z0-9]{3,}", question.lower())
    return [w for w in words if w not in _STOPWORDS]


def _pick_relevant_passages(question: str, text: str, max_passages: int = 2) -> list[str]:
    """Pick the passages of ``text`` that best match the question, in reading order."""
    tokens = _question_tokens(question)
    if not tokens:
        return []
    scored: list[tuple[int, int, str]] = []
    for index, segment in enumerate(_SENTENCE_SPLIT.split(text)):
        passage = segment.strip()
        if not (40 <= len(passage) <= 400):
            continue
        low = passage.lower()
        score = sum(1 for token in tokens if token in low)
        if score > 0:
            # Earlier passages win on ties: negate the position for sorting.
            scored.append((score, -index, passage))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [passage for _score, _pos, passage in scored[:max_passages]]


def _agent_answer(question: str, sources: list[dict[str, str]]) -> str | None:
    """One TinyFish Agent run (paid). Returns a cleaned answer, or ``None`` on failure."""
    other_urls = [s["url"] for s in sources[1:]]
    goal = (
        "Answer the user's question accurately using web sources. "
        f"Question: {question}. "
        f"Other candidate sources: {', '.join(other_urls) if other_urls else 'none'}. "
        "If this page does not contain the answer, navigate to the other sources. "
        "Answer in the same language as the question, without markdown fences, and "
        "never start the answer with '>' characters. "
        "Keep the answer concise and factual."
    )

    try:
        run = _client().agent.run(
            url=sources[0]["url"],
            goal=goal,
            output_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer"],
            },
        )
    except Exception:  # pragma: no cover - network dependent
        return None

    if getattr(run, "error", None):
        return None
    result = getattr(run, "result", None)
    if isinstance(result, dict):
        answer = str(result.get("answer", "") or "")
    elif result:
        answer = str(result)
    else:
        return None
    answer = _clean_live_answer(answer)
    return answer or None


def _excerpt_answer(
    question: str,
    sources: list[dict[str, str]],
    n_urls: int = 3,
    note: str | None = None,
) -> str | None:
    """Free fallback - fetch the top pages and excerpt the passages matching the question.

    Uses plain page fetch (free on every account) and local keyword scoring, so it
    works even where the paid/optional ``highlights`` feature is not enabled.
    """
    try:
        response = _client().fetch.get_contents(
            [s["url"] for s in sources[:n_urls]],
            purpose="extract passages for answering the user's question",
        )
    except Exception:  # pragma: no cover - network dependent
        return None

    blocks = []
    for item in getattr(response, "results", []) or []:
        page_text = getattr(item, "text", None) or ""
        title = (getattr(item, "title", None) or "").strip()
        url = getattr(item, "url", None) or ""
        passages = _pick_relevant_passages(question, page_text)
        if not passages:
            continue
        quoted = "\n\n".join(f"> {passage}" for passage in passages)
        heading = f"[{title}]({url})" if title and url else (url or title or "source")
        blocks.append(f"**{heading}**\n\n{quoted}")
    if not blocks:
        return None
    prefix = note or "*The agent run was unavailable - the most relevant passages, fetched live and free:*"
    return prefix + "\n\n" + "\n\n---\n\n".join(blocks)


def _sources_only_answer(sources: list[dict[str, str]]) -> str:
    lines = [
        f"- [{s['title'] or s['url']}]({s['url']})"
        f"{(' - ' + s['snippet']) if s['snippet'] else ''}"
        for s in sources
    ]
    return (
        "*Live synthesis was unavailable - these are the sources I found:*\n\n"
        + "\n".join(lines)
    )


def answer_external(question: str, use_agent: bool = True) -> dict[str, Any]:
    """Answer a live-web question using TinyFish.

    Search (free) finds sources, then a TinyFish Agent run (paid, ~$0.08) navigates
    the pages and synthesises the answer. If the agent run fails, a free ``fetch``
    pass excerpts the top pages instead, so the internet route keeps working even
    when the agent is unavailable. Everything is cached in SQLite - repeating a
    question costs nothing.
    """
    cached = _cache_get(question)
    if cached:
        cached["answer"] = _clean_live_answer(cached["answer"])
        cached.setdefault("cost_usd", 0.0)
        return cached

    sources = search_sources(question)
    if not sources:
        return {
            "answer": "No sources were found for that question.",
            "sources": [],
            "cached": False,
            "cost_usd": 0.0,
        }

    used_sources = [s["url"] for s in sources]
    if use_agent:
        answer = _agent_answer(question, sources)
        paid = bool(answer)
    else:
        answer = None
        paid = False

    if not answer:
        note = (
            "*The agent run was unavailable - the most relevant passages, fetched live "
            "and free:*"
            if use_agent
            else "*Live synthesis is off - the most relevant passages from the web (free"
            " fetch):*"
        )
        answer = _excerpt_answer(question, sources, note=note) or _sources_only_answer(sources)

    _cache_put(question, answer, used_sources)
    return {
        "answer": answer,
        "sources": used_sources,
        "cached": False,
        "cost_usd": LIVE_ANSWER_COST_USD if paid else 0.0,
    }
