"""The HR Insight Assistant.

Two routes, chosen by whether the question is about *this* workforce or the wider world:

**Internal route (free, instant, deterministic).** A rule-based intent router matches the
question against a set of known intents and answers it directly from the loaded dataset,
the training report, and the fairness report. No LLM is involved, so the answer is
reproducible, auditable, and costs nothing. It is also unit-testable - see
``tests/test_assistant.py``.

**External route (TinyFish).** For anything that needs the live web, TinyFish Search and
Fetch gather sources for free, and a TinyFish Agent run synthesises the answer. The agent
includes its own LLM inference, so no separate model provider key is needed. Each run
costs roughly $0.08 (about five agent steps at $0.016 each), so every answer is cached
in SQLite and repeated questions cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DB = ROOT / "data" / "db" / "assistant_cache.db"

#: Cost of one live-web answer, in USD, for display in the UI.
LIVE_ANSWER_COST_USD = 0.08

#: Example questions shown in the UI.
EXAMPLE_QUESTIONS = [
    "How many employees are high risk?",
    "Which talent segment has the highest attrition risk?",
    "How many high risk employees are in Sales?",
    "What is our attrition rate?",
    "What is the fairness gap on age?",
    "How accurate is the attrition model?",
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

    def help(q: str) -> str | None:
        if not _contains(q, "help", "what can you", "examples"):
            return None
        return (
            "I can answer questions about **this workforce** instantly and for free "
            "(counts by risk band, department or talent segment, the attrition rate, the "
            "fairness audit, model accuracy, recommended actions). For anything about the "
            "wider labour market I search the live web, which uses a small amount of "
            "TinyFish wallet credit."
        )

    return [
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
    """Answer a question about this workforce, or return ``None`` if it is not internal."""
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
    """Raised when the TinyFish API key is missing or the API call fails."""


def search_sources(question: str, limit: int = 4) -> list[dict[str, str]]:
    """Free web search via TinyFish. Returns ranked ``{title, url, snippet}`` dicts."""
    try:
        from tinyfish import TinyFish
    except ImportError as exc:  # pragma: no cover
        raise TinyFishUnavailable(
            "The tinyfish package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    if not os.environ.get("TINYFISH_API_KEY"):
        raise TinyFishUnavailable(
            "TINYFISH_API_KEY is not set, so live web answers are unavailable."
        )

    try:
        client = TinyFish()
        response = client.search.query(question)
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


def answer_external(question: str, use_agent: bool = True) -> dict[str, Any]:
    """Answer a live-web question using TinyFish.

    Search and Fetch are free. The Agent run (which carries the LLM inference) costs
    roughly $0.08. Set ``use_agent=False`` to gather sources without spending credit -
    the UI then shows the raw source list instead of a synthesised answer.
    """
    cached = _cache_get(question)
    if cached:
        return cached

    sources = search_sources(question)
    if not sources:
        return {
            "answer": "No sources were found for that question.",
            "sources": [],
            "cached": False,
            "cost_usd": 0.0,
        }

    if not use_agent:
        lines = [
            f"- [{s['title'] or s['url']}]({s['url']}){(' - ' + s['snippet']) if s['snippet'] else ''}"
            for s in sources
        ]
        return {
            "answer": "Sources found (live synthesis disabled to conserve wallet credit):\n\n"
            + "\n".join(lines),
            "sources": [s["url"] for s in sources],
            "cached": False,
            "cost_usd": 0.0,
        }

    try:
        from tinyfish import TinyFish
    except ImportError as exc:  # pragma: no cover
        raise TinyFishUnavailable("The tinyfish package is not installed.") from exc

    other_urls = [s["url"] for s in sources[1:]]
    goal = (
        "Answer the user's question accurately using web sources. "
        f"Question: {question}. "
        f"Other candidate sources: {', '.join(other_urls) if other_urls else 'none'}. "
        "If this page does not contain the answer, navigate to the other sources. "
        "Keep the answer concise and factual."
    )

    try:
        client = TinyFish()
        run = client.agent.run(
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
    except Exception as exc:  # pragma: no cover - network dependent
        raise TinyFishUnavailable(f"Agent run failed: {exc}") from exc

    result = getattr(run, "result", None)
    answer = ""
    used_sources = [s["url"] for s in sources]
    if isinstance(result, dict):
        answer = str(result.get("answer", "") or "")
        used_sources = result.get("sources") or used_sources
    elif result:
        answer = str(result)

    if not answer.strip():
        lines = [
            f"- [{s['title'] or s['url']}]({s['url']})" for s in sources
        ]
        answer = "The agent run did not return a synthesised answer. Sources:\n\n" + "\n".join(lines)

    _cache_put(question, answer, used_sources)
    return {
        "answer": answer,
        "sources": used_sources,
        "cached": False,
        "cost_usd": LIVE_ANSWER_COST_USD,
    }
