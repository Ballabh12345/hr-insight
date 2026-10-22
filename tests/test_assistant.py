"""The assistant's internal router - deterministic, offline, and testable."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def context(employees_df):
    from app.services import assistant_service as assistant
    from app.services import data_service as data, model_service

    return (
        employees_df,
        data.load_summary(),
        model_service.load_fairness_report(),
    )


def _ask(question, context):
    from app.services import assistant_service as assistant

    employees, summary, fairness = context
    return assistant.answer_internal(question, employees, summary, fairness)


@pytest.mark.parametrize(
    "question,expected_fragment",
    [
        ("How many employees are high risk?", "High"),
        ("count of low risk employees", "Low"),
        ("how many high risk employees are in Sales?", "Sales"),
        ("what is our attrition rate?", "attrition rate"),
        ("which talent segment has the highest attrition risk?", "Talent segments"),
        ("what is the fairness gap on age?", "fairness audit"),
        ("how accurate is the attrition model?", "ROC-AUC"),
        ("who has the highest risk?", "highest predicted attrition risks"),
        ("what retention actions are recommended?", "Recommended retention actions"),
        ("help", "I can answer questions"),
    ],
)
def test_internal_router_answers_known_intents(question, expected_fragment, context):
    answer = _ask(question, context)
    assert answer is not None, f"no internal answer for: {question}"
    assert expected_fragment.lower() in answer.lower()


def test_router_returns_none_for_a_labour_market_question(context):
    """A question about the outside world must fall through to the live route."""
    assert _ask("what is the average salary for a data scientist in india in 2026?", context) is None


def test_router_answers_are_stable(context):
    first = _ask("how many employees are high risk?", context)
    second = _ask("how many employees are high risk?", context)
    assert first == second


def test_router_handles_empty_question(context):
    assert _ask("   ", context) is None


def test_department_matching_handles_abbreviations():
    from app.services import assistant_service as assistant

    assert assistant._match_department("how many in r&d") == "Research & Development"
    assert assistant._match_department("how many in research and development") == "Research & Development"
    assert assistant._match_department("sales team headcount") == "Sales"
    assert assistant._match_department("human resources staff") == "Human Resources"
    assert assistant._match_department("marketing") is None


def test_risk_band_matching():
    from app.services import assistant_service as assistant

    assert assistant._match_risk_band("high risk employees") == "High"
    assert assistant._match_risk_band("risk medium cases") == "Medium"
    assert assistant._match_risk_band("low-risk staff") == "Low"
    assert assistant._match_risk_band("everyone") is None


def test_cache_round_trip(tmp_path, monkeypatch):
    from app.services import assistant_service as assistant

    monkeypatch.setattr(assistant, "CACHE_DB", tmp_path / "cache.db")
    assistant._cache_put("what is x?", "the answer", ["https://example.com"])
    hit = assistant._cache_get("What is X?")
    assert hit is not None
    assert hit["answer"] == "the answer"
    assert hit["sources"] == ["https://example.com"]
    assert hit["cached"] is True


def test_cache_miss_returns_none(tmp_path, monkeypatch):
    from app.services import assistant_service as assistant

    monkeypatch.setattr(assistant, "CACHE_DB", tmp_path / "cache.db")
    assert assistant._cache_get("never asked") is None


def test_live_answer_cost_constant_is_positive():
    from app.services import assistant_service as assistant

    assert assistant.LIVE_ANSWER_COST_USD > 0
