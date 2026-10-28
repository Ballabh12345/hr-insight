"""The assistant's internal router - deterministic, offline, and testable."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def context(employees_df):
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


@pytest.mark.parametrize(
    "question,expected_fragment",
    [
        ("well done", "thank you"),
        ("thanks!", "thank you"),
        ("ok bye", "goodbye"),
        ("hmm", "anything else"),
        ("perfect", "thank you"),
    ],
)
def test_smalltalk_is_answered_locally(question, expected_fragment, context):
    """Compliments/acknowledgements must never fall through to the paid web route."""
    answer = _ask(question, context)
    assert answer is not None, f"no local answer for smalltalk: {question}"
    assert expected_fragment.lower() in answer.lower()


@pytest.mark.parametrize(
    "question,expected_fragment",
    [
        ("how will we use it in real life?", "resignation"),
        ("what is the practical use of this app?", "retention plan"),
        ("how would a manager use this in practice?", "decision"),
    ],
)
def test_real_life_use_answers_locally(question, expected_fragment, context):
    answer = _ask(question, context)
    assert answer is not None, f"no internal answer for: {question}"
    assert expected_fragment.lower() in answer.lower()


def test_router_returns_none_for_a_labour_market_question(context):
    """A question about the outside world must fall through to the live route."""
    assert _ask("what is the average salary for a data scientist in india in 2026?", context) is None


@pytest.mark.parametrize(
    "question,expected_fragment",
    [
        ("what is this?", "HR Insight"),
        ("what is this project about?", "attrition"),
        ("what is its use?", "Flag who is at risk"),
        ("how does it work?", "pipeline"),
        ("what pages does the app have?", "Fairness audit"),
        ("what model is used?", "ROC-AUC"),
        ("what dataset is used?", "IBM HR Analytics"),
        ("what is the nine-box?", "talent"),
        ("what is SHAP?", "log-odds"),
        ("what is the decision log?", "SQLite"),
        ("what is the decision threshold?", "0.384"),
        ("what are the project results?", "ROC-AUC"),
        ("what are the limitations?", "Imbalanced"),
        ("who built this?", "CCSML0301"),
        ("how do I run it?", "streamlit run app/main.py"),
        ("what tech stack is used?", "Streamlit"),
        ("what is TinyFish?", "assistant"),
        ("who are you?", "assistant"),
        ("is my data safe?", "local"),
        ("what is attrition?", "leaving"),
        ("what is SDG 8?", "SDG 8"),
        ("what features does the model use?", "IBM-HR"),
    ],
)
def test_project_knowledge_intents(question, expected_fragment, context):
    """Meta-questions about the app itself are answered locally, never via the web."""
    answer = _ask(question, context)
    assert answer is not None, f"no internal answer for: {question}"
    assert expected_fragment.lower() in answer.lower()


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


def test_clean_live_answer_strips_agent_junk():
    from app.services import assistant_service as assistant

    assert assistant._clean_live_answer(">>> **hello**") == "**hello**"
    assert assistant._clean_live_answer("```python\nhi\n```") == "hi"
    assert assistant._clean_live_answer("  plain answer.  ") == "plain answer."


def test_pick_relevant_passages_scores_and_orders():
    from app.services import assistant_service as assistant

    text = (
        "Rental prices differ by city. "
        "The average rent for a 2BHK in Bangalore is about 42,000 rupees per month. "
        "Navigation menu contents that mean nothing. "
        "Bangalore rents also jumped this year for smaller flats, according to brokers."
    )
    picked = assistant._pick_relevant_passages("average 2BHK rent in bangalore", text)
    assert len(picked) == 2
    assert "42,000" in picked[0]  # highest keyword overlap first
    assert "Navigation menu" not in " ".join(picked)


def test_answer_external_falls_back_to_free_excerpts(tmp_path, monkeypatch):
    from app.services import assistant_service as assistant

    monkeypatch.setattr(assistant, "CACHE_DB", tmp_path / "cache.db")
    sources = [{"title": "A page", "url": "https://example.com/a", "snippet": "s"}]
    monkeypatch.setattr(assistant, "search_sources", lambda question, limit=4: sources)
    monkeypatch.setattr(assistant, "_agent_answer", lambda question, sources: None)
    monkeypatch.setattr(
        assistant, "_excerpt_answer", lambda question, sources, n_urls=3, note=None: "**[A page](https://example.com/a)**\n\n> the answer lives here"
    )

    result = assistant.answer_external("some live question", use_agent=True)
    assert result["cost_usd"] == 0.0
    assert "the answer lives here" in result["answer"]
    assert result["sources"] == ["https://example.com/a"]


def test_answer_external_agent_failure_returns_sources(tmp_path, monkeypatch):
    from app.services import assistant_service as assistant

    monkeypatch.setattr(assistant, "CACHE_DB", tmp_path / "cache.db")
    sources = [{"title": "T", "url": "https://example.com/x", "snippet": "snippet text"}]
    monkeypatch.setattr(assistant, "search_sources", lambda question, limit=4: sources)
    monkeypatch.setattr(assistant, "_agent_answer", lambda question, sources: None)
    monkeypatch.setattr(assistant, "_excerpt_answer", lambda question, sources, n_urls=3, note=None: None)

    result = assistant.answer_external("another live question")
    assert "example.com/x" in result["answer"]
    assert result["cost_usd"] == 0.0


def test_answer_external_caches_paid_answer(tmp_path, monkeypatch):
    from app.services import assistant_service as assistant

    monkeypatch.setattr(assistant, "CACHE_DB", tmp_path / "cache.db")
    sources = [{"title": "T", "url": "https://example.com/c", "snippet": ""}]
    monkeypatch.setattr(assistant, "search_sources", lambda question, limit=4: sources)
    monkeypatch.setattr(assistant, "_agent_answer", lambda question, sources: "synthesised answer")

    first = assistant.answer_external("cached question?")
    assert first["cost_usd"] == assistant.LIVE_ANSWER_COST_USD
    monkeypatch.setattr(assistant, "_agent_answer", lambda question, sources: "different answer")
    second = assistant.answer_external("cached question?")
    assert second["cached"] is True
    assert second["answer"] == "synthesised answer"
    assert "cost_usd" in second
