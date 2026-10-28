"""The HR Insights Assistant - internal router plus an optional live-web route.

Layout contract: the conversation reads top-to-bottom (examples card, history,
then the chat input pinned at the bottom of the app). The live-web toggle and
its cost note appear exactly once, next to the examples.
"""

from __future__ import annotations

import streamlit as st

from app import theme
from app.services import assistant_service as assistant
from app.services import data_service as data, model_service


def _run_prompt(prompt: str, live: bool) -> None:
    """Answer one question, render the exchange, and store both messages."""
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = assistant.answer_internal(prompt, data.load_employees(), data.load_summary(), model_service.load_fairness_report())
        used_live = False
        cost = 0.0

        if answer is None:
            if not live:
                answer = (
                    "That question needs a live web search, which is currently disabled. "
                    "Enable **Allow live web answers** above, or ask something about this "
                    'workforce - for example *"how many high risk employees are in Sales?"*.'
                )
            else:
                try:
                    with st.spinner(
                        "Searching the live web and synthesising an answer - "
                        "this can take up to a minute..."
                    ):
                        result = assistant.answer_external(prompt)
                    answer = result["answer"]
                    cost = result.get("cost_usd", 0.0)
                    used_live = True
                    if result.get("sources"):
                        st.markdown("\n".join(f"- [{u}]({u})" for u in result["sources"][:5]))
                    if result.get("cached"):
                        st.caption("Served from cache - no credit used.")
                except assistant.TinyFishUnavailable as exc:
                    answer = f"Live search is unavailable: {exc}"

        st.markdown(answer)
        if used_live and cost:
            st.caption(f"Used ~${cost:.2f} of TinyFish wallet credit.")

    st.session_state["messages"].append({"role": "assistant", "content": answer})


def render() -> None:
    st.markdown(
        theme.app_header(
            "HR Insights Assistant",
            "Questions about this workforce are answered locally, instantly and for free. "
            "Questions about the wider labour market use a live web search.",
        ),
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(theme.section_label("Try one of these"), unsafe_allow_html=True)
        live = st.toggle(
            "Allow live web answers",
            value=True,
            help=(
                "Live answers use a TinyFish Agent run, which costs about "
                f"${assistant.LIVE_ANSWER_COST_USD:.2f} of wallet credit per new answer. "
                "If the agent is unavailable, a free web-fetch excerpt is used instead. "
                "Answers are cached, so repeating a question is free. "
                "Turn this off to gather sources only, at no cost."
            ),
        )
        st.markdown(
            f'<div class="small-note">Live answers cost roughly '
            f"${assistant.LIVE_ANSWER_COST_USD:.2f} each (about five agent steps). "
            "Cached answers are free.</div>",
            unsafe_allow_html=True,
        )
        selected_example = st.pills(
            "Example questions",
            list(assistant.EXAMPLE_QUESTIONS),
            label_visibility="collapsed",
            selection_mode="single",
        )
        stats = assistant.cache_stats()
        st.markdown(
            f'<div class="small-note">{stats["cached_answers"]} answers cached &middot; '
            f"~${stats['credits_saved_usd']:.2f} of credit saved</div>",
            unsafe_allow_html=True,
        )

    # Selecting an example pill schedules the question for the normal dispatch
    # below, so every answer is rendered in one place and the history stays
    # single-source (no duplicated bubbles on rerun).
    if selected_example and selected_example != st.session_state.get("asked_example"):
        st.session_state["asked_example"] = selected_example
        st.session_state["pending_prompt"] = selected_example
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask about attrition risk, the talent matrix, the fairness audit, or the labour market..."
    )
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        _run_prompt(prompt, live)
        st.rerun()
