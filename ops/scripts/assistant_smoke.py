"""End-to-end check of the assistant: one internal question, one live-web question.

Run against a live server:
    streamlit run app/main.py --server.port 8601
    python ops/scripts/assistant_smoke.py --port 8601
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "screenshots"

QUESTIONS = [
    "How many employees are high risk?",                      # internal, free
    "What is the fairness gap on age?",                      # internal, free
    "What is the average salary for a Data Scientist in India in 2026?",  # live web
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8601)
    parser.add_argument("--skip-live", action="store_true", help="skip the paid live-web question")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    questions = QUESTIONS[:2] if args.skip_live else QUESTIONS

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=2)
        page.goto(f"http://localhost:{args.port}/", wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(5000)

        page.get_by_text("Assistant", exact=True).first.click()
        page.wait_for_timeout(2500)

        for question in questions:
            print(f"[ask] {question}")
            page.get_by_placeholder(
                "Ask about attrition risk, the talent matrix, the fairness audit, or the labour market..."
            ).fill(question)
            page.keyboard.press("Enter")
            # Give the agent run time to complete.
            page.wait_for_timeout(20_000 if "salary" in question else 4000)
            page.wait_for_timeout(3000)

        path = OUT / "08_assistant_conversation.png"
        page.screenshot(path=str(path), full_page=False)
        print(f"[shot] {path.relative_to(ROOT)}")

        print("\n--- transcript ---")
        for question in questions:
            print(f"Q: {question}")
        print("\n(page text captured; check the screenshot for the answers)")

        browser.close()


if __name__ == "__main__":
    main()
