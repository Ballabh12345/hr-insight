"""Capture README/demo screenshots of the running application.

Usage:
    streamlit run app/main.py --server.port 8601     # in another shell
    python ops/scripts/screenshots.py --port 8601
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "screenshots"

PAGES = [
    ("Overview", "01_overview"),
    ("Employees", "02_employees"),
    ("Employee detail", "03_employee_detail"),
    ("Fairness audit", "04_fairness_audit"),
    ("Model performance", "05_model_performance"),
    ("Decision log", "06_decision_log"),
    ("Assistant", "07_assistant"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8601)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    url = f"http://localhost:{args.port}/"

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page.goto(url, wait_until="networkidle", timeout=90_000)
        # Streamlit renders over a websocket; wait for the first block to paint.
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=90_000)
        page.wait_for_timeout(6000)

        for label, filename in PAGES:
            try:
                page.get_by_text(label, exact=True).first.click(timeout=15_000)
                page.wait_for_timeout(3500)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] could not open {label!r}: {exc}", file=sys.stderr)
            path = OUT / f"{filename}.png"
            page.screenshot(path=str(path), full_page=False)
            print(f"[shot] {path.relative_to(ROOT)}")

        browser.close()

    print(f"\nScreenshots written to {OUT}")


if __name__ == "__main__":
    main()
