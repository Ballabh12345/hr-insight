"""Application smoke tests - every page must render without raising."""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "main.py"

PAGES = [
    "Overview",
    "Employees",
    "Employee detail",
    "Fairness audit",
    "Model performance",
    "Decision log",
    "Assistant",
]


def _run() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()
    return at


@pytest.fixture(scope="module")
def app_test():
    return _run()


def test_app_starts_without_exception(app_test):
    assert not app_test.exception, app_test.exception


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _run()
    radio = at.sidebar.radio[0]
    radio.set_value(page)
    at.run()
    assert not at.exception, f"{page} raised: {at.exception}"


def test_navigation_contains_every_page(app_test):
    # Nav labels render as "<glyph>  <page name>"; the pin is on the page set.
    options = [o.split("  ", 1)[-1] for o in app_test.sidebar.radio[0].options]
    assert options == PAGES


def test_employees_table_renders_rows(app_test):
    at = _run()
    at.sidebar.radio[0].set_value("Employees")
    at.run()
    assert len(at.dataframe) >= 1
    assert not at.exception


def test_employee_detail_renders_an_employee(app_test):
    at = _run()
    at.sidebar.radio[0].set_value("Employee detail")
    at.run()
    assert len(at.selectbox) >= 1
    assert not at.exception
