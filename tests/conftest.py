"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def raw_df():
    """The raw IBM HR dataset, downloaded and verified."""
    from ml.common import io

    return pd_read_csv(io.download_ibm_hr())


@pytest.fixture(scope="session")
def cleaned_df(raw_df):
    """The cleaned modelling frame."""
    from ml.common import features as F

    return F.clean(F.add_age_band(raw_df))


@pytest.fixture(scope="session")
def employees_df():
    """The assembled employee dataset the application serves."""
    from app.services import data_service as data

    return data.load_employees()


def pd_read_csv(path):
    import pandas as pd

    return pd.read_csv(path)
