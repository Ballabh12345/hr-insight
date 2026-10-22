"""Filesystem paths, dataset acquisition, and artifact IO for HR Insight."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CATALOG_DIR = DATA_DIR / "catalog"
DB_DIR = DATA_DIR / "db"

ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
DOCS_DIR = ROOT / "docs"

DB_PATH = DB_DIR / "hr_insight.db"

for _d in (RAW_DIR, PROCESSED_DIR, CATALOG_DIR, DB_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

#: Kaggle-free mirror of the IBM HR Analytics Employee Attrition & Performance dataset.
#: The dataset is fictional - created by IBM data scientists - and released under CC0.
IBM_HR_URL = (
    "https://raw.githubusercontent.com/shantanu1109/"
    "IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction/"
    "master/DATASET/IBM-HR-Analytics-Employee-Attrition-and-Performance.csv"
)

#: SHA256 of the canonical CSV (1470 rows x 35 columns).
IBM_HR_SHA256 = "a5c31e38bd7fafc9bc333884eb181b06b41b8e5e488e8f7ccb27199fb3be7659"

IBM_HR_PATH = RAW_DIR / "ibm_hr.csv"


def sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_ibm_hr(force: bool = False) -> Path:
    """Download the IBM HR dataset if absent, then verify its checksum.

    Returns the path to the local CSV. Raises RuntimeError on checksum mismatch.
    """
    if IBM_HR_PATH.exists() and not force:
        digest = sha256(IBM_HR_PATH)
        if digest == IBM_HR_SHA256:
            return IBM_HR_PATH
        raise RuntimeError(
            f"Existing {IBM_HR_PATH.name} has unexpected checksum {digest}. "
            "Delete it and re-run, or pass force=True to overwrite."
        )

    IBM_HR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = IBM_HR_PATH.with_suffix(".csv.part")
    with urllib.request.urlopen(IBM_HR_URL, timeout=120) as resp:  # noqa: S310
        tmp.write_bytes(resp.read())

    digest = sha256(tmp)
    if digest != IBM_HR_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for downloaded dataset.\n  expected {IBM_HR_SHA256}\n  got      {digest}"
        )
    tmp.replace(IBM_HR_PATH)
    return IBM_HR_PATH


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def save_artifact(obj: Any, name: str) -> Path:
    """Persist a Python object to ``ml/artifacts/<name>``."""
    path = ARTIFACTS_DIR / name
    joblib.dump(obj, path, compress=3)
    return path


def load_artifact(name: str) -> Any:
    """Load an object previously written by :func:`save_artifact`."""
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact '{name}' not found in {ARTIFACTS_DIR}. "
            "Run `python ml/train_all.py` to build the models."
        )
    return joblib.load(path)


def artifact_exists(name: str) -> bool:
    """True when the named artifact is present on disk."""
    return (ARTIFACTS_DIR / name).exists()


def write_json(obj: Any, path: Path) -> Path:
    """Write JSON to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    """Read JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
