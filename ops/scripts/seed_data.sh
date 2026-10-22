#!/usr/bin/env bash
# Verify the dataset is present and matches the expected checksum.
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import sys
sys.path.insert(0, ".")
from ml.common import io

path = io.download_ibm_hr()
digest = io.sha256(path)
print(f"[hr-insight] dataset: {path}")
print(f"[hr-insight] sha256:  {digest}")
assert digest == io.IBM_HR_SHA256, "dataset checksum mismatch"
print("[hr-insight] dataset OK")
PY
