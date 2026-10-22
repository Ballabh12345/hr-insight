#!/usr/bin/env bash
# Rebuild every model artifact and the dataset the application serves.
#
# Safe to run on a schedule: the pipeline is deterministic (random_state=42
# throughout), so a rebuild with unchanged inputs produces identical outputs.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[hr-insight] retraining started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python ml/train_all.py
echo "[hr-insight] retraining finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
