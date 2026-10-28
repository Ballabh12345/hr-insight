"""Optional experiment tracking (MLflow) for HR Insight training runs.

MLflow is an optional dependency. When it is not installed the helpers become
no-ops, so training scripts keep working in a minimal environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[Path] | None = None,
) -> None:
    """Log a run to MLflow if available, otherwise do nothing."""
    mlflow = _mlflow()
    if mlflow is None:
        return
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _is_number(v)})
            for path in artifacts or []:
                if Path(path).exists():
                    mlflow.log_artifact(str(path))
    except Exception:
        # Tracking must never break a training run.
        return


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False