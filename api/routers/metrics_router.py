"""
metrics_router.py
=================
/metrics endpoints: in-memory performance metrics store with Prometheus export.

Author: basakanimesh14
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from api.schemas import MetricsResponse

router = APIRouter(prefix="/metrics", tags=["Metrics"])

# ---------------------------------------------------------------------------
# Thread-safe in-memory metrics store
# ---------------------------------------------------------------------------

class MetricsStore:
    """
    Thread-safe in-memory store for classification performance metrics.

    Maintains a sliding window of per-prediction (y_true, y_pred, y_prob)
    tuples and computes aggregate statistics on demand.
    """

    _MAX_HISTORY = 50  # max number of snapshots to retain

    def __init__(self, baseline_f1: float = 0.80, drift_threshold: float = 0.05) -> None:
        self._lock          = threading.Lock()
        self.baseline_f1    = baseline_f1
        self.drift_threshold= drift_threshold
        self._predictions: deque = deque(maxlen=1000)  # (y_true, y_pred, y_prob)
        self._history: deque[MetricsResponse] = deque(maxlen=self._MAX_HISTORY)
        self._last_updated  = datetime.utcnow().isoformat()

    def record(self, y_true: int, y_pred: int, y_prob: float) -> None:
        """Record a single prediction."""
        with self._lock:
            self._predictions.append((y_true, y_pred, y_prob))
            self._last_updated = datetime.utcnow().isoformat()

    def compute(self) -> MetricsResponse:
        """Compute metrics from accumulated predictions."""
        with self._lock:
            preds = list(self._predictions)

        if not preds:
            return MetricsResponse(
                macro_f1=0.0, accuracy=0.0, roc_auc=0.5,
                n_predictions=0, drift_detected=False,
                last_updated=self._last_updated,
                baseline_f1=self.baseline_f1,
                drift_delta=0.0,
            )

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import numpy as np

        y_true  = [p[0] for p in preds]
        y_pred  = [p[1] for p in preds]
        y_prob  = [p[2] for p in preds]

        acc  = float(accuracy_score(y_true, y_pred))
        f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        drift_delta   = abs(f1 - self.baseline_f1)
        drift_detected= drift_delta > self.drift_threshold

        snapshot = MetricsResponse(
            macro_f1=round(f1, 4),
            accuracy=round(acc, 4),
            roc_auc=round(auc, 4),
            n_predictions=len(preds),
            drift_detected=drift_detected,
            last_updated=self._last_updated,
            baseline_f1=self.baseline_f1,
            drift_delta=round(drift_delta, 4),
        )
        with self._lock:
            self._history.append(snapshot)
        return snapshot

    def history(self) -> list[MetricsResponse]:
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        with self._lock:
            self._predictions.clear()
            self._last_updated = datetime.utcnow().isoformat()


# Singleton store (shared across requests)
_metrics_store = MetricsStore()


def get_metrics_store() -> MetricsStore:
    return _metrics_store


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=MetricsResponse,
    summary="Current classification performance metrics",
)
def get_metrics() -> MetricsResponse:
    """
    Return current Macro-F1, Accuracy, ROC-AUC, and drift status
    computed over all predictions received since last reset.
    """
    return _metrics_store.compute()


@router.post(
    "/update",
    summary="Record a ground-truth label for drift monitoring",
)
def update_metrics(y_true: int, y_pred: int, y_prob: float) -> dict:
    """
    Record a verified (y_true, y_pred, y_prob) tuple.

    Called after clinical validation of a prediction.
    """
    if y_true not in (0, 1) or y_pred not in (0, 1):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="y_true and y_pred must be 0 or 1.",
        )
    _metrics_store.record(y_true, y_pred, y_prob)
    return {"status": "recorded", "n_predictions": len(list(_metrics_store._predictions))}


@router.get(
    "/history",
    response_model=list[MetricsResponse],
    summary="Historical metric snapshots",
)
def get_metrics_history() -> list[MetricsResponse]:
    """Return the last N metric snapshots."""
    return _metrics_store.history()


@router.delete(
    "/reset",
    summary="Reset the metrics store",
)
def reset_metrics() -> dict:
    """Clear all accumulated predictions and history."""
    _metrics_store.reset()
    return {"status": "reset", "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/prometheus",
    response_class=None,
    summary="Prometheus-format metrics export",
    include_in_schema=True,
)
def prometheus_metrics():
    """
    Return metrics in Prometheus text exposition format.

    Add this endpoint to your Prometheus scrape config:
      - targets: ["localhost:8000"]
        metrics_path: /metrics/prometheus
    """
    from fastapi.responses import PlainTextResponse
    m = _metrics_store.compute()
    lines = [
        "# HELP neuro_neo4j_macro_f1 Macro-averaged F1 score",
        "# TYPE neuro_neo4j_macro_f1 gauge",
        f"neuro_neo4j_macro_f1 {m.macro_f1}",
        "# HELP neuro_neo4j_accuracy Classification accuracy",
        "# TYPE neuro_neo4j_accuracy gauge",
        f"neuro_neo4j_accuracy {m.accuracy}",
        "# HELP neuro_neo4j_roc_auc ROC-AUC score",
        "# TYPE neuro_neo4j_roc_auc gauge",
        f"neuro_neo4j_roc_auc {m.roc_auc}",
        "# HELP neuro_neo4j_n_predictions Total predictions made",
        "# TYPE neuro_neo4j_n_predictions counter",
        f"neuro_neo4j_n_predictions {m.n_predictions}",
        "# HELP neuro_neo4j_drift_detected Whether drift has been detected (1=yes)",
        "# TYPE neuro_neo4j_drift_detected gauge",
        f"neuro_neo4j_drift_detected {1 if m.drift_detected else 0}",
    ]
    return PlainTextResponse(content="\n".join(lines) + "\n")
