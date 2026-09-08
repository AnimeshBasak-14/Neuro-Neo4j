"""
drift_monitor.py
================
Concept and Data Drift detection for the Neuro-Neo4j pipeline.

Monitors Macro-F1 degradation over a sliding window of predictions.
Triggers an alert when performance drops below the established baseline.

Author: basakanimesh14
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADWIN-inspired Sliding Window Detector
# ---------------------------------------------------------------------------

class ConceptDriftDetector:
    """
    Simple sliding-window drift detector inspired by ADWIN.

    Maintains two windows (reference vs. recent) and flags drift
    when their means diverge beyond a threshold.

    Parameters
    ----------
    window_size : int    Number of recent samples in the detection window
    threshold   : float  Minimum mean difference to declare drift
    """

    def __init__(self, window_size: int = 50, threshold: float = 0.05) -> None:
        self.window_size = window_size
        self.threshold   = threshold
        self._reference: list[float] = []
        self._recent:    deque       = deque(maxlen=window_size)

    def update(self, value: float) -> bool:
        """
        Add a new observation. Returns True if drift is detected.

        Parameters
        ----------
        value : float  Observed metric (e.g., per-prediction accuracy)

        Returns
        -------
        bool  True if statistically significant drift detected
        """
        self._recent.append(value)

        if not self._reference and len(self._recent) >= self.window_size:
            # First window fills the reference
            self._reference = list(self._recent)
            return False

        if self._reference and len(self._recent) >= self.window_size // 2:
            ref_mean    = np.mean(self._reference)
            recent_mean = np.mean(list(self._recent))
            if abs(ref_mean - recent_mean) > self.threshold:
                logger.warning(
                    "Drift detected! ref_mean=%.4f recent_mean=%.4f Δ=%.4f",
                    ref_mean, recent_mean, abs(ref_mean - recent_mean),
                )
                return True
        return False

    def reset_reference(self) -> None:
        """Update reference window to current observations."""
        self._reference = list(self._recent)
        logger.info("Drift detector: reference window updated.")


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Monitors Macro-F1 over a sliding prediction window.

    Records per-prediction ground-truth outcomes (after clinical validation)
    and computes running metrics. Detects drift when |F1 − baseline| > threshold.

    Parameters
    ----------
    baseline_f1      : float  F1 from the validated test set
    threshold        : float  Drift alert threshold
    log_file         : str    Path to JSON log file
    sliding_window   : int    Size of prediction window
    """

    def __init__(
        self,
        baseline_f1:   float = 0.80,
        threshold:     float = 0.05,
        log_file:      str   = "logs/drift.json",
        sliding_window: int  = 100,
    ) -> None:
        self.baseline_f1    = baseline_f1
        self.threshold      = threshold
        self.log_file       = Path(log_file)
        self.sliding_window = sliding_window

        self._lock       = threading.Lock()
        self._predictions: deque = deque(maxlen=sliding_window)  # (y_true, y_pred, y_prob, ts)
        self._detector   = ConceptDriftDetector(window_size=sliding_window // 2, threshold=threshold)

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DriftMonitor initialised | baseline_f1=%.3f threshold=%.3f window=%d",
            baseline_f1, threshold, sliding_window,
        )

    # ------------------------------------------------------------------
    # Record
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        y_true:    int,
        y_pred:    int,
        y_prob:    float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a verified prediction for drift tracking.

        Parameters
        ----------
        y_true     : int       Ground-truth label (0 or 1)
        y_pred     : int       Predicted label
        y_prob     : float     Probability of positive class
        timestamp  : datetime  Recording time (defaults to now)
        """
        ts = (timestamp or datetime.utcnow()).isoformat()
        with self._lock:
            self._predictions.append((y_true, y_pred, y_prob, ts))
        # Update drift detector with per-sample accuracy
        self._detector.update(float(y_true == y_pred))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_current_metrics(self) -> dict:
        """
        Compute metrics over the current sliding window.

        Returns
        -------
        dict: accuracy, macro_f1, n_samples, drift_detected, drift_delta
        """
        with self._lock:
            preds = list(self._predictions)

        if len(preds) < 2:
            return {
                "accuracy": None, "macro_f1": None,
                "n_samples": len(preds), "drift_detected": False, "drift_delta": 0.0,
            }

        y_true = [p[0] for p in preds]
        y_pred = [p[1] for p in preds]

        acc = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        drift_delta    = abs(f1 - self.baseline_f1)
        drift_detected = drift_delta > self.threshold

        return {
            "accuracy":       round(acc, 4),
            "macro_f1":       round(f1, 4),
            "n_samples":      len(preds),
            "drift_detected": drift_detected,
            "drift_delta":    round(drift_delta, 4),
            "baseline_f1":    self.baseline_f1,
            "threshold":      self.threshold,
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def detect_drift(self) -> bool:
        """
        Returns True if Macro-F1 has degraded beyond the threshold.

        Returns
        -------
        bool
        """
        metrics = self.compute_current_metrics()
        return bool(metrics.get("drift_detected", False))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_drift_report(self) -> dict:
        """
        Generate a comprehensive drift report.

        Returns
        -------
        dict with current metrics, history summary, and alert flag
        """
        metrics = self.compute_current_metrics()
        return {
            "timestamp":      datetime.utcnow().isoformat(),
            "baseline_f1":    self.baseline_f1,
            "drift_threshold":self.threshold,
            "window_size":    self.sliding_window,
            **metrics,
            "alert": "🚨 DRIFT DETECTED" if metrics.get("drift_detected") else "✅ NO DRIFT",
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_log(self) -> None:
        """Append current metrics snapshot to JSON log file."""
        report = self.get_drift_report()
        logs: list[dict] = []
        if self.log_file.exists():
            with open(self.log_file) as f:
                logs = json.load(f)
        logs.append(report)
        with open(self.log_file, "w") as f:
            json.dump(logs, f, indent=2)
        logger.debug("Drift log updated: %s", self.log_file)

    def load_log(self) -> list[dict]:
        """Load drift history from JSON log file."""
        if not self.log_file.exists():
            return []
        with open(self.log_file) as f:
            return json.load(f)

    def reset(self) -> None:
        """Clear all prediction history."""
        with self._lock:
            self._predictions.clear()
        self._detector.reset_reference()
        logger.info("DriftMonitor reset.")
