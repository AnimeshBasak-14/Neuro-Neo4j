"""
retrain_trigger.py
==================
Automated retraining trigger for the Neuro-Neo4j pipeline.

Monitors drift status and new data accumulation in Neo4j,
then triggers a retraining cycle when conditions are met.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from monitoring.drift_monitor import DriftMonitor

logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Monitors drift and data accumulation; triggers retraining when warranted.

    Retraining is triggered when BOTH conditions are true:
      1. DriftMonitor detects significant Macro-F1 degradation
      2. At least `min_new_samples` new subjects have been added to Neo4j

    Parameters
    ----------
    drift_monitor     : DriftMonitor
    min_new_samples   : int    Minimum new subjects required to retrain
    config            : dict   Config dict (monitoring section)
    neo4j_driver      : Any    Neo4j driver (None = mock mode)
    """

    def __init__(
        self,
        drift_monitor:   DriftMonitor,
        min_new_samples: int = 20,
        config:          dict | None = None,
        neo4j_driver:    Optional[object] = None,
    ) -> None:
        self.drift_monitor   = drift_monitor
        self.min_new_samples = min_new_samples
        self.config          = config or {}
        self.neo4j_driver    = neo4j_driver
        self._mock           = neo4j_driver is None

        self._last_train_time:    Optional[datetime] = None
        self._retrain_count:      int = 0
        self._is_running:         bool = False
        self._thread:             Optional[threading.Thread] = None
        self._stop_event:         threading.Event = threading.Event()

        monitor_cfg = self.config.get("monitoring", self.config)
        self._interval = monitor_cfg.get("monitor_interval_seconds", 3600)

        logger.info(
            "RetrainingTrigger initialised | min_new_samples=%d interval=%ds mock=%s",
            min_new_samples, self._interval, self._mock,
        )

    # ------------------------------------------------------------------
    # Core trigger logic
    # ------------------------------------------------------------------

    def check_and_trigger(self) -> bool:
        """
        Evaluate drift and data conditions; trigger retraining if warranted.

        Returns
        -------
        bool  True if retraining was triggered
        """
        logger.info("RetrainingTrigger: checking conditions...")

        drift_detected = self.drift_monitor.detect_drift()
        new_samples    = self._count_new_samples_in_neo4j()
        report         = self.drift_monitor.get_drift_report()

        logger.info(
            "  Drift detected:   %s (Δ F1 = %.4f)",
            drift_detected, report.get("drift_delta", 0.0),
        )
        logger.info(
            "  New samples:      %d (threshold: %d)",
            new_samples, self.min_new_samples,
        )

        if drift_detected and new_samples >= self.min_new_samples:
            logger.warning(
                "⚠️  Retraining conditions met! Triggering retraining cycle..."
            )
            self._launch_retraining()
            return True

        if drift_detected:
            logger.warning(
                "Drift detected but insufficient new samples (%d < %d). "
                "Waiting for more data...",
                new_samples, self.min_new_samples,
            )
        else:
            logger.info("No retraining required at this time.")

        return False

    # ------------------------------------------------------------------
    # Neo4j: count new subjects since last training
    # ------------------------------------------------------------------

    def _count_new_samples_in_neo4j(self) -> int:
        """
        Query Neo4j for the number of new subjects added since last training.

        Returns dummy count in mock mode.

        Returns
        -------
        int  Number of new subjects
        """
        if self._mock:
            import numpy as np
            # Simulate: return a random count between 0 and 30
            count = int(np.random.default_rng(int(time.time())).integers(0, 30))
            logger.debug("[MOCK] New samples since last training: %d", count)
            return count

        if self._last_train_time is None:
            # First run: count all subjects
            query = "MATCH (s:Subject) RETURN count(s) AS n"
        else:
            query = """
            MATCH (s:Subject)
            WHERE s.createdAt > datetime($lastTrain)
            RETURN count(s) AS n
            """
        try:
            with self.neo4j_driver.session() as session:
                params = {"lastTrain": self._last_train_time.isoformat()} if self._last_train_time else {}
                result = session.run(query, parameters=params)
                return int(result.single()["n"])
        except Exception as exc:
            logger.error("Neo4j query failed in retrain trigger: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Retraining launch
    # ------------------------------------------------------------------

    def _launch_retraining(self) -> None:
        """
        Launch the retraining cycle.

        In production: submits a training job to the compute cluster.
        In dry-run / development: logs the event and simulates a 5-second run.
        """
        self._last_train_time = datetime.utcnow()
        self._retrain_count  += 1

        logger.info("=" * 60)
        logger.info("🔄 RETRAINING CYCLE #%d initiated at %s",
                    self._retrain_count, self._last_train_time.isoformat())
        logger.info("   Mode: %s", "MOCK (dry run)" if self._mock else "PRODUCTION")
        logger.info("   Drift report: %s", self.drift_monitor.get_drift_report())
        logger.info("=" * 60)

        if self._mock:
            logger.info("[MOCK] Simulating 5-second training run...")
            time.sleep(1)  # Short mock sleep
            logger.info("[MOCK] Retraining complete. Model checkpoint updated.")
        else:
            # In production: subprocess.run(["uv", "run", "python", "models/gnn/train.py"])
            logger.info("Production retraining job submission not implemented in this scaffold.")

        # Reset drift monitor after retraining
        self.drift_monitor.reset()
        logger.info("DriftMonitor reset after retraining.")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def start_background_monitoring(self) -> None:
        """Start a background thread that checks drift periodically."""
        if self._is_running:
            logger.warning("Background monitoring already running.")
            return

        self._stop_event.clear()
        self._is_running = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            name="RetrainingMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Background monitoring started (interval=%ds).", self._interval
        )

    def stop_background_monitoring(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Background monitoring stopped.")

    def _monitoring_loop(self) -> None:
        """Periodic monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check_and_trigger()
                self.drift_monitor.save_log()
            except Exception as exc:
                logger.error("Monitoring loop error: %s", exc)
            self._stop_event.wait(timeout=self._interval)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current trigger status."""
        return {
            "is_running":       self._is_running,
            "retrain_count":    self._retrain_count,
            "last_train_time":  self._last_train_time.isoformat() if self._last_train_time else None,
            "min_new_samples":  self.min_new_samples,
            "interval_seconds": self._interval,
            "mock_mode":        self._mock,
        }
