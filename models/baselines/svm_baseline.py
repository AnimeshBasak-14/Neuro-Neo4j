"""
svm_baseline.py
===============
Baseline 1: Support Vector Machine on flattened connectivity vectors.

Graph Archetype: Linear/Grid — connectivity matrix is unrolled into
an independent feature list (upper triangle), losing all relational
structure. This establishes the performance floor that GNNs must exceed.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.metrics import MetricsReport

logger = logging.getLogger(__name__)

# AAL-116: upper triangle has 116*115//2 = 6670 features
N_UPPER_TRIANGLE_116 = 116 * 115 // 2


class SVMBaseline:
    """
    SVM classifier on flattened functional connectivity vectors.

    Feature extraction: upper triangle of the (n_regions × n_regions)
    correlation matrix is vectorised into a flat (n_features,) array.
    StandardScaler normalisation is applied inside a sklearn Pipeline.

    Parameters
    ----------
    config : dict
        Baseline SVM config (from config.yaml `baselines.svm` section).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        svm_cfg = config.get("svm", config)  # accept nested or flat config
        self.kernel       = svm_cfg.get("kernel", "rbf")
        self.C            = svm_cfg.get("C", 1.0)
        self.gamma        = svm_cfg.get("gamma", "scale")
        self.class_weight = svm_cfg.get("class_weight", "balanced")
        self.probability  = svm_cfg.get("probability", True)

        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False

        logger.info(
            "SVMBaseline initialised | kernel=%s C=%.2f gamma=%s",
            self.kernel, self.C, self.gamma,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        edge_list_df: pd.DataFrame,
        n_regions: int = 116,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build subject-level feature matrices from edge list DataFrame.

        Reconstructs the (n_regions × n_regions) connectivity matrix per
        subject, then extracts the upper triangle as a flat feature vector.

        Parameters
        ----------
        edge_list_df : pd.DataFrame
            Long-format edge list with columns [subject_id, source, target, weight, diagnosis].
        n_regions    : int  Number of brain regions

        Returns
        -------
        X : np.ndarray  (n_subjects, n_features)
        y : np.ndarray  (n_subjects,)  binary labels (0=Healthy, 1=Diseased)
        """
        subject_ids = edge_list_df["subject_id"].unique()
        region_names = sorted(
            list(set(edge_list_df["source"].unique()) | set(edge_list_df["target"].unique()))
        )
        region_to_idx = {r: i for i, r in enumerate(region_names)}
        n_actual = len(region_names)

        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for subj in subject_ids:
            subj_df = edge_list_df[edge_list_df["subject_id"] == subj]
            matrix = np.zeros((n_actual, n_actual), dtype=np.float32)

            for _, row in subj_df.iterrows():
                i = region_to_idx.get(row["source"], -1)
                j = region_to_idx.get(row["target"], -1)
                if i >= 0 and j >= 0:
                    matrix[i, j] = row["weight"]
                    matrix[j, i] = row["weight"]

            # Extract upper triangle
            upper = matrix[np.triu_indices(n_actual, k=1)]
            X_list.append(upper)

            label = 1 if subj_df.iloc[0]["diagnosis"] == "Diseased" else 0
            y_list.append(label)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        logger.info(
            "SVMBaseline features prepared: X=%s y=%s (Healthy=%d Diseased=%d)",
            X.shape, y.shape, int((y == 0).sum()), int((y == 1).sum()),
        )
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMBaseline":
        """
        Fit the SVM pipeline (StandardScaler → SVC).

        Parameters
        ----------
        X : np.ndarray  (n_samples, n_features)
        y : np.ndarray  (n_samples,)

        Returns
        -------
        self
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                class_weight=self.class_weight,
                probability=self.probability,
            )),
        ])
        self.pipeline.fit(X, y)
        self._is_fitted = True
        logger.info("SVMBaseline fitted on %d samples", len(y))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Shape (n, 2)."""
        self._check_fitted()
        return self.pipeline.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> MetricsReport:
        """
        Evaluate on test set and return a MetricsReport.

        Parameters
        ----------
        X_test : np.ndarray
        y_test : np.ndarray

        Returns
        -------
        MetricsReport
        """
        y_pred   = self.predict(X_test)
        y_scores = self.predict_proba(X_test)
        report = MetricsReport.from_arrays(y_test, y_pred, y_scores, split="test")
        logger.info("SVMBaseline evaluation:\n%s", report)
        return report

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def dry_run(self) -> None:
        """
        Print model summary without fitting.

        Validates that the model can be instantiated and configured
        correctly. No data required.
        """
        print("=" * 50)
        print("SVMBaseline — Dry Run Summary")
        print("=" * 50)
        print(f"  Kernel       : {self.kernel}")
        print(f"  C            : {self.C}")
        print(f"  Gamma        : {self.gamma}")
        print(f"  Class weight : {self.class_weight}")
        print(f"  Probability  : {self.probability}")
        print(f"  Feature dim  : {N_UPPER_TRIANGLE_116} (AAL-116 upper triangle)")
        print(f"  Pipeline     : StandardScaler → SVC")
        print("  Status       : NOT fitted (dry run)")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("SVMBaseline must be fitted before inference. Call .fit() first.")
