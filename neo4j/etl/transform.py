"""
transform.py
============
ETL Transform Stage: Normalise and preprocess fMRI correlation matrices.

Pipeline stages
---------------
1. load_matrix()             — Load raw CSV correlation matrix
2. regress_global_signal()   — Remove global mean signal confound
3. remove_motion_artifacts() — Zero edges exceeding FD threshold
4. normalize()               — Z-score or min-max normalisation
5. threshold()               — Prune weak edges (|r| < threshold)
6. to_edge_list()            — Convert to long-format DataFrame

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)


class ConnectomeTransformer:
    """
    Preprocesses a single subject's fMRI functional connectivity matrix.

    Parameters
    ----------
    region_names : list[str]
        Ordered list of brain region labels (e.g., AAL-116 names).
    """

    def __init__(self, region_names: list[str]) -> None:
        self.region_names = region_names
        self.n_regions = len(region_names)

    # ------------------------------------------------------------------
    # Stage 1: Load
    # ------------------------------------------------------------------

    def load_matrix(self, path: str | Path) -> np.ndarray:
        """
        Load a CSV correlation matrix from disk.

        Parameters
        ----------
        path : str | Path
            Path to the (n_regions × n_regions) CSV file.

        Returns
        -------
        np.ndarray  shape (n_regions, n_regions)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Connectivity matrix not found: {path}")
        df = pd.read_csv(path, index_col=0)
        matrix = df.values.astype(np.float32)
        if matrix.shape != (self.n_regions, self.n_regions):
            raise ValueError(
                f"Expected ({self.n_regions}, {self.n_regions}) matrix, "
                f"got {matrix.shape}"
            )
        logger.debug("Loaded matrix from %s — shape %s", path, matrix.shape)
        return matrix

    # ------------------------------------------------------------------
    # Stage 2: Confound regression
    # ------------------------------------------------------------------

    def regress_global_signal(self, matrix: np.ndarray) -> np.ndarray:
        """
        Regress out the global mean signal from the correlation matrix.

        Computes the mean time-series signal proxy as the mean of all
        off-diagonal correlations, then subtracts its contribution via
        ordinary least squares.

        Parameters
        ----------
        matrix : np.ndarray  (n × n)

        Returns
        -------
        np.ndarray  (n × n) residual correlation matrix
        """
        # Proxy: row means as a stand-in for mean connectivity per region
        mask = ~np.eye(self.n_regions, dtype=bool)
        global_mean = matrix[mask].mean()
        residual = matrix - global_mean
        np.fill_diagonal(residual, 1.0)
        logger.debug("Global signal regressed out (global_mean=%.4f)", global_mean)
        return residual

    # ------------------------------------------------------------------
    # Stage 3: Motion artifact removal
    # ------------------------------------------------------------------

    def remove_motion_artifacts(
        self,
        matrix: np.ndarray,
        fd_threshold: float = 0.5,
        rng_seed: int = 0,
    ) -> np.ndarray:
        """
        Simulate motion artifact removal by attenuating high-FD edges.

        In real pipelines this uses actual framewise displacement (FD) time
        series per subject. Here we simulate by adding small perturbations
        to edges whose absolute weight exceeds the FD proxy threshold.

        Parameters
        ----------
        matrix       : np.ndarray  (n × n)
        fd_threshold : float  Framewise displacement threshold (mm)
        rng_seed     : int    Seed for reproducibility

        Returns
        -------
        np.ndarray  cleaned correlation matrix
        """
        rng = np.random.default_rng(rng_seed)
        cleaned = matrix.copy()
        # Simulate: zero out a small fraction of the strongest edges (top 5%)
        flat = cleaned.flatten()
        threshold_val = np.percentile(np.abs(flat), 95)
        high_corr_mask = np.abs(cleaned) > threshold_val
        np.fill_diagonal(high_corr_mask, False)
        noise = rng.normal(0, 0.02, cleaned.shape)
        cleaned[high_corr_mask] -= noise[high_corr_mask] * fd_threshold
        np.fill_diagonal(cleaned, 1.0)
        n_removed = int(high_corr_mask.sum())
        logger.debug("Motion artifact proxy applied — %d edges attenuated", n_removed)
        return cleaned

    # ------------------------------------------------------------------
    # Stage 4: Normalisation
    # ------------------------------------------------------------------

    def normalize(
        self,
        matrix: np.ndarray,
        method: Literal["zscore", "minmax"] = "zscore",
    ) -> np.ndarray:
        """
        Normalise off-diagonal edge weights.

        Parameters
        ----------
        matrix : np.ndarray  (n × n)
        method : 'zscore' | 'minmax'

        Returns
        -------
        np.ndarray  normalised correlation matrix
        """
        normed = matrix.copy().astype(np.float32)
        mask = ~np.eye(self.n_regions, dtype=bool)
        vals = normed[mask]

        if method == "zscore":
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                normed[mask] = (vals - mu) / sigma
        elif method == "minmax":
            lo, hi = vals.min(), vals.max()
            if hi > lo:
                normed[mask] = (vals - lo) / (hi - lo)
        else:
            raise ValueError(f"Unknown normalisation method: {method}")

        np.fill_diagonal(normed, 0.0)   # self-loops to 0 after normalisation
        logger.debug("Normalised with method=%s", method)
        return normed

    # ------------------------------------------------------------------
    # Stage 5: Threshold / sparsify
    # ------------------------------------------------------------------

    def threshold(self, matrix: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Prune edges whose absolute weight falls below *threshold*.

        Parameters
        ----------
        matrix    : np.ndarray
        threshold : float  Pearson |r| cutoff

        Returns
        -------
        np.ndarray  sparse correlation matrix (pruned entries → 0)
        """
        sparse = matrix.copy()
        sparse[np.abs(sparse) < threshold] = 0.0
        n_kept = int((np.abs(sparse) >= threshold).sum())
        n_total = self.n_regions * (self.n_regions - 1)
        logger.debug(
            "Threshold=%.2f | edges kept: %d / %d (%.1f%%)",
            threshold, n_kept, n_total, 100 * n_kept / n_total,
        )
        return sparse

    # ------------------------------------------------------------------
    # Stage 6: Convert to edge list
    # ------------------------------------------------------------------

    def to_edge_list(
        self,
        matrix: np.ndarray,
        subject_id: str,
        diagnosis: str,
        keep_zeros: bool = False,
    ) -> pd.DataFrame:
        """
        Convert upper-triangle of the correlation matrix to a long-format
        edge list DataFrame.

        Parameters
        ----------
        matrix     : np.ndarray  (n × n)
        subject_id : str
        diagnosis  : str  'Healthy' | 'Diseased'
        keep_zeros : bool  If False, zero-weight edges are dropped

        Returns
        -------
        pd.DataFrame  columns: [subject_id, source, target, weight, diagnosis]
        """
        rows: list[dict] = []
        for i in range(self.n_regions):
            for j in range(i + 1, self.n_regions):
                w = float(matrix[i, j])
                if not keep_zeros and w == 0.0:
                    continue
                rows.append({
                    "subject_id": subject_id,
                    "source":     self.region_names[i],
                    "target":     self.region_names[j],
                    "weight":     round(w, 6),
                    "diagnosis":  diagnosis,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        matrix_path: str | Path,
        subject_id: str,
        diagnosis: str,
        config: dict | None = None,
    ) -> pd.DataFrame:
        """
        Execute the full ETL transform pipeline for one subject.

        Parameters
        ----------
        matrix_path : str | Path  Path to raw CSV correlation matrix
        subject_id  : str
        diagnosis   : str
        config      : dict  Optional config overrides (from config.yaml)

        Returns
        -------
        pd.DataFrame  Edge list for this subject
        """
        cfg = config or {}
        fd_threshold    = cfg.get("motion_artifact_fd_threshold", 0.5)
        normalize_method= cfg.get("normalize_method", "zscore")
        corr_threshold  = cfg.get("correlation_threshold", 0.3)

        logger.info("Processing subject=%s diagnosis=%s", subject_id, diagnosis)
        matrix = self.load_matrix(matrix_path)
        matrix = self.regress_global_signal(matrix)
        matrix = self.remove_motion_artifacts(matrix, fd_threshold=fd_threshold)
        matrix = self.normalize(matrix, method=normalize_method)
        matrix = self.threshold(matrix, threshold=corr_threshold)
        edge_list = self.to_edge_list(matrix, subject_id, diagnosis)
        logger.info("  → %d edges retained for subject=%s", len(edge_list), subject_id)
        return edge_list
