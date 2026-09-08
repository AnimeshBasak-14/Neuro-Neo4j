"""
domain_shift.py
===============
Scanner-site domain shift analysis for the Neuro-Neo4j pipeline.

Different MRI hardware introduces systematic biases in functional
connectivity estimates (spectral noise, SNR variation). This module
quantifies inter-site distribution shift to ensure GNN generalisation.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DomainShiftAnalyzer:
    """
    Analyses feature distribution shifts across MRI scanner sites.

    Wraps scipy.stats.ks_2samp (Kolmogorov-Smirnov test) to flag
    features with statistically significant site-specific distributions.

    Parameters
    ----------
    scanner_site_col : str  Column name for scanner site labels
    """

    def __init__(self, scanner_site_col: str = "ScannerSite") -> None:
        self.scanner_site_col = scanner_site_col
        self._ks_results: Optional[pd.DataFrame] = None
        logger.info("DomainShiftAnalyzer initialised.")

    # ------------------------------------------------------------------
    # Site statistics
    # ------------------------------------------------------------------

    def compute_site_statistics(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean and std of each feature per scanner site.

        Parameters
        ----------
        features_df : pd.DataFrame
            Must contain the scanner_site_col and numeric feature columns.

        Returns
        -------
        pd.DataFrame  Multi-index (site, statistic) × feature
        """
        if self.scanner_site_col not in features_df.columns:
            raise ValueError(
                f"Column '{self.scanner_site_col}' not found in features_df. "
                f"Available: {list(features_df.columns)}"
            )
        feature_cols = [c for c in features_df.select_dtypes(include=np.number).columns
                        if c != self.scanner_site_col]

        stats = features_df.groupby(self.scanner_site_col)[feature_cols].agg(["mean", "std"])
        logger.info(
            "Site statistics computed for %d sites, %d features.",
            features_df[self.scanner_site_col].nunique(), len(feature_cols),
        )
        return stats

    # ------------------------------------------------------------------
    # KS test
    # ------------------------------------------------------------------

    def ks_test_across_sites(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run pairwise Kolmogorov-Smirnov tests across scanner sites.

        For each pair of sites (A, B) and each numeric feature, computes:
        - KS statistic
        - p-value
        - Significant (p < 0.05)

        Parameters
        ----------
        features_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame  columns: [feature, site_a, site_b, ks_stat, p_value, significant]
        """
        feature_cols = [c for c in features_df.select_dtypes(include=np.number).columns
                        if c != self.scanner_site_col]
        sites = features_df[self.scanner_site_col].unique()

        rows: list[dict] = []
        for feat in feature_cols:
            for i, site_a in enumerate(sites):
                for site_b in sites[i + 1:]:
                    vals_a = features_df[features_df[self.scanner_site_col] == site_a][feat].dropna()
                    vals_b = features_df[features_df[self.scanner_site_col] == site_b][feat].dropna()
                    if len(vals_a) < 2 or len(vals_b) < 2:
                        continue
                    stat, pval = ks_2samp(vals_a, vals_b)
                    rows.append({
                        "feature":     feat,
                        "site_a":      site_a,
                        "site_b":      site_b,
                        "ks_stat":     round(float(stat), 4),
                        "p_value":     round(float(pval), 4),
                        "significant": bool(pval < 0.05),
                    })

        self._ks_results = pd.DataFrame(rows)
        n_sig = int(self._ks_results["significant"].sum())
        logger.info(
            "KS tests complete: %d pairs tested, %d significant shifts detected.",
            len(rows), n_sig,
        )
        return self._ks_results

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_site_distributions(
        self,
        features_df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Plot feature distributions per scanner site as boxplots.

        Parameters
        ----------
        features_df : pd.DataFrame
        output_path : str | None  If provided, save figure to this path.
        """
        feature_cols = [c for c in features_df.select_dtypes(include=np.number).columns
                        if c != self.scanner_site_col][:6]  # max 6 features
        sites = features_df[self.scanner_site_col].unique()
        n_feats = len(feature_cols)

        if n_feats == 0:
            logger.warning("No numeric features to plot.")
            return

        fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 5))
        if n_feats == 1:
            axes = [axes]

        for ax, feat in zip(axes, feature_cols):
            data = [features_df[features_df[self.scanner_site_col] == s][feat].dropna() for s in sites]
            ax.boxplot(data, labels=sites)
            ax.set_title(feat, fontsize=10)
            ax.set_xlabel("Scanner Site")
            ax.set_ylabel("Value")
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Feature Distribution by Scanner Site (Domain Shift Analysis)", fontsize=12)
        plt.tight_layout()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Domain shift plot saved → %s", output_path)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Drift detection flag
    # ------------------------------------------------------------------

    def detect_significant_shift(self, p_threshold: float = 0.05) -> bool:
        """
        Returns True if any feature exhibits significant site-level shift.

        Parameters
        ----------
        p_threshold : float

        Returns
        -------
        bool
        """
        if self._ks_results is None:
            raise RuntimeError("Run ks_test_across_sites() first.")
        n_sig = int((self._ks_results["p_value"] < p_threshold).sum())
        detected = n_sig > 0
        logger.info(
            "Domain shift detected: %s (%d / %d feature-pairs significant at p<%.2f)",
            detected, n_sig, len(self._ks_results), p_threshold,
        )
        return detected

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_report(self) -> dict:
        """
        Generate a comprehensive domain shift report.

        Returns
        -------
        dict
        """
        if self._ks_results is None:
            return {"error": "Run ks_test_across_sites() before generating report."}

        n_total = len(self._ks_results)
        n_sig   = int(self._ks_results["significant"].sum())
        sig_features = self._ks_results[self._ks_results["significant"]]["feature"].unique().tolist()

        return {
            "n_pairs_tested":          n_total,
            "n_significant":           n_sig,
            "significant_features":    sig_features,
            "shift_detected":          n_sig > 0,
            "recommendation": (
                "Apply site harmonisation (ComBat / Z-score per site) before training."
                if n_sig > 0 else
                "No significant site shift detected — proceed without harmonisation."
            ),
            "details": self._ks_results.to_dict(orient="records"),
        }
