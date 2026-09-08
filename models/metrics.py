"""
metrics.py
==========
Evaluation metrics for the Neuro-Neo4j classification pipeline.

Metrics
-------
- Accuracy      : Correct / Total predictions
- Macro-F1      : Class-balanced harmonic mean of precision and recall
- ROC-AUC       : Area under Receiver Operating Characteristic curve
- Jaccard Index : Biomarker subgraph set-overlap stability (|A∩B|/|A∪B|)

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray  Ground-truth labels (int)
    y_pred : np.ndarray  Predicted labels (int)

    Returns
    -------
    float  Accuracy in [0, 1]
    """
    return float(accuracy_score(y_true, y_pred))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Macro-averaged F1 score.

    Macro-F1 is the primary clinical robustness metric: it treats each
    class equally regardless of support, penalising models that fail to
    identify the minority diseased class.

    Parameters
    ----------
    y_true : np.ndarray  Ground-truth labels
    y_pred : np.ndarray  Predicted labels

    Returns
    -------
    float  Macro-F1 in [0, 1]
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    multi_class: str = "ovr",
) -> float:
    """
    Compute ROC-AUC.

    For binary classification, y_scores should be the probability of the
    positive (Diseased) class.

    Parameters
    ----------
    y_true      : np.ndarray  Ground-truth labels
    y_scores    : np.ndarray  Predicted scores or probabilities.
                  Shape (n_samples,) for binary or (n_samples, n_classes) for OVR.
    multi_class : str  Strategy for multi-class ('ovr' or 'ovo')

    Returns
    -------
    float  ROC-AUC in [0, 1]
    """
    n_classes = len(np.unique(y_true))
    try:
        if n_classes == 2:
            # Binary: accept both (n,) and (n, 2) shapes
            if y_scores.ndim == 2:
                y_scores = y_scores[:, 1]
            return float(roc_auc_score(y_true, y_scores))
        else:
            return float(
                roc_auc_score(y_true, y_scores, multi_class=multi_class, average="macro")
            )
    except Exception as exc:
        logger.warning("ROC-AUC computation failed: %s — returning 0.5", exc)
        return 0.5


def compute_jaccard(set_a: set, set_b: set) -> float:
    """
    Compute the Jaccard Index between two sets.

    Used to measure the stability of biomarker subgraphs identified by
    GNNExplainer across different runs or model checkpoints.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    set_a : set  First set of items (e.g., edge indices)
    set_b : set  Second set of items

    Returns
    -------
    float  Jaccard Index in [0, 1]. Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Composite metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    biomarker_set_a: Optional[set] = None,
    biomarker_set_b: Optional[set] = None,
) -> dict:
    """
    Compute the full suite of evaluation metrics.

    Parameters
    ----------
    y_true          : np.ndarray  Ground-truth labels
    y_pred          : np.ndarray  Predicted labels
    y_scores        : np.ndarray  Predicted probabilities (n, 2) or (n,)
    biomarker_set_a : Optional[set]  First set for Jaccard stability
    biomarker_set_b : Optional[set]  Second set for Jaccard stability

    Returns
    -------
    dict with keys: accuracy, macro_f1, roc_auc, jaccard
    """
    y_true   = np.asarray(y_true)
    y_pred   = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    metrics = {
        "accuracy":  compute_accuracy(y_true, y_pred),
        "macro_f1":  compute_macro_f1(y_true, y_pred),
        "roc_auc":   compute_roc_auc(y_true, y_scores),
    }

    if biomarker_set_a is not None and biomarker_set_b is not None:
        metrics["jaccard"] = compute_jaccard(biomarker_set_a, biomarker_set_b)
    else:
        metrics["jaccard"] = None

    logger.info(
        "Metrics — Acc: %.4f | Macro-F1: %.4f | ROC-AUC: %.4f | Jaccard: %s",
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["roc_auc"],
        f"{metrics['jaccard']:.4f}" if metrics["jaccard"] is not None else "N/A",
    )
    return metrics


# ---------------------------------------------------------------------------
# MetricsReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricsReport:
    """
    Structured container for evaluation metrics.

    Attributes
    ----------
    accuracy  : float  Overall accuracy
    macro_f1  : float  Macro-averaged F1 score (primary metric)
    roc_auc   : float  Area under ROC curve
    jaccard   : float | None  Biomarker subgraph Jaccard stability
    split     : str    Which split this report corresponds to (train/val/test)
    n_samples : int    Number of samples evaluated
    extra     : dict   Any additional metrics (e.g., per-class F1)
    """
    accuracy:  float
    macro_f1:  float
    roc_auc:   float
    jaccard:   Optional[float] = None
    split:     str  = "test"
    n_samples: int  = 0
    extra:     dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to plain dictionary."""
        return {
            "split":     self.split,
            "n_samples": self.n_samples,
            "accuracy":  round(self.accuracy, 4),
            "macro_f1":  round(self.macro_f1, 4),
            "roc_auc":   round(self.roc_auc, 4),
            "jaccard":   round(self.jaccard, 4) if self.jaccard is not None else None,
            **self.extra,
        }

    def __str__(self) -> str:
        lines = [
            f"── MetricsReport [{self.split.upper()} | n={self.n_samples}] ──",
            f"  Accuracy  : {self.accuracy:.4f}",
            f"  Macro-F1  : {self.macro_f1:.4f}  ← primary clinical metric",
            f"  ROC-AUC   : {self.roc_auc:.4f}",
            f"  Jaccard   : {self.jaccard:.4f}" if self.jaccard else "  Jaccard   : N/A",
        ]
        if self.extra:
            for k, v in self.extra.items():
                lines.append(f"  {k:<10}: {v}")
        return "\n".join(lines)

    @classmethod
    def from_arrays(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        split: str = "test",
        biomarker_set_a: Optional[set] = None,
        biomarker_set_b: Optional[set] = None,
    ) -> "MetricsReport":
        """
        Convenience constructor from raw prediction arrays.

        Parameters
        ----------
        y_true          : Ground truth labels
        y_pred          : Predicted labels
        y_scores        : Predicted probabilities
        split           : Split name (train/val/test)
        biomarker_set_a : Optional first biomarker set for Jaccard
        biomarker_set_b : Optional second biomarker set for Jaccard

        Returns
        -------
        MetricsReport
        """
        d = compute_all_metrics(y_true, y_pred, y_scores, biomarker_set_a, biomarker_set_b)
        return cls(
            accuracy  = d["accuracy"],
            macro_f1  = d["macro_f1"],
            roc_auc   = d["roc_auc"],
            jaccard   = d["jaccard"],
            split     = split,
            n_samples = len(y_true),
        )
