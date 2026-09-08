"""
subject_split.py
================
Subject-level train / validation / test split.

CRITICAL: Splits must be at the SUBJECT level, not at the node or
edge level. Splitting at finer granularity would allow the model to
see parts of a patient's brain during training and other parts during
testing, leaking identity information and invalidating generalisation.

Author: basakanimesh14
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def subject_level_split(
    subjects_df:    pd.DataFrame,
    ratios:         tuple[float, float, float] = (0.70, 0.15, 0.15),
    stratify_col:   str = "Diagnosis",
    random_seed:    int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split subjects into train / validation / test sets at the subject level.

    Stratification preserves class distribution (Healthy/Diseased ratio)
    across all three splits.

    Parameters
    ----------
    subjects_df  : pd.DataFrame
        Must contain at least [PatientID, Diagnosis] columns.
    ratios       : tuple[float, float, float]
        Proportions for (train, val, test). Must sum to 1.0.
    stratify_col : str
        Column to stratify on (default: 'Diagnosis').
    random_seed  : int

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame  (subject-level)

    Raises
    ------
    ValueError  If ratios don't sum to 1.0 or subjects are insufficient.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios):.4f}"
    train_ratio, val_ratio, test_ratio = ratios

    n = len(subjects_df)
    if n < 10:
        raise ValueError(f"Insufficient subjects for splitting (n={n}, minimum 10 required).")

    stratify = subjects_df[stratify_col] if stratify_col in subjects_df.columns else None

    # Step 1: split off the test set
    val_test_ratio = val_ratio + test_ratio
    train_df, valtest_df = train_test_split(
        subjects_df,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=stratify,
    )

    # Step 2: split val/test from the held-out pool
    val_from_valtest = val_ratio / val_test_ratio
    stratify_valtest = valtest_df[stratify_col] if stratify_col in valtest_df.columns else None
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=(1.0 - val_from_valtest),
        random_state=random_seed,
        stratify=stratify_valtest,
    )

    # Log split details
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df[stratify_col].value_counts().to_dict() if stratify_col in df.columns else {}
        logger.info(
            "%-6s split: %3d subjects | %s",
            name, len(df), dist,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    id_col:   str = "PatientID",
) -> bool:
    """
    Assert that no subject appears in more than one split.

    This is the primary safeguard against data leakage in graph-level
    classification tasks. A single subject's brain data must be entirely
    in one split — never distributed across splits.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
    id_col                    : str  Subject identifier column

    Returns
    -------
    bool  True if no leakage detected

    Raises
    ------
    AssertionError  If any subject appears in multiple splits
    """
    train_ids = set(train_df[id_col].tolist())
    val_ids   = set(val_df[id_col].tolist())
    test_ids  = set(test_df[id_col].tolist())

    train_val_overlap  = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap   = val_ids & test_ids

    assert not train_val_overlap,  f"Data leakage! Train/Val overlap: {train_val_overlap}"
    assert not train_test_overlap, f"Data leakage! Train/Test overlap: {train_test_overlap}"
    assert not val_test_overlap,   f"Data leakage! Val/Test overlap: {val_test_overlap}"

    total = len(train_ids) + len(val_ids) + len(test_ids)
    logger.info(
        "✅ No data leakage detected. Total subjects: %d "
        "(train=%d, val=%d, test=%d)",
        total, len(train_ids), len(val_ids), len(test_ids),
    )
    return True


def get_split_statistics(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    label_col: str = "Diagnosis",
) -> dict:
    """
    Return detailed statistics for each split.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
    label_col                 : str

    Returns
    -------
    dict  with keys 'train', 'val', 'test', each containing:
          {n_subjects, class_distribution, healthy_pct, diseased_pct}
    """
    def _stats(df: pd.DataFrame) -> dict:
        n = len(df)
        dist = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
        h = dist.get("Healthy",  0)
        d = dist.get("Diseased", 0)
        return {
            "n_subjects":          n,
            "class_distribution":  dist,
            "healthy_pct":         round(100 * h / max(n, 1), 1),
            "diseased_pct":        round(100 * d / max(n, 1), 1),
        }

    stats = {
        "train": _stats(train_df),
        "val":   _stats(val_df),
        "test":  _stats(test_df),
        "total": len(train_df) + len(val_df) + len(test_df),
    }
    logger.info("Split statistics: %s", stats)
    return stats
