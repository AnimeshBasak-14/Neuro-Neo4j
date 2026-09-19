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
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def subject_level_split(
    subjects_df:    pd.DataFrame | None = None,
    ratios:         tuple[float, float, float] = (0.70, 0.15, 0.15),
    stratify_col:   str = "Diagnosis",
    random_seed:    int = 42,
    dataset:        Any = None,
    train_ratio:    float | None = None,
    val_ratio:      float | None = None,
    seed:           int | None = None,
) -> tuple[Any, Any, Any]:
    """
    Split subjects into train / validation / test sets at the subject level.

    Stratification preserves class distribution (Healthy/Diseased ratio)
    across all three splits.

    Supports either:
      1. subjects_df (DataFrame) -> returns (train_df, val_df, test_df)
      2. dataset (BrainConnectomeDataset) -> returns (train_idx, val_idx, test_idx)
    """
    if seed is not None:
        random_seed = seed

    if dataset is not None:
        if subjects_df is None:
            if hasattr(dataset, "subjects_csv") and Path(dataset.subjects_csv).exists():
                subjects_df = pd.read_csv(dataset.subjects_csv)
            else:
                # Extract from dataset objects
                subjects_df = pd.DataFrame([
                    {"PatientID": getattr(d, "subject_id", f"sub_{i}"),
                     "Diagnosis": "Diseased" if getattr(d, "y", torch.tensor([0])).item() == 1 else "Healthy"}
                    for i, d in enumerate(dataset)
                ])
        if train_ratio is not None and val_ratio is not None:
            test_ratio = max(0.0, round(1.0 - train_ratio - val_ratio, 6))
            ratios = (train_ratio, val_ratio, test_ratio)

    if subjects_df is None:
        raise ValueError("Either subjects_df or dataset must be provided to subject_level_split.")

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

    if dataset is not None:
        subject_to_idx = {getattr(d, "subject_id", f"sub_{i}"): i for i, d in enumerate(dataset)}
        id_col = "PatientID" if "PatientID" in subjects_df.columns else subjects_df.columns[0]
        train_idx = [subject_to_idx[sid] for sid in train_df[id_col] if sid in subject_to_idx]
        val_idx   = [subject_to_idx[sid] for sid in val_df[id_col] if sid in subject_to_idx]
        test_idx  = [subject_to_idx[sid] for sid in test_df[id_col] if sid in subject_to_idx]
        return train_idx, val_idx, test_idx

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
