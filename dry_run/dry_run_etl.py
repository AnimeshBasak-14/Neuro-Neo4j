"""
dry_run_etl.py
==============
ETL Pipeline smoke test — no file I/O, no Neo4j required.

Author: basakanimesh14
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_etl_pipeline() -> bool:
    """
    Smoke test for the ConnectomeTransformer pipeline.

    Creates a synthetic 116×116 correlation matrix in-memory,
    runs the full ETL pipeline, and validates the output.

    Returns
    -------
    bool  True if PASS, False if FAIL
    """
    t0 = time.time()
    try:
        from neo4j_layer.etl.transform import ConnectomeTransformer
        from data.synthetic.generate_synthetic import AAL116_REGIONS

        region_names = AAL116_REGIONS  # 116 regions
        transformer  = ConnectomeTransformer(region_names=region_names)

        # Generate synthetic 116×116 correlation matrix
        rng    = np.random.default_rng(42)
        A      = rng.standard_normal((116, 116))
        cov    = A @ A.T + 116 * np.eye(116)
        d      = np.sqrt(np.diag(cov))
        matrix = (cov / np.outer(d, d)).astype(np.float32)

        # Run each pipeline stage
        matrix = transformer.regress_global_signal(matrix)
        assert matrix.shape == (116, 116), "regress_global_signal output shape mismatch"

        matrix = transformer.remove_motion_artifacts(matrix, fd_threshold=0.5)
        assert matrix.shape == (116, 116), "remove_motion_artifacts output shape mismatch"

        matrix = transformer.normalize(matrix, method="zscore")
        assert matrix.shape == (116, 116), "normalize output shape mismatch"

        matrix = transformer.threshold(matrix, threshold=0.3)
        assert matrix.shape == (116, 116), "threshold output shape mismatch"

        edge_list = transformer.to_edge_list(
            matrix, subject_id="sub_test", diagnosis="Healthy"
        )

        # Validate output
        required_cols = {"subject_id", "source", "target", "weight", "diagnosis"}
        assert required_cols.issubset(set(edge_list.columns)), \
            f"Missing columns: {required_cols - set(edge_list.columns)}"
        assert len(edge_list) > 0, "Edge list is empty after thresholding"
        assert (edge_list["subject_id"] == "sub_test").all(), "subject_id not set correctly"
        assert (edge_list["diagnosis"] == "Healthy").all(), "diagnosis not set correctly"
        assert edge_list["weight"].notna().all(), "Edge weights contain NaN"

        elapsed = time.time() - t0
        print(f"  ✅ ETL pipeline | {len(edge_list)} edges retained | {elapsed*1000:.1f}ms")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ ETL pipeline FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False
