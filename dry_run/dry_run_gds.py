"""
dry_run_gds.py
==============
GDS Feature Engineering smoke test (mock mode — no Neo4j required).

Author: basakanimesh14
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_gds_mock() -> bool:
    """
    Smoke test for GDSFeatureEngineer in mock mode.

    Passes driver=None to activate mock mode (returns dummy DataFrames).
    No Neo4j instance required.

    Returns
    -------
    bool  True if PASS
    """
    t0 = time.time()
    try:
        from neo4j_layer.gds.feature_projection import GDSFeatureEngineer

        # Instantiate with driver=None → mock mode
        engineer = GDSFeatureEngineer(driver=None, gds_graph_name="test_brain")

        # Run full GDS pipeline
        results = engineer.run_all_features()

        assert "projection"  in results, "Missing 'projection' key in results"
        assert "pageRank"    in results, "Missing 'pageRank' key in results"
        assert "degree"      in results, "Missing 'degree' key in results"
        assert "clustering"  in results, "Missing 'clustering' key in results"
        assert "betweenness" in results, "Missing 'betweenness' key in results"

        # Fetch node features
        features_df = engineer.fetch_node_features("test_subject")

        assert len(features_df) == 116, \
            f"Expected 116 rows (AAL regions), got {len(features_df)}"
        required_cols = {"atlasId", "anatomicalLabel", "degree", "pageRank", "clustering", "betweenness"}
        assert required_cols.issubset(set(features_df.columns)), \
            f"Missing columns: {required_cols - set(features_df.columns)}"
        assert features_df.select_dtypes(include="number").notna().all().all(), \
            "NaN values in numeric features"

        elapsed = time.time() - t0
        print(f"  ✅ GDS mock | 116 nodes | {len(results)} algorithms | {elapsed*1000:.1f}ms")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ GDS mock FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False
