"""
dry_run_baselines.py
====================
Baseline model smoke tests — no training, no data required.

Author: basakanimesh14
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MOCK_CONFIG = {
    "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced", "probability": True},
    "random_forest": {"n_estimators": 200, "max_depth": None, "class_weight": "balanced_subsample", "random_state": 42},
    "node2vec": {
        "dimensions": 64, "walk_length": 30, "num_walks": 200,
        "p": 1.0, "q": 1.0, "workers": 4, "seed": 42,
        "logistic_regression": {"max_iter": 1000, "class_weight": "balanced"},
    },
}


def test_svm_dry_run() -> bool:
    """Smoke test: SVMBaseline instantiation and dry_run()."""
    t0 = time.time()
    try:
        from models.baselines.svm_baseline import SVMBaseline
        model = SVMBaseline(config=MOCK_CONFIG)
        assert model.kernel == "rbf"
        assert model.C == 1.0
        assert not model._is_fitted
        model.dry_run()
        elapsed = time.time() - t0
        print(f"  ✅ SVM baseline | {elapsed*1000:.1f}ms")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ SVM baseline FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False


def test_rf_gdv_dry_run() -> bool:
    """Smoke test: RFGDVBaseline instantiation and dry_run()."""
    t0 = time.time()
    try:
        from models.baselines.rf_gdv_baseline import RFGDVBaseline, compute_gdv_features
        import networkx as nx
        import numpy as np

        model = RFGDVBaseline(config=MOCK_CONFIG)
        assert model.n_estimators == 200
        assert not model._is_fitted
        model.dry_run()

        # Also test GDV feature computation on a small graph
        G = nx.erdos_renyi_graph(20, 0.4, seed=42)
        gdv = compute_gdv_features(G, n_regions=20)
        assert gdv.shape == (20, 5), f"Expected GDV shape (20, 5), got {gdv.shape}"
        assert not np.isnan(gdv).any(), "GDV contains NaN values"

        elapsed = time.time() - t0
        print(f"  ✅ RF-GDV baseline | GDV shape={gdv.shape} | {elapsed*1000:.1f}ms")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ RF-GDV baseline FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False


def test_node2vec_dry_run() -> bool:
    """Smoke test: Node2VecBaseline instantiation and dry_run()."""
    t0 = time.time()
    try:
        from models.baselines.node2vec_baseline import Node2VecBaseline
        model = Node2VecBaseline(config=MOCK_CONFIG)
        assert model.dimensions == 64
        assert not model._is_fitted
        model.dry_run()
        elapsed = time.time() - t0
        print(f"  ✅ node2vec baseline | {elapsed*1000:.1f}ms")
        return True
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ node2vec baseline FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False
