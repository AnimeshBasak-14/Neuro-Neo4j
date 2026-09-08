"""
dry_run_explainer.py
====================
GNNExplainer smoke test — no trained model, no GPU required.

Author: basakanimesh14
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MOCK_EXPLAINER_CONFIG = {
    "explainer": {
        "epochs": 10,           # Very short for dry run
        "lr": 0.01,
        "edge_mask_threshold": 0.5,
        "top_k_edges": 10,
        "top_k_regions": 5,
    }
}


def test_explainer_dry_run() -> bool:
    """
    Smoke test: BrainGNNExplainer on dummy Data object.

    Creates an untrained BrainGNNModel and runs explainer.dry_run()
    on a synthetic single-graph Data object.

    Returns
    -------
    bool  True if PASS
    """
    t0 = time.time()
    try:
        from torch_geometric.data import Data
        from models.gnn.gcn_gat_model import BrainGNNModel
        from models.gnn.explainer import BrainGNNExplainer

        torch.manual_seed(42)

        # Untrained model
        model = BrainGNNModel(
            in_channels=5,
            hidden_channels=32,
            num_classes=2,
            num_heads=2,
            dropout=0.0,
            num_gcn_layers=1,
            num_gat_layers=1,
        )
        model.eval()

        # Synthetic Data
        N = 20
        x          = torch.randn(N, 5)
        edge_index = torch.randint(0, N, (2, 50))
        edge_attr  = torch.rand(50, 1)
        batch      = torch.zeros(N, dtype=torch.long)
        y          = torch.tensor([1], dtype=torch.long)  # Diseased

        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            batch=batch, y=y, num_nodes=N, subject_id="test_subject"
        )

        # Instantiate explainer and dry run
        explainer = BrainGNNExplainer(model=model, config=MOCK_EXPLAINER_CONFIG)
        explainer.dry_run(data)

        # Validate explanation result
        region_names = [f"AAL_{i}" for i in range(N)]
        result = explainer.explain_graph(data, target_class=1, region_names=region_names)

        assert result.edge_mask.shape[0] == edge_index.shape[1], \
            f"Edge mask size mismatch: {result.edge_mask.shape[0]} != {edge_index.shape[1]}"
        assert len(result.top_edges_df) > 0, "No top edges returned"
        assert len(result.top_regions_df) > 0, "No top regions returned"
        assert result.target_class == 1

        elapsed = time.time() - t0
        print(
            f"  ✅ GNNExplainer | api={explainer._api_mode} | "
            f"top_edges={len(result.top_edges_df)} | "
            f"top_regions={len(result.top_regions_df)} | {elapsed*1000:.1f}ms"
        )
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ GNNExplainer FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False
