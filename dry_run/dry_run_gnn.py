"""
dry_run_gnn.py
==============
GNN architecture smoke tests — no GPU, no training required.

Author: basakanimesh14
"""

from __future__ import annotations

import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_model_forward_pass() -> bool:
    """
    Smoke test: BrainGNNModel forward pass on random tensors.

    Creates a synthetic single-graph batch:
      - 20 nodes, each with 5 features
      - 60 edges (random)
    Verifies output shape is (1, 2) for binary classification.

    Returns
    -------
    bool  True if PASS
    """
    t0 = time.time()
    try:
        from models.gnn.gcn_gat_model import BrainGNNModel

        torch.manual_seed(42)
        model = BrainGNNModel(
            in_channels=5,
            hidden_channels=64,
            num_classes=2,
            num_heads=4,
            dropout=0.3,
            num_gcn_layers=2,
            num_gat_layers=1,
        )
        model.eval()

        N = 20
        x          = torch.randn(N, 5)
        edge_index = torch.randint(0, N, (2, 60))
        batch      = torch.zeros(N, dtype=torch.long)

        with torch.no_grad():
            logits = model(x, edge_index, batch=batch)

        assert logits.shape == (1, 2), \
            f"Expected output shape (1, 2), got {logits.shape}"
        assert not torch.isnan(logits).any(), "NaN in model output"

        n_params = model.count_parameters()
        elapsed  = time.time() - t0
        print(f"  ✅ GNN forward pass | shape={logits.shape} | params={n_params:,} | {elapsed*1000:.1f}ms")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ GNN forward pass FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False


def test_dataset_loading() -> bool:
    """
    Smoke test: BrainConnectomeDataset from mock CSV files.

    Creates minimal in-memory DataFrames (2 subjects, 10 regions each),
    saves to a temp directory, and loads the dataset.

    Returns
    -------
    bool  True if PASS
    """
    t0 = time.time()
    try:
        from models.gnn.dataset import BrainConnectomeDataset

        # Build mock data
        subjects_df = pd.DataFrame({
            "PatientID":   ["sub_0001", "sub_0002"],
            "Diagnosis":   ["Healthy",  "Diseased"],
            "Label":       [0,           1],
            "Age":         [60,          65],
            "ScannerSite": ["site_A",    "site_B"],
        })

        regions = [f"region_{i}" for i in range(10)]
        rng     = np.random.default_rng(42)
        edge_rows = []
        for subj, diag in [("sub_0001", "Healthy"), ("sub_0002", "Diseased")]:
            for i in range(10):
                for j in range(i + 1, 10):
                    w = float(rng.uniform(0.31, 0.95))
                    edge_rows.append({
                        "subject_id": subj,
                        "source":     regions[i],
                        "target":     regions[j],
                        "weight":     w,
                        "diagnosis":  diag,
                    })
        edge_df = pd.DataFrame(edge_rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            subj_path = Path(tmpdir) / "subjects.csv"
            edge_path = Path(tmpdir) / "edge_list.csv"
            subjects_df.to_csv(subj_path, index=False)
            edge_df.to_csv(edge_path, index=False)

            ds = BrainConnectomeDataset(
                root=tmpdir,
                subjects_csv=str(subj_path),
                edge_list_csv=str(edge_path),
            )

        assert len(ds) == 2, f"Expected 2 graphs, got {len(ds)}"
        assert ds[0].x.shape[1] == 5, f"Expected 5 node features, got {ds[0].x.shape[1]}"
        assert ds[0].y.shape == (1,), f"Expected label shape (1,), got {ds[0].y.shape}"

        elapsed = time.time() - t0
        print(f"  ✅ Dataset loading | {len(ds)} graphs | node_feats={ds[0].x.shape} | {elapsed*1000:.1f}ms")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ Dataset loading FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False


def test_trainer_dry_run() -> bool:
    """
    Smoke test: GNNTrainer.dry_run() — forward pass without backprop.

    Builds a minimal DataLoader with 2 synthetic Data objects
    and calls trainer.dry_run(loader).

    Returns
    -------
    bool  True if PASS
    """
    t0 = time.time()
    try:
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from models.gnn.gcn_gat_model import BrainGNNModel
        from models.gnn.train import GNNTrainer

        torch.manual_seed(42)
        device = torch.device("cpu")

        # Create 2 synthetic graphs
        def _make_data(label: int) -> Data:
            N = 15
            x          = torch.randn(N, 5)
            edge_index = torch.randint(0, N, (2, 40))
            edge_attr  = torch.rand(40, 1)
            y          = torch.tensor([label], dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=N)

        dataset = [_make_data(0), _make_data(1)]
        loader  = DataLoader(dataset, batch_size=2, shuffle=False)

        model     = BrainGNNModel(in_channels=5, hidden_channels=32, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        trainer   = GNNTrainer(model, optimizer, criterion, device, config={})

        trainer.dry_run(loader)

        elapsed = time.time() - t0
        print(f"  ✅ Trainer dry run | {elapsed*1000:.1f}ms")
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ❌ Trainer dry run FAILED: {exc} | {elapsed*1000:.1f}ms")
        import traceback; traceback.print_exc()
        return False
