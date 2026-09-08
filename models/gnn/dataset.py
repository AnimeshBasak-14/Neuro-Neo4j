"""
dataset.py
==========
PyTorch Geometric Dataset: BrainConnectomeDataset

Loads brain connectivity data from processed CSV files (offline / dry-run mode).
Fully compatible with PyG DataLoader without inheriting InMemoryDataset
(avoids PyG 2.8 init ordering issues with custom __init__ logic).

Each graph G represents one subject's functional connectome.

Node features (5-dim per node):
  [degree, pageRank, clustering_coeff, betweenness, eigenvector_centrality]

Edge features (1-dim):
  [pearson_correlation_weight]

Graph label:
  0 = Healthy
  1 = Diseased

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class BrainConnectomeDataset:
    """
    Custom PyG-compatible dataset for fMRI brain connectomes.

    Loads from CSV files (offline mode). Fully compatible with PyG
    DataLoader via __len__ + __getitem__ interface.

    Parameters
    ----------
    root            : str  (unused, kept for API compatibility)
    subjects_csv    : str  Path to subjects metadata CSV
    edge_list_csv   : str  Path to combined edge list CSV
    neo4j_driver    : neo4j.Driver | None
    transform       : callable | None
    pre_transform   : callable | None
    """

    def __init__(
        self,
        root: str,
        subjects_csv: str,
        edge_list_csv: str,
        neo4j_driver: Optional[Any] = None,
        transform=None,
        pre_transform=None,
    ) -> None:
        self.root          = root
        self.subjects_csv  = subjects_csv
        self.edge_list_csv = edge_list_csv
        self.neo4j_driver  = neo4j_driver
        self.transform     = transform
        self.pre_transform = pre_transform

        # Always initialise list BEFORE any loading
        self._data_list: list[Data] = []
        self._load_from_csv()

    # ------------------------------------------------------------------
    # PyG DataLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        data = self._data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def _load_from_csv(self) -> None:
        if not Path(self.subjects_csv).exists():
            logger.warning("subjects.csv not found at %s — dataset will be empty.", self.subjects_csv)
            return
        if not Path(self.edge_list_csv).exists():
            logger.warning("edge_list.csv not found at %s — dataset will be empty.", self.edge_list_csv)
            return

        subjects_df  = pd.read_csv(self.subjects_csv)
        edge_list_df = pd.read_csv(self.edge_list_csv)

        logger.info(
            "Loading dataset: %d subjects, %d total edges",
            len(subjects_df), len(edge_list_df),
        )

        loaded = 0
        for _, row in subjects_df.iterrows():
            subject_id = row["PatientID"]
            label      = int(row["Label"])
            subj_edges = edge_list_df[edge_list_df["subject_id"] == subject_id]

            if subj_edges.empty:
                logger.debug("No edges for subject=%s — skipping", subject_id)
                continue

            try:
                data = self._build_pyg_data(subj_edges, label, subject_id)
                self._data_list.append(data)
                loaded += 1
            except Exception as exc:
                logger.error("Failed to build graph for subject=%s: %s", subject_id, exc)

        logger.info("Dataset loaded: %d graphs", loaded)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_pyg_data(
        self,
        subj_edges: pd.DataFrame,
        label: int,
        subject_id: str,
    ) -> Data:
        sources = subj_edges["source"].values.tolist()
        targets = subj_edges["target"].values.tolist()
        all_nodes = list(dict.fromkeys(sources + targets))
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}
        n_nodes = len(all_nodes)

        G = nx.Graph()
        for _, row in subj_edges.iterrows():
            w = float(row["weight"])
            if w != 0.0:
                G.add_edge(row["source"], row["target"], weight=abs(w))
        for n in all_nodes:
            if n not in G:
                G.add_node(n)

        x = self._compute_node_features(G, all_nodes)
        edge_index, edge_attr = self._build_edge_index(subj_edges, node_to_idx)

        data = Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            y          = torch.tensor([label], dtype=torch.long),
            subject_id = subject_id,
            num_nodes  = n_nodes,
        )
        return data

    def _compute_node_features(
        self, G: nx.Graph, node_order: list[str]
    ) -> torch.Tensor:
        n = len(node_order)
        feats = np.zeros((n, 5), dtype=np.float32)

        if G.number_of_nodes() == 0:
            return torch.tensor(feats)

        max_deg = max(dict(G.degree()).values()) if G.number_of_edges() > 0 else 1
        degrees = {node: G.degree(node) / max(max_deg, 1) for node in G.nodes()}

        try:
            page_rank = nx.pagerank(G, weight="weight")
        except Exception:
            page_rank = {nd: 1.0 / max(G.number_of_nodes(), 1) for nd in G.nodes()}

        clustering = nx.clustering(G, weight="weight")

        try:
            betweenness = nx.betweenness_centrality(G, normalized=True, weight="weight")
        except Exception:
            betweenness = {nd: 0.0 for nd in G.nodes()}

        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
            max_ev = max(eigenvector.values()) if eigenvector else 1.0
            eigenvector = {k: v / max(max_ev, 1e-9) for k, v in eigenvector.items()}
        except Exception:
            eigenvector = degrees

        for i, node in enumerate(node_order):
            feats[i, 0] = degrees.get(node, 0.0)
            feats[i, 1] = page_rank.get(node, 0.0)
            feats[i, 2] = clustering.get(node, 0.0)
            feats[i, 3] = betweenness.get(node, 0.0)
            feats[i, 4] = eigenvector.get(node, 0.0)

        return torch.tensor(feats, dtype=torch.float32)

    def _build_edge_index(
        self,
        subj_edges: pd.DataFrame,
        node_to_idx: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_list: list[int] = []
        tgt_list: list[int] = []
        wgt_list: list[float] = []

        for _, row in subj_edges.iterrows():
            i = node_to_idx.get(row["source"], -1)
            j = node_to_idx.get(row["target"], -1)
            if i < 0 or j < 0:
                continue
            w = float(row["weight"])
            src_list += [i, j]
            tgt_list += [j, i]
            wgt_list += [w, w]

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1), dtype=torch.float32)

        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_attr  = torch.tensor([[w] for w in wgt_list], dtype=torch.float32)
        return edge_index, edge_attr

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        if not self._data_list:
            print("Dataset is empty.")
            return
        labels        = [int(d.y.item()) for d in self._data_list]
        n_nodes_list  = [d.num_nodes for d in self._data_list]
        n_edges_list  = [d.edge_index.shape[1] // 2 for d in self._data_list]
        print("=" * 55)
        print("BrainConnectomeDataset Summary")
        print("=" * 55)
        print(f"  Total graphs    : {len(self._data_list)}")
        print(f"  Healthy (0)     : {labels.count(0)}")
        print(f"  Diseased (1)    : {labels.count(1)}")
        print(f"  Avg nodes/graph : {np.mean(n_nodes_list):.1f}")
        print(f"  Avg edges/graph : {np.mean(n_edges_list):.1f}")
        print(f"  Node feat. dim  : {self._data_list[0].x.shape[1]}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Quick validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("BrainConnectomeDataset - Dry Run")

    subjects_df = pd.DataFrame({
        "PatientID": ["sub_0001", "sub_0002"],
        "Diagnosis": ["Healthy", "Diseased"],
        "Label":     [0, 1],
        "Age":       [60, 65],
        "ScannerSite": ["site_A", "site_B"],
    })
    regions   = [f"region_{i}" for i in range(10)]
    edge_rows = []
    rng = np.random.default_rng(42)
    for subj, diag in [("sub_0001", "Healthy"), ("sub_0002", "Diseased")]:
        for i in range(10):
            for j in range(i + 1, 10):
                w = float(rng.uniform(0.3, 1.0))
                edge_rows.append({"subject_id": subj, "source": regions[i],
                                  "target": regions[j], "weight": w, "diagnosis": diag})
    edge_list_df = pd.DataFrame(edge_rows)

    with tempfile.TemporaryDirectory() as tmpdir:
        subj_path = Path(tmpdir) / "subjects.csv"
        edge_path = Path(tmpdir) / "edge_list.csv"
        subjects_df.to_csv(subj_path, index=False)
        edge_list_df.to_csv(edge_path, index=False)

        ds = BrainConnectomeDataset(
            root=tmpdir,
            subjects_csv=str(subj_path),
            edge_list_csv=str(edge_path),
        )
        ds.summary()

        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch  = next(iter(loader))
        print(f"  Batch x shape   : {batch.x.shape}")
        print(f"  Batch y         : {batch.y}")
        print("[PASS] BrainConnectomeDataset dry run PASSED")
