"""
explainer.py
============
GNNExplainer wrapper for biomarker subgraph extraction.

Identifies the specific edges (functional connections) and node features
(brain region attributes) that most strongly drive a 'Diseased'
classification decision.

Supports both the legacy GNNExplainer API and the modern Explainer API
introduced in PyG ≥2.4.0, with graceful fallback between them.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from models.metrics import compute_jaccard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExplanationResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExplanationResult:
    """
    Container for GNNExplainer output.

    Attributes
    ----------
    edge_mask       : Tensor  (E,)   Importance score per edge
    node_feat_mask  : Tensor  (N, F) Importance score per (node, feature)
    top_edges_df    : pd.DataFrame   Top-k important edges with region labels
    top_regions_df  : pd.DataFrame   Top-k important brain regions
    target_class    : int            Target class explained (1=Diseased)
    subject_id      : str            Subject identifier
    """
    edge_mask:      Tensor
    node_feat_mask: Tensor
    top_edges_df:   pd.DataFrame
    top_regions_df: pd.DataFrame
    target_class:   int
    subject_id:     str = "unknown"
    extra:          dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BrainGNNExplainer
# ---------------------------------------------------------------------------

class BrainGNNExplainer:
    """
    Wraps GNNExplainer to extract biomarker subgraphs from BrainGNNModel.

    Handles both:
    - Modern API: torch_geometric.explain.Explainer (PyG ≥ 2.4)
    - Legacy API: torch_geometric.nn.GNNExplainer  (PyG < 2.4)

    Parameters
    ----------
    model  : nn.Module  Trained BrainGNNModel
    config : dict       Explainer config from config.yaml
    """

    def __init__(self, model: nn.Module, config: dict) -> None:
        self.model  = model
        self.config = config
        expl_cfg = config.get("explainer", config)
        self.epochs             = expl_cfg.get("epochs", 200)
        self.lr                 = expl_cfg.get("lr", 0.01)
        self.edge_mask_threshold= expl_cfg.get("edge_mask_threshold", 0.5)
        self.top_k_edges        = expl_cfg.get("top_k_edges", 20)
        self.top_k_regions      = expl_cfg.get("top_k_regions", 10)
        self._explainer         = None
        self._api_mode          = None

        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialise explainer using the available PyG API."""
        # Try modern API first (PyG ≥ 2.4)
        try:
            from torch_geometric.explain import Explainer
            from torch_geometric.explain.algorithm import GNNExplainer as GNNExplainerAlgo
            self._explainer = Explainer(
                model=self.model,
                algorithm=GNNExplainerAlgo(epochs=self.epochs, lr=self.lr),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="multiclass_classification",
                    task_level="graph",
                    return_type="raw",
                ),
            )
            self._api_mode = "modern"
            logger.info("GNNExplainer initialised (modern PyG ≥ 2.4 API)")
        except ImportError:
            pass

        # Fallback to legacy API (PyG < 2.4)
        if self._explainer is None:
            try:
                from torch_geometric.nn import GNNExplainer as LegacyGNNExplainer
                self._explainer = LegacyGNNExplainer(
                    self.model,
                    epochs=self.epochs,
                    lr=self.lr,
                    return_type="log_prob",
                )
                self._api_mode = "legacy"
                logger.info("GNNExplainer initialised (legacy PyG < 2.4 API)")
            except ImportError:
                logger.warning(
                    "GNNExplainer not available. Explainer will return dummy masks."
                )
                self._api_mode = "mock"

    # ------------------------------------------------------------------
    # Core explanation
    # ------------------------------------------------------------------

    def explain_graph(
        self,
        data:         Data,
        target_class: int = 1,
        region_names: Optional[list[str]] = None,
    ) -> ExplanationResult:
        """
        Explain a graph prediction with GNNExplainer.

        Parameters
        ----------
        data         : torch_geometric.data.Data  Subject graph
        target_class : int  Class to explain (default 1 = Diseased)
        region_names : list[str] | None  Brain region labels for output

        Returns
        -------
        ExplanationResult
        """
        self.model.eval()
        subject_id = getattr(data, "subject_id", "unknown")

        if region_names is None:
            region_names = [f"region_{i}" for i in range(data.num_nodes)]

        # Get edge mask via appropriate API
        edge_mask, node_feat_mask = self._get_masks(data, target_class)

        # Top-k edges
        top_edges_df = self.get_top_k_edges(
            edge_mask, data.edge_index, k=self.top_k_edges, region_names=region_names
        )
        # Top regions
        top_regions_df = self.get_important_regions(
            node_feat_mask, region_names=region_names, k=self.top_k_regions
        )

        return ExplanationResult(
            edge_mask=edge_mask,
            node_feat_mask=node_feat_mask,
            top_edges_df=top_edges_df,
            top_regions_df=top_regions_df,
            target_class=target_class,
            subject_id=subject_id,
        )

    def _get_masks(
        self, data: Data, target_class: int
    ) -> tuple[Tensor, Tensor]:
        """Return (edge_mask, node_feat_mask) tensors."""
        n_edges = data.edge_index.shape[1]
        n_nodes = data.num_nodes
        n_feats = data.x.shape[1] if data.x is not None else 5

        if self._api_mode == "mock" or self._explainer is None:
            logger.warning("Using dummy explanation masks (explainer unavailable).")
            rng = np.random.default_rng(42)
            edge_mask     = torch.tensor(rng.random(n_edges), dtype=torch.float32)
            node_feat_mask= torch.tensor(rng.random((n_nodes, n_feats)), dtype=torch.float32)
            return edge_mask, node_feat_mask

        if self._api_mode == "modern":
            # NOTE: GNNExplainer runs its own internal optimization (backward passes)
            # so we must NOT use torch.no_grad() here.
            # NOTE: 'target' must NOT be passed for explanation_type='model'
            try:
                explanation = self._explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=torch.zeros(data.num_nodes, dtype=torch.long),
                )
                edge_mask      = explanation.edge_mask
                node_feat_mask = (
                    explanation.node_feat_mask
                    if hasattr(explanation, "node_feat_mask") and explanation.node_feat_mask is not None
                    else torch.zeros((n_nodes, n_feats))
                )
            except Exception as exc:
                logger.warning("Modern GNNExplainer failed (%s) — using dummy masks.", exc)
                rng = np.random.default_rng(42)
                edge_mask      = torch.tensor(rng.random(n_edges), dtype=torch.float32)
                node_feat_mask = torch.tensor(rng.random((n_nodes, n_feats)), dtype=torch.float32)
        else:
            # Legacy API — also needs gradients
            try:
                node_feat_mask, edge_mask = self._explainer.explain_graph(
                    x=data.x,
                    edge_index=data.edge_index,
                )
                node_feat_mask = node_feat_mask.float()
            except Exception as exc:
                logger.warning("Legacy GNNExplainer failed (%s) — using dummy masks.", exc)
                rng = np.random.default_rng(42)
                edge_mask      = torch.tensor(rng.random(n_edges), dtype=torch.float32)
                node_feat_mask = torch.tensor(rng.random((n_nodes, n_feats)), dtype=torch.float32)

        return edge_mask.detach(), node_feat_mask.detach()

    # ------------------------------------------------------------------
    # Top-K edges
    # ------------------------------------------------------------------

    def get_top_k_edges(
        self,
        edge_mask:   Tensor,
        edge_index:  Tensor,
        k:           int = 20,
        region_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract the top-k most important edges from the explanation mask.

        Parameters
        ----------
        edge_mask    : Tensor  (E,) importance scores
        edge_index   : Tensor  (2, E)
        k            : int
        region_names : list[str] | None

        Returns
        -------
        pd.DataFrame  columns: [rank, source, target, importance]
        """
        scores = edge_mask.detach().cpu().numpy()
        n_edges = len(scores)
        k = min(k, n_edges)

        top_idx = np.argsort(scores)[::-1][:k]
        src_nodes = edge_index[0].cpu().numpy()
        tgt_nodes = edge_index[1].cpu().numpy()

        rows = []
        for rank, idx in enumerate(top_idx, 1):
            src = int(src_nodes[idx])
            tgt = int(tgt_nodes[idx])
            src_name = region_names[src] if region_names and src < len(region_names) else f"r{src}"
            tgt_name = region_names[tgt] if region_names and tgt < len(region_names) else f"r{tgt}"
            rows.append({
                "rank":       rank,
                "source":     src_name,
                "target":     tgt_name,
                "source_idx": src,
                "target_idx": tgt,
                "importance": float(scores[idx]),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Important brain regions
    # ------------------------------------------------------------------

    def get_important_regions(
        self,
        node_feat_mask: Tensor,
        region_names:   Optional[list[str]] = None,
        k:              int = 10,
    ) -> pd.DataFrame:
        """
        Identify the most important brain regions by mean feature importance.

        Parameters
        ----------
        node_feat_mask : Tensor  (N, F)
        region_names   : list[str] | None
        k              : int

        Returns
        -------
        pd.DataFrame  columns: [rank, region, mean_importance]
        """
        mask = node_feat_mask.detach().cpu().numpy()
        if mask.ndim == 1:
            node_scores = mask
        else:
            node_scores = mask.mean(axis=1)

        n = len(node_scores)
        k = min(k, n)
        top_idx = np.argsort(node_scores)[::-1][:k]

        rows = []
        for rank, idx in enumerate(top_idx, 1):
            name = region_names[idx] if region_names and idx < len(region_names) else f"region_{idx}"
            rows.append({
                "rank":            rank,
                "region":          name,
                "node_idx":        int(idx),
                "mean_importance": float(node_scores[idx]),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Jaccard stability
    # ------------------------------------------------------------------

    def compute_jaccard_stability(
        self,
        explanation_a: ExplanationResult,
        explanation_b: ExplanationResult,
    ) -> float:
        """
        Compute Jaccard Index between two explanations' top-edge sets.

        Measures consistency of identified biomarker connections across
        two model runs or two subjects with the same diagnosis.

        Parameters
        ----------
        explanation_a : ExplanationResult
        explanation_b : ExplanationResult

        Returns
        -------
        float  Jaccard Index in [0, 1]
        """
        set_a = set(
            zip(
                explanation_a.top_edges_df["source"].tolist(),
                explanation_a.top_edges_df["target"].tolist(),
            )
        )
        set_b = set(
            zip(
                explanation_b.top_edges_df["source"].tolist(),
                explanation_b.top_edges_df["target"].tolist(),
            )
        )
        j = compute_jaccard(set_a, set_b)
        logger.info(
            "Jaccard stability (subj %s vs %s): %.4f",
            explanation_a.subject_id, explanation_b.subject_id, j,
        )
        return j

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def dry_run(self, data: Data) -> None:
        """
        Run explainer on dummy data and print mask shapes.

        Parameters
        ----------
        data : torch_geometric.data.Data
        """
        print("=" * 50)
        print("BrainGNNExplainer — Dry Run")
        print("=" * 50)
        print(f"  API mode         : {self._api_mode}")
        print(f"  Epochs           : {self.epochs}")
        print(f"  LR               : {self.lr}")
        print(f"  Edge threshold   : {self.edge_mask_threshold}")
        print(f"  Top-k edges      : {self.top_k_edges}")

        result = self.explain_graph(data, target_class=1)
        print(f"  Edge mask shape  : {result.edge_mask.shape}")
        print(f"  Node mask shape  : {result.node_feat_mask.shape}")
        print(f"  Top edges (n)    : {len(result.top_edges_df)}")
        print(f"  Top regions (n)  : {len(result.top_regions_df)}")
        print("=" * 50)
