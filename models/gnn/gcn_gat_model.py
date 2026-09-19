"""
gcn_gat_model.py
================
Hybrid GCN + GAT Brain Classification Model with SwiGLU & GATv2.

Architecture
------------
1. Input linear projection  : (in_channels → hidden_channels)
2. GCN layers               : Global neighbourhood structure aggregation
   h_i^(l+1) = σ( Σ_{j∈N(i)} (1/c_ij) · W^(l) · h_j^(l) )
3. GATv2 / GAT layers       : Dynamic attention-weighted aggregation (multi-head)
   GATv2 fixes static attention limitation (Brody et al., 2021)
   α_ij = softmax(a^T LeakyReLU(W_l h_i + W_r h_j))
4. Global mean pooling      : Graph-level representation (B, hidden_channels)
5. SwiGLU classifier head   : SOTA gated feed-forward activation (Shazeer, 2020)
   SwiGLU(x) = (Swish(x W_gate) ⊗ (x W_up)) W_down
   Suppresses scanner noise & non-linear artifacts across sites.

Regularisation: BatchNorm, Dropout, Residual connections where dims match.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, global_mean_pool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SOTA Activation: SwiGLU (Swish Gated Linear Unit)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (Shazeer, 2020).

    SwiGLU(x) = (Swish(x W_gate + b_gate) ⊗ (x W_up + b_up)) W_down + b_down

    Multiplicative gating filters out noisy dimensional fluctuations while
    smooth non-monotonic gradients prevent dead neurons.

    Parameters
    ----------
    in_features     : int  Input dimension
    hidden_features : int  Intermediate gated projection dimension
    out_features    : int  Output projection dimension
    bias            : bool Whether to include bias terms (default True)
    """

    def __init__(
        self,
        in_features:     int,
        hidden_features: int,
        out_features:    int,
        bias:            bool = True,
    ) -> None:
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.w_up   = nn.Linear(in_features, hidden_features, bias=bias)
        self.w_down = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # F.silu is the official PyTorch implementation of Swish: x * sigmoid(x)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# BrainGNNModel
# ---------------------------------------------------------------------------

class BrainGNNModel(nn.Module):
    """
    Hybrid GCN + GAT model for graph-level brain disease classification.

    Parameters
    ----------
    in_channels      : int    Dimension of input node features (default 5)
    hidden_channels  : int    Hidden layer width (default 64)
    num_classes      : int    Number of output classes (default 2)
    num_heads        : int    Number of GAT attention heads (default 4)
    dropout          : float  Dropout probability (default 0.3)
    num_gcn_layers   : int    Number of GCN message-passing layers (default 2)
    num_gat_layers   : int    Number of GAT layers (default 1)
    use_swiglu       : bool   Enable SwiGLU gated activation in classifier (default True)
    use_gatv2        : bool   Use GATv2 dynamic attention instead of standard GAT (default True)
    """

    def __init__(
        self,
        in_channels:     int = 5,
        hidden_channels: int = 64,
        num_classes:     int = 2,
        num_heads:       int = 4,
        dropout:         float = 0.3,
        num_gcn_layers:  int = 2,
        num_gat_layers:  int = 1,
        use_swiglu:      bool = True,
        use_gatv2:       bool = True,
    ) -> None:
        super().__init__()

        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes     = num_classes
        self.num_heads       = num_heads
        self.dropout         = dropout
        self.num_gcn_layers  = num_gcn_layers
        self.num_gat_layers  = num_gat_layers
        self.use_swiglu      = use_swiglu
        self.use_gatv2       = use_gatv2

        # ----------------------------------------------------------
        # Input projection (maps raw features → hidden space)
        # ----------------------------------------------------------
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ----------------------------------------------------------
        # GCN layers (isotropic structural aggregation)
        # ----------------------------------------------------------
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns   = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_bns.append(nn.BatchNorm1d(hidden_channels))

        # ----------------------------------------------------------
        # GAT / GATv2 layers (multi-head dynamic attention)
        # ----------------------------------------------------------
        self.gat_convs = nn.ModuleList()
        self.gat_bns   = nn.ModuleList()
        self.gat_projs = nn.ModuleList()

        gat_cls = GATv2Conv if use_gatv2 else GATConv

        for _ in range(num_gat_layers):
            self.gat_convs.append(
                gat_cls(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            # After concat: num_heads * (hidden // num_heads) = hidden_channels
            self.gat_bns.append(nn.BatchNorm1d(hidden_channels))
            self.gat_projs.append(nn.Identity())

        # ----------------------------------------------------------
        # Classifier Head (SwiGLU SOTA Gated Unit or Standard MLP)
        # ----------------------------------------------------------
        if use_swiglu:
            self.classifier = nn.Sequential(
                SwiGLU(
                    in_features=hidden_channels,
                    hidden_features=hidden_channels,
                    out_features=hidden_channels // 2,
                    bias=True,
                ),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, num_classes),
            )

        self._init_weights()
        logger.info(
            "BrainGNNModel initialised | params=%d | SwiGLU=%s | GATv2=%s",
            self.count_parameters(),
            use_swiglu,
            use_gatv2,
        )

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        edge_attr:  Optional[Tensor] = None,
        batch:      Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through GCN → GATv2 → global pooling → SwiGLU MLP.

        Parameters
        ----------
        x          : Tensor  (N, in_channels)  Node features
        edge_index : Tensor  (2, E)            COO edge connectivity
        edge_attr  : Tensor  (E, 1)            Edge weights (optional)
        batch      : Tensor  (N,)              Graph assignment vector

        Returns
        -------
        Tensor  (B, num_classes)  Unnormalised logits per graph
        """
        # Input projection
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GCN layers with residual connection
        for conv, bn in zip(self.gcn_convs, self.gcn_bns):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual (same dim)

        # GATv2 / GAT layers with residual connection
        for conv, bn, proj in zip(self.gat_convs, self.gat_bns, self.gat_projs):
            h_new = conv(h, edge_index)   # concat heads → hidden_channels
            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # residual

        # Global mean pooling → graph-level representation
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_repr = global_mean_pool(h, batch)   # (B, hidden_channels)

        # Classifier head (SwiGLU)
        logits = self.classifier(graph_repr)       # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Attention weights extraction (for explainability)
    # ------------------------------------------------------------------

    def get_attention_weights(
        self,
        x:          Tensor,
        edge_index: Tensor,
    ) -> list[Tensor]:
        """
        Extract attention coefficients α_ij for all GAT/GATv2 layers.

        Parameters
        ----------
        x          : Tensor  (N, in_channels)
        edge_index : Tensor  (2, E)

        Returns
        -------
        list of Tensor  One (E, num_heads) attention tensor per GAT layer
        """
        h = F.relu(self.input_proj(x))

        for conv, bn in zip(self.gcn_convs, self.gcn_bns):
            h_new = F.relu(bn(conv(h, edge_index)))
            h = h + h_new

        attention_weights: list[Tensor] = []
        for conv, bn in zip(self.gat_convs, self.gat_bns):
            h_new, (_, alpha) = conv(h, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            h_new = F.elu(bn(h_new))
            h = h + h_new

        return attention_weights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a human-readable architecture summary string."""
        lines = [
            "BrainGNNModel Architecture (with SwiGLU & GATv2)",
            "=" * 50,
            f"  Input dim          : {self.in_channels}",
            f"  Hidden dim         : {self.hidden_channels}",
            f"  GCN layers         : {self.num_gcn_layers}",
            f"  GAT layers         : {self.num_gat_layers} ({'GATv2' if self.use_gatv2 else 'GAT'})",
            f"  GAT heads          : {self.num_heads}",
            f"  Activation (Head)  : {'SwiGLU' if self.use_swiglu else 'ReLU'}",
            f"  Dropout            : {self.dropout}",
            f"  Output classes     : {self.num_classes}",
            f"  Trainable params   : {self.count_parameters():,}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BrainGNNModel with SwiGLU & GATv2 — Dry Run")
    torch.manual_seed(42)

    model = BrainGNNModel(
        in_channels=5,
        hidden_channels=64,
        num_classes=2,
        num_heads=4,
        num_gcn_layers=2,
        num_gat_layers=1,
        use_swiglu=True,
        use_gatv2=True,
    )
    print(model.summary())

    x = torch.randn(20, 5)
    edge_index = torch.randint(0, 20, (2, 60))
    logits = model(x, edge_index)
    print(f"  Logits shape: {logits.shape} (Expected: [1, 2])")
    print(f"  Sample probabilities: {F.softmax(logits, dim=-1).tolist()}")
    print("  Model forward pass: SUCCESS")
