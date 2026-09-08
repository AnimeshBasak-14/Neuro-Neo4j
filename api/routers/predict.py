"""
predict.py
==========
/predict endpoint: GNN-based brain disease classification.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, HTTPException, status

from api.schemas import ConnectomeInput, PredictionResponse
from api.routers.metrics_router import get_metrics_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model: Optional[object] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    """Load or initialise the GNN model."""
    global _model
    if _model is not None:
        return _model

    from models.gnn.gcn_gat_model import BrainGNNModel
    model = BrainGNNModel(
        in_channels=5,
        hidden_channels=64,
        num_classes=2,
        num_heads=4,
        dropout=0.3,
        num_gcn_layers=2,
        num_gat_layers=1,
    )

    ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=_device)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Model loaded from checkpoint: %s", ckpt_path)
        except Exception as exc:
            logger.warning("Could not load checkpoint (%s) — using untrained model.", exc)
    else:
        logger.warning("No checkpoint found — using untrained model (dry run mode).")

    model.to(_device).eval()
    _model = model
    return _model


def _connectome_to_pyg(payload: ConnectomeInput):
    """Convert ConnectomeInput → torch_geometric.data.Data."""
    from torch_geometric.data import Data

    nodes = list(dict.fromkeys(
        [e.source for e in payload.edge_list] + [e.target for e in payload.edge_list]
    ))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Node features: use provided or zeros
    if payload.node_features:
        x_rows = []
        for node in nodes:
            feats = payload.node_features.get(node, [0.0] * 5)
            x_rows.append(feats[:5] + [0.0] * max(0, 5 - len(feats)))
        x = torch.tensor(x_rows, dtype=torch.float32)
    else:
        x = torch.zeros((n_nodes, 5), dtype=torch.float32)

    src_list, tgt_list, wgt_list = [], [], []
    for edge in payload.edge_list:
        i = node_to_idx[edge.source]
        j = node_to_idx[edge.target]
        src_list += [i, j]
        tgt_list += [j, i]
        wgt_list += [edge.weight, edge.weight]

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_attr  = torch.tensor([[w] for w in wgt_list], dtype=torch.float32)
    batch      = torch.zeros(n_nodes, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, num_nodes=n_nodes)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=PredictionResponse,
    summary="Classify a single brain connectome as Healthy or Diseased",
)
def predict(payload: ConnectomeInput) -> PredictionResponse:
    """
    Run GNN inference on a subject's functional connectivity graph.

    Steps
    -----
    1. Convert edge list → PyG Data object
    2. Forward pass through BrainGNNModel
    3. Softmax → class probabilities
    4. Return prediction + inference latency
    """
    model = _load_model()
    data  = _connectome_to_pyg(payload)

    t0 = time.time()
    try:
        with torch.no_grad():
            data = data.to(_device)
            logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]
    except Exception as exc:
        logger.error("Inference failed for subject=%s: %s", payload.subject_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )
    elapsed_ms = (time.time() - t0) * 1000
    pred = int(probs.argmax())

    # Record for drift monitoring (no ground truth yet, skip)
    return PredictionResponse(
        subject_id=payload.subject_id,
        prediction=pred,
        label="Diseased" if pred == 1 else "Healthy",
        probability_healthy=float(probs[0]),
        probability_diseased=float(probs[1]),
        inference_time_ms=round(elapsed_ms, 2),
    )


@router.post(
    "/batch",
    response_model=list[PredictionResponse],
    summary="Batch-classify multiple brain connectomes",
)
def predict_batch(payloads: list[ConnectomeInput]) -> list[PredictionResponse]:
    """
    Run inference on multiple subjects in sequence.

    For true batched GPU inference, load all subjects into a single
    DataLoader batch. This endpoint iterates sequentially for simplicity.
    """
    if not payloads:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request body must contain at least one ConnectomeInput.",
        )
    return [predict(p) for p in payloads]
