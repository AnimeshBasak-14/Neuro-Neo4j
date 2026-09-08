"""
explain.py
==========
/explain endpoint: GNNExplainer biomarker subgraph extraction.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
import time

import torch
from fastapi import APIRouter, Depends, HTTPException, status

from api.neo4j_client import Neo4jClient, get_neo4j_client
from api.schemas import (
    ExplanationEdge,
    ExplanationRegion,
    ExplanationRequest,
    ExplanationResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/explain", tags=["Explainability"])


def _get_model():
    """Lazy-load the GNN model (reuses predict router's singleton if available)."""
    try:
        from api.routers.predict import _load_model
        return _load_model()
    except Exception as exc:
        logger.error("Could not load model for explainer: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model load failed: {exc}",
        )


@router.post(
    "",
    response_model=ExplanationResponse,
    summary="Extract biomarker subgraph via GNNExplainer",
)
def explain(
    request: ExplanationRequest,
    neo4j:   Neo4jClient = Depends(get_neo4j_client),
) -> ExplanationResponse:
    """
    Identify the brain connections and regions most responsible
    for a 'Diseased' classification decision.

    Steps
    -----
    1. Retrieve subject graph from Neo4j (or mock)
    2. Build PyG Data object
    3. Run BrainGNNExplainer
    4. Return top-k edges and regions with importance scores
    """
    from models.gnn.gcn_gat_model import BrainGNNModel
    from models.gnn.explainer import BrainGNNExplainer
    from torch_geometric.data import Data

    model  = _get_model()
    device = next(model.parameters()).device

    # Fetch graph
    try:
        graph_data = neo4j.get_subject_graph(request.subject_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    edges = graph_data["edges"]
    nodes = graph_data["nodes"]
    region_names = [n["anatomicalLabel"] for n in nodes]
    node_to_idx  = {r: i for i, r in enumerate(region_names)}
    n_nodes = len(region_names)

    # Build tensors
    src_list, tgt_list, wgt_list = [], [], []
    for e in edges:
        i = node_to_idx.get(e["source"], -1)
        j = node_to_idx.get(e["target"], -1)
        if i >= 0 and j >= 0:
            src_list += [i, j]; tgt_list += [j, i]; wgt_list += [e["weight"], e["weight"]]

    x          = torch.zeros((n_nodes, 5), dtype=torch.float32)
    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long) if src_list else torch.zeros((2, 0), dtype=torch.long)
    edge_attr  = torch.tensor([[w] for w in wgt_list], dtype=torch.float32) if wgt_list else torch.zeros((0, 1))
    batch      = torch.zeros(n_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, num_nodes=n_nodes)
    data = data.to(device)

    # Run explainer
    explainer_config = {
        "explainer": {
            "epochs": 100,   # Reduced for API responsiveness
            "lr": 0.01,
            "edge_mask_threshold": 0.5,
            "top_k_edges": request.top_k,
            "top_k_regions": 10,
        }
    }

    t0 = time.time()
    try:
        explainer = BrainGNNExplainer(model=model, config=explainer_config)
        result    = explainer.explain_graph(
            data=data,
            target_class=request.target_class,
            region_names=region_names,
        )
    except Exception as exc:
        logger.error("Explanation failed for subject=%s: %s", request.subject_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation error: {exc}",
        )
    elapsed_ms = (time.time() - t0) * 1000

    top_edges = [
        ExplanationEdge(**{k: row[k] for k in ("rank", "source", "target", "importance")})
        for _, row in result.top_edges_df.iterrows()
    ]
    top_regions = [
        ExplanationRegion(**{k: row[k] for k in ("rank", "region", "mean_importance")})
        for _, row in result.top_regions_df.iterrows()
    ]

    return ExplanationResponse(
        subject_id=request.subject_id,
        target_class=request.target_class,
        label="Diseased" if request.target_class == 1 else "Healthy",
        top_edges=top_edges,
        top_regions=top_regions,
        explanation_time_ms=round(elapsed_ms, 2),
    )
