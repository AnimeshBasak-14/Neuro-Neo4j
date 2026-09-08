"""
schemas.py
==========
Pydantic v2 request/response schemas for the Neuro-Neo4j FastAPI service.

Author: basakanimesh14
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared models
# ---------------------------------------------------------------------------

class EdgeInput(BaseModel):
    """A single edge in the connectome graph."""
    source: str = Field(..., description="Source brain region label (AAL-116)")
    target: str = Field(..., description="Target brain region label (AAL-116)")
    weight: float = Field(..., ge=-1.0, le=1.0, description="Pearson correlation weight")

    model_config = {
        "json_schema_extra": {
            "examples": [{"source": "Hippocampus_L", "target": "Hippocampus_R", "weight": 0.72}]
        }
    }


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class ConnectomeInput(BaseModel):
    """Input payload for a single subject classification request."""
    subject_id:    str             = Field(..., description="Unique patient identifier")
    edge_list:     list[EdgeInput] = Field(..., description="List of functional edges")
    node_features: Optional[dict[str, list[float]]] = Field(
        None,
        description="Optional pre-computed node features {region_label: [f1,f2,f3,f4,f5]}"
    )

    @field_validator("edge_list")
    @classmethod
    def validate_edge_list_not_empty(cls, v: list[EdgeInput]) -> list[EdgeInput]:
        if not v:
            raise ValueError("edge_list must contain at least one edge.")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "subject_id": "sub_0001",
                "edge_list": [
                    {"source": "Hippocampus_L", "target": "Hippocampus_R", "weight": 0.72},
                    {"source": "Frontal_Sup_L", "target": "Frontal_Sup_R", "weight": 0.45},
                ]
            }]
        }
    }


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""
    subject_id:          str   = Field(..., description="Patient identifier")
    prediction:          int   = Field(..., description="Predicted class (0=Healthy, 1=Diseased)")
    label:               str   = Field(..., description="Human-readable label")
    probability_healthy: float = Field(..., ge=0.0, le=1.0)
    probability_diseased:float = Field(..., ge=0.0, le=1.0)
    inference_time_ms:   float = Field(..., description="Inference latency in milliseconds")
    model_version:       str   = Field(default="0.1.0")


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------

class ExplanationRequest(BaseModel):
    """Input payload for the /explain endpoint."""
    subject_id:   str = Field(..., description="Patient identifier to explain")
    target_class: int = Field(default=1, ge=0, le=1, description="Class to explain (default 1=Diseased)")
    top_k:        int = Field(default=20, ge=1, le=100, description="Number of top biomarker edges to return")


class ExplanationEdge(BaseModel):
    """A single important functional connection (biomarker edge)."""
    rank:       int
    source:     str
    target:     str
    importance: float


class ExplanationRegion(BaseModel):
    """A single important brain region."""
    rank:            int
    region:          str
    mean_importance: float


class ExplanationResponse(BaseModel):
    """Response from the /explain endpoint."""
    subject_id:         str
    target_class:       int
    label:              str
    top_edges:          list[ExplanationEdge]
    top_regions:        list[ExplanationRegion]
    explanation_time_ms:float
    jaccard_stability:  Optional[float] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status:         str  = Field(..., description="'healthy' or 'degraded'")
    version:        str  = Field(default="0.1.0")
    neo4j_connected:bool
    model_loaded:   bool
    timestamp:      str  = Field(default_factory=lambda: datetime.utcnow().isoformat())
    uptime_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    """Response from /metrics endpoint."""
    macro_f1:        float
    accuracy:        float
    roc_auc:         float
    n_predictions:   int
    drift_detected:  bool
    last_updated:    str
    baseline_f1:     Optional[float] = None
    drift_delta:     Optional[float] = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error response."""
    error:  str
    detail: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
