"""
health.py
=========
/health and /ready endpoints for the Neuro-Neo4j API.

Author: basakanimesh14
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from api.neo4j_client import Neo4jClient, get_neo4j_client
from api.schemas import HealthResponse

router = APIRouter(prefix="", tags=["Health"])
_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe — checks Neo4j connectivity and model availability",
)
def health_check(neo4j: Neo4jClient = Depends(get_neo4j_client)) -> HealthResponse:
    """
    Liveness probe.

    Returns 200 if the service is operational, 503 if any critical
    dependency (Neo4j, model) is unavailable.

    Checks
    ------
    - Neo4j reachability via driver.ping()
    - Existence of the model checkpoint file
    """
    neo4j_ok   = neo4j.ping()
    model_path = Path("checkpoints/best_model.pt")
    model_ok   = model_path.exists()
    uptime     = time.time() - _start_time

    overall_status = "healthy" if (neo4j_ok and model_ok) else "degraded"

    response = HealthResponse(
        status=overall_status,
        neo4j_connected=neo4j_ok,
        model_loaded=model_ok,
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=round(uptime, 1),
    )

    if overall_status == "degraded":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response


@router.get(
    "/ready",
    summary="Readiness probe — minimal check for load balancer integration",
)
def readiness_probe() -> dict:
    """
    Kubernetes / load balancer readiness probe.

    Always returns 200 if the process is alive.
    More detailed status is available at /health.
    """
    return {
        "status":    "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_s":  round(time.time() - _start_time, 1),
    }
