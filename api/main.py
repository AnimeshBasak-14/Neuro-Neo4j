"""
main.py
=======
Neuro-Neo4j FastAPI Application.

Endpoints
---------
GET  /             → Swagger redirect
GET  /health       → Liveness probe
GET  /ready        → Readiness probe
GET  /metrics      → Performance metrics (Macro-F1, accuracy, AUC, drift)
GET  /metrics/prometheus → Prometheus-format export
POST /predict      → Single subject classification
POST /predict/batch → Batch classification
POST /explain      → GNNExplainer biomarker extraction

Author: basakanimesh14
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from api.neo4j_client import init_neo4j_client, close_neo4j_client
from api.routers import health, metrics_router, predict, explain
from api.schemas import ErrorResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuro_neo4j.api")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    logger.warning("config.yaml not found — using defaults.")
    return {}


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: connect Neo4j on startup, close on shutdown."""
    config = _load_config()
    neo4j_cfg = config.get("neo4j", {})

    uri      = os.getenv("NEO4J_URI",      neo4j_cfg.get("uri",      "bolt://localhost:7687"))
    user     = os.getenv("NEO4J_USER",     neo4j_cfg.get("user",     "neo4j"))
    password = os.getenv("NEO4J_PASSWORD", neo4j_cfg.get("password", "neuro_password"))
    database = os.getenv("NEO4J_DATABASE", neo4j_cfg.get("database", "neo4j"))

    logger.info("=== Neuro-Neo4j API starting up ===")
    init_neo4j_client(uri=uri, user=user, password=password, database=database)

    # Warm up model (loads checkpoint if available, else initialises fresh)
    try:
        from api.routers.predict import _load_model
        _load_model()
        logger.info("GNN model warm-up complete.")
    except Exception as exc:
        logger.warning("Model warm-up failed: %s", exc)

    logger.info("=== API ready on http://0.0.0.0:8000 ===")
    yield

    # Shutdown
    logger.info("Shutting down — closing Neo4j connection...")
    close_neo4j_client()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

config = _load_config()
api_cfg = config.get("api", {})

app = FastAPI(
    title=api_cfg.get("title", "Neuro-Neo4j Inference API"),
    description=api_cfg.get(
        "description",
        "GNN-based brain biomarker classification and explainability service for neurodegenerative disorders.",
    ),
    version=api_cfg.get("version", "0.1.0"),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_cfg.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(metrics_router.router)
app.include_router(predict.router)
app.include_router(explain.router)

# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to Swagger UI."""
    return RedirectResponse(url="/docs")

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError: %s", exc)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="ValueError", detail=str(exc)).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    logger.error("RuntimeError: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="RuntimeError", detail=str(exc)).model_dump(),
    )
