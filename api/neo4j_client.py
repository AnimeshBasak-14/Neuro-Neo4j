"""
neo4j_client.py
===============
Neo4j driver wrapper for the Neuro-Neo4j FastAPI service.

Provides connection management, query helpers, and a mock mode for
dry-run / CI environments where Neo4j is unavailable.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Thin wrapper around the neo4j.Driver for the Neuro-Neo4j service.

    Parameters
    ----------
    uri      : str  Bolt URI, e.g. bolt://localhost:7687
    user     : str
    password : str
    database : str  (default 'neo4j')
    """

    def __init__(
        self,
        uri:      str,
        user:     str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self.uri      = uri
        self.user     = user
        self.database = database
        self._driver  = None
        self._mock    = False

        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            logger.info("Neo4j connected: %s", uri)
        except Exception as exc:
            logger.warning(
                "Neo4j connection failed (%s) — switching to mock mode.", exc
            )
            self._mock = True

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver is not None:
            try:
                self._driver.close()
                logger.info("Neo4j driver closed.")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """
        Test Neo4j connectivity.

        Returns
        -------
        bool  True if connected, False otherwise
        """
        if self._mock:
            return False
        try:
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1")
            return True
        except Exception as exc:
            logger.warning("Neo4j ping failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _run(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        if self._mock:
            logger.debug("[MOCK Neo4j] %s", query[:80])
            return []
        with self._driver.session(database=self.database) as session:
            result = session.run(query, parameters=params or {})
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Subject graph retrieval
    # ------------------------------------------------------------------

    def get_subject_graph(self, subject_id: str) -> dict:
        """
        Retrieve a subject's brain graph (nodes + edges) from Neo4j.

        Parameters
        ----------
        subject_id : str  Patient identifier

        Returns
        -------
        dict with keys: 'subject', 'nodes', 'edges'
        """
        if self._mock:
            rng = np.random.default_rng(42)
            n_nodes = 116
            nodes = [{"atlasId": i, "anatomicalLabel": f"region_{i}"} for i in range(n_nodes)]
            edges = [
                {"source": f"region_{i}", "target": f"region_{j}", "weight": float(rng.uniform(0.3, 0.9))}
                for i in range(10) for j in range(i+1, 10)
            ]
            return {
                "subject": {"patientId": subject_id, "diagnosis": "Unknown"},
                "nodes":   nodes,
                "edges":   edges,
            }

        subject_query = """
        MATCH (s:Subject {patientId: $subjectId})
        RETURN s.patientId AS patientId, s.diagnosis AS diagnosis, s.age AS age
        """
        subjects = self._run(subject_query, {"subjectId": subject_id})
        if not subjects:
            raise ValueError(f"Subject '{subject_id}' not found in Neo4j.")

        edge_query = """
        MATCH (src:BrainRegion)-[e:CONNECTED_TO {subjectId: $subjectId}]->(tgt:BrainRegion)
        RETURN src.anatomicalLabel AS source, tgt.anatomicalLabel AS target, e.weight AS weight
        """
        edges = self._run(edge_query, {"subjectId": subject_id})

        node_query = """
        MATCH (r:BrainRegion)
        RETURN r.atlasId AS atlasId, r.anatomicalLabel AS anatomicalLabel
        ORDER BY r.atlasId
        """
        nodes = self._run(node_query)

        return {"subject": subjects[0], "nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Node features retrieval
    # ------------------------------------------------------------------

    def get_node_features(self, subject_id: str) -> pd.DataFrame:
        """
        Retrieve GDS-computed node features for a subject.

        Parameters
        ----------
        subject_id : str

        Returns
        -------
        pd.DataFrame  columns: [atlasId, anatomicalLabel, degree, pageRank, clustering, betweenness]
        """
        if self._mock:
            rng = np.random.default_rng(42)
            return pd.DataFrame({
                "atlasId":         range(116),
                "anatomicalLabel": [f"region_{i}" for i in range(116)],
                "degree":          rng.random(116).tolist(),
                "pageRank":        rng.random(116).tolist(),
                "clustering":      rng.random(116).tolist(),
                "betweenness":     rng.random(116).tolist(),
            })

        query = """
        MATCH (r:BrainRegion)
        RETURN
            r.atlasId          AS atlasId,
            r.anatomicalLabel  AS anatomicalLabel,
            coalesce(r.degree,      0.0) AS degree,
            coalesce(r.pageRank,    0.0) AS pageRank,
            coalesce(r.clustering,  0.0) AS clustering,
            coalesce(r.betweenness, 0.0) AS betweenness
        ORDER BY r.atlasId
        """
        records = self._run(query)
        return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

_client: Optional[Neo4jClient] = None


def init_neo4j_client(uri: str, user: str, password: str, database: str = "neo4j") -> None:
    """Initialise the singleton Neo4j client (call at app startup)."""
    global _client
    _client = Neo4jClient(uri=uri, user=user, password=password, database=database)


def get_neo4j_client() -> Generator[Neo4jClient, None, None]:
    """FastAPI dependency that yields the shared Neo4j client."""
    global _client
    if _client is None:
        _client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="neuro_password",
        )
    yield _client


def close_neo4j_client() -> None:
    """Close the singleton client (call at app shutdown)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
