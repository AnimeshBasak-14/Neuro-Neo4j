"""
feature_projection.py
=====================
Neo4j GDS (Graph Data Science) feature engineering for brain connectomes.

Computes high-order topological features prior to GNN training:
  - PageRank           → hub region detection
  - Degree Centrality  → local connectivity density
  - Clustering Coeff.  → local clique structure
  - Betweenness        → bridge region identification

All GDS operations use the ``gds.graph.project`` named-graph pattern.
Mock mode is activated when driver=None (for dry-run / CI environments).

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GDSFeatureEngineer:
    """
    Computes topological node features via Neo4j GDS algorithms.

    Parameters
    ----------
    driver         : neo4j.Driver | None
        Live Neo4j driver. If None, runs in mock mode (returns dummy data).
    gds_graph_name : str
        Name for the in-memory GDS graph projection.
    config         : dict
        GDS configuration section from config.yaml.
    """

    def __init__(
        self,
        driver: Any | None,
        gds_graph_name: str = "brain_connectome",
        config: dict | None = None,
    ) -> None:
        self.driver = driver
        self.gds_graph_name = gds_graph_name
        self.config = config or {}
        self._mock = driver is None

        if self._mock:
            logger.warning(
                "GDSFeatureEngineer running in MOCK mode — no Neo4j driver provided."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_query(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        if self._mock:
            logger.debug("[MOCK] Query: %s", query[:80])
            return []
        with self.driver.session() as session:
            result = session.run(query, parameters=params or {})
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Graph Projection
    # ------------------------------------------------------------------

    def project_graph(self, subject_id: str | None = None) -> dict:
        """
        Create a named GDS in-memory graph projection for the brain connectome.

        Projects BrainRegion nodes and CONNECTED_TO relationships.
        If subject_id is provided, projects only edges for that subject.

        Parameters
        ----------
        subject_id : str | None
            If provided, project only edges belonging to this subject.

        Returns
        -------
        dict  GDS projection result summary
        """
        if self._mock:
            logger.info("[MOCK] project_graph(subject_id=%s)", subject_id)
            return {"graphName": self.gds_graph_name, "nodeCount": 116, "relationshipCount": 1000}

        # Drop existing projection if it exists
        self.drop_projection()

        relationship_query = "CONNECTED_TO"
        if subject_id:
            # Use Cypher projection to filter by subject
            query = """
            CALL gds.graph.project.cypher(
                $graphName,
                'MATCH (r:BrainRegion) RETURN id(r) AS id',
                'MATCH (src:BrainRegion)-[e:CONNECTED_TO {subjectId: $subjectId}]->(tgt:BrainRegion)
                 RETURN id(src) AS source, id(tgt) AS target, e.weight AS weight',
                {parameters: {subjectId: $subjectId}}
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            result = self._run_query(query, {"graphName": self.gds_graph_name, "subjectId": subject_id})
        else:
            # Native projection for all subjects
            query = """
            CALL gds.graph.project(
                $graphName,
                'BrainRegion',
                {
                    CONNECTED_TO: {
                        orientation: 'UNDIRECTED',
                        properties: {weight: {property: 'weight', defaultValue: 0.0}}
                    }
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            result = self._run_query(query, {"graphName": self.gds_graph_name})

        if result:
            logger.info(
                "GDS graph projected: %s nodes=%d, edges=%d",
                result[0].get("graphName"), result[0].get("nodeCount"), result[0].get("relationshipCount"),
            )
            return result[0]
        return {}

    # ------------------------------------------------------------------
    # Algorithm: PageRank
    # ------------------------------------------------------------------

    def compute_page_rank(self, write_property: str = "pageRank") -> dict:
        """
        Run gds.pageRank.write to identify hub brain regions.

        Hub regions with high PageRank are central to the connectome
        network and are often implicated in neurodegeneration.

        Parameters
        ----------
        write_property : str  Neo4j property to write scores into

        Returns
        -------
        dict  GDS result summary
        """
        if self._mock:
            logger.info("[MOCK] compute_page_rank(write_property=%s)", write_property)
            return {"ranIterations": 20, "didConverge": True, "nodePropertiesWritten": 116}

        cfg = self.config.get("page_rank", {})
        query = """
        CALL gds.pageRank.write($graphName, {
            maxIterations:  $maxIterations,
            dampingFactor:  $dampingFactor,
            writeProperty:  $writeProperty,
            relationshipWeightProperty: 'weight'
        })
        YIELD ranIterations, didConverge, nodePropertiesWritten
        RETURN ranIterations, didConverge, nodePropertiesWritten
        """
        result = self._run_query(query, {
            "graphName":     self.gds_graph_name,
            "maxIterations": cfg.get("max_iterations", 20),
            "dampingFactor": cfg.get("damping_factor", 0.85),
            "writeProperty": write_property,
        })
        summary = result[0] if result else {}
        logger.info("PageRank written: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Algorithm: Degree Centrality
    # ------------------------------------------------------------------

    def compute_degree_centrality(self, write_property: str = "degree") -> dict:
        """
        Run gds.degree.write to compute local connectivity density.

        Degree centrality reflects how many direct functional connections
        each brain region maintains.

        Parameters
        ----------
        write_property : str

        Returns
        -------
        dict  GDS result summary
        """
        if self._mock:
            logger.info("[MOCK] compute_degree_centrality(write_property=%s)", write_property)
            return {"nodePropertiesWritten": 116}

        query = """
        CALL gds.degree.write($graphName, {
            writeProperty: $writeProperty,
            relationshipWeightProperty: 'weight'
        })
        YIELD nodePropertiesWritten
        RETURN nodePropertiesWritten
        """
        result = self._run_query(query, {
            "graphName":     self.gds_graph_name,
            "writeProperty": write_property,
        })
        summary = result[0] if result else {}
        logger.info("Degree centrality written: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Algorithm: Local Clustering Coefficient
    # ------------------------------------------------------------------

    def compute_clustering_coefficient(self, write_property: str = "clustering") -> dict:
        """
        Run gds.localClusteringCoefficient.write.

        Clustering coefficient measures the tendency of a region's
        neighbours to also be interconnected (local clique structure).

        Parameters
        ----------
        write_property : str

        Returns
        -------
        dict
        """
        if self._mock:
            logger.info("[MOCK] compute_clustering_coefficient()")
            return {"nodePropertiesWritten": 116}

        query = """
        CALL gds.localClusteringCoefficient.write($graphName, {
            writeProperty: $writeProperty
        })
        YIELD nodePropertiesWritten, averageClusteringCoefficient
        RETURN nodePropertiesWritten, averageClusteringCoefficient
        """
        result = self._run_query(query, {
            "graphName":     self.gds_graph_name,
            "writeProperty": write_property,
        })
        summary = result[0] if result else {}
        logger.info("Clustering coefficient written: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Algorithm: Betweenness Centrality
    # ------------------------------------------------------------------

    def compute_betweenness_centrality(self, write_property: str = "betweenness") -> dict:
        """
        Run gds.betweenness.write.

        Betweenness centrality identifies bridge regions whose dysfunction
        can disconnect major neural networks — a key biomarker in AD/PD.

        Parameters
        ----------
        write_property : str

        Returns
        -------
        dict
        """
        if self._mock:
            logger.info("[MOCK] compute_betweenness_centrality()")
            return {"nodePropertiesWritten": 116}

        query = """
        CALL gds.betweenness.write($graphName, {
            writeProperty: $writeProperty
        })
        YIELD nodePropertiesWritten, minimumScore, maximumScore, scoreSum
        RETURN nodePropertiesWritten, minimumScore, maximumScore, scoreSum
        """
        result = self._run_query(query, {
            "graphName":     self.gds_graph_name,
            "writeProperty": write_property,
        })
        summary = result[0] if result else {}
        logger.info("Betweenness centrality written: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Drop projection
    # ------------------------------------------------------------------

    def drop_projection(self) -> None:
        """Drop the named GDS graph projection if it exists."""
        if self._mock:
            logger.debug("[MOCK] drop_projection()")
            return
        query = """
        CALL gds.graph.drop($graphName, false)
        YIELD graphName
        RETURN graphName
        """
        self._run_query(query, {"graphName": self.gds_graph_name})

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_all_features(self) -> dict[str, dict]:
        """
        Run the full GDS feature engineering pipeline.

        Order: project → pageRank → degree → clustering → betweenness

        Returns
        -------
        dict  Summary of all algorithm results
        """
        logger.info("=== GDS Feature Engineering Pipeline ===")
        results: dict[str, dict] = {}
        results["projection"]  = self.project_graph()
        results["pageRank"]    = self.compute_page_rank()
        results["degree"]      = self.compute_degree_centrality()
        results["clustering"]  = self.compute_clustering_coefficient()
        results["betweenness"] = self.compute_betweenness_centrality()
        self.drop_projection()
        logger.info("=== GDS pipeline complete ===")
        return results

    # ------------------------------------------------------------------
    # Fetch node features
    # ------------------------------------------------------------------

    def fetch_node_features(self, subject_id: str) -> pd.DataFrame:
        """
        Retrieve GDS-computed node features for a given subject.

        Returns a DataFrame with columns:
          [atlasId, anatomicalLabel, degree, pageRank, clustering, betweenness]

        If in mock mode, returns a dummy DataFrame with 116 rows.

        Parameters
        ----------
        subject_id : str

        Returns
        -------
        pd.DataFrame
        """
        if self._mock:
            logger.info("[MOCK] fetch_node_features(subject_id=%s)", subject_id)
            rng = np.random.default_rng(42)
            return pd.DataFrame({
                "atlasId":        range(116),
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
        records = self._run_query(query)
        if not records:
            logger.warning("No node features found in Neo4j for subject=%s", subject_id)
            return pd.DataFrame()
        return pd.DataFrame(records)
