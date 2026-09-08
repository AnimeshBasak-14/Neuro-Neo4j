"""
rf_gdv_baseline.py
==================
Baseline 2: Random Forest on Graphlet Degree Vectors (GDV).

Graph Archetype: Automorphism Orbits — GDVs capture the local structural
symmetries and orbital positions of nodes in the network. Aggregated
graph-level GDV statistics (mean, std, max) serve as subject features.

Graphlet features computed:
  - Degree sequence (orbit 0)
  - Local clustering coefficients (orbit 1: triangle participation)
  - Triangle counts per node (orbit 2)
  - 4-clique participation (approximate)

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from models.metrics import MetricsReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graphlet Degree Vector computation
# ---------------------------------------------------------------------------

def compute_gdv_features(G: nx.Graph, n_regions: int = 116) -> np.ndarray:
    """
    Compute per-node graphlet degree vector features for a brain graph.

    Implemented orbits
    ------------------
    Orbit 0 : Degree  (edges touching node)
    Orbit 1 : Clustering coefficient  (triangle participation ratio)
    Orbit 2 : Triangle count  (number of triangles the node participates in)
    Orbit 3 : Square count  (4-cycle participation, approximate)
    Orbit 4 : Wedge count  (path of length 2 centred at node)

    Parameters
    ----------
    G         : nx.Graph  Brain functional connectivity graph
    n_regions : int       Expected number of nodes (for zero-padding)

    Returns
    -------
    np.ndarray  (n_nodes, n_orbits) per-node feature matrix
    """
    nodes = list(G.nodes())
    n_nodes = len(nodes)

    if n_nodes == 0:
        return np.zeros((n_regions, 5), dtype=np.float32)

    # Orbit 0: Degree
    degrees = np.array([G.degree(n) for n in nodes], dtype=np.float32)

    # Orbit 1: Clustering coefficient
    clustering = np.array(
        [nx.clustering(G, n) for n in nodes], dtype=np.float32
    )

    # Orbit 2: Triangle count
    triangles = np.array(
        [nx.triangles(G, n) for n in nodes], dtype=np.float32
    )

    # Orbit 3: Square count (approximate via common neighbour pairs)
    squares = np.zeros(n_nodes, dtype=np.float32)
    for idx, node in enumerate(nodes):
        nbrs = set(G.neighbors(node))
        sq_count = 0
        for nbr in nbrs:
            nbr_nbrs = set(G.neighbors(nbr)) - {node}
            sq_count += len(nbrs & nbr_nbrs)
        squares[idx] = sq_count / 2.0  # each square counted twice

    # Orbit 4: Wedge count (paths of length 2 centred at node)
    wedges = degrees * (degrees - 1) / 2.0

    gdv = np.column_stack([degrees, clustering, triangles, squares, wedges])
    return gdv.astype(np.float32)


def aggregate_gdv(gdv: np.ndarray) -> np.ndarray:
    """
    Aggregate node-level GDV matrix into a graph-level feature vector.

    Aggregation: [mean, std, max] across all nodes → (n_orbits * 3,) vector.

    Parameters
    ----------
    gdv : np.ndarray  (n_nodes, n_orbits)

    Returns
    -------
    np.ndarray  (n_orbits * 3,)
    """
    if gdv.shape[0] == 0:
        return np.zeros(15, dtype=np.float32)
    mean = gdv.mean(axis=0)
    std  = gdv.std(axis=0)
    maxi = gdv.max(axis=0)
    return np.concatenate([mean, std, maxi])


# ---------------------------------------------------------------------------
# Random Forest Baseline
# ---------------------------------------------------------------------------

class RFGDVBaseline:
    """
    Random Forest classifier on graph-level GDV feature vectors.

    For each subject graph:
      1. Reconstruct networkx.Graph from edge list
      2. Compute per-node GDV features (5 orbits)
      3. Aggregate to graph-level vector via [mean, std, max]
      4. Fit Random Forest

    Parameters
    ----------
    config : dict
        Random Forest config (from config.yaml `baselines.random_forest` section).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        rf_cfg = config.get("random_forest", config)
        self.n_estimators  = rf_cfg.get("n_estimators", 200)
        self.max_depth     = rf_cfg.get("max_depth", None)
        self.class_weight  = rf_cfg.get("class_weight", "balanced_subsample")
        self.random_state  = rf_cfg.get("random_state", 42)

        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False

        logger.info(
            "RFGDVBaseline initialised | n_estimators=%d max_depth=%s",
            self.n_estimators, self.max_depth,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, subj_df: pd.DataFrame) -> nx.Graph:
        """Build a networkx Graph from a subject's edge list."""
        G = nx.Graph()
        for _, row in subj_df.iterrows():
            w = float(row["weight"])
            if w != 0.0:
                G.add_edge(row["source"], row["target"], weight=w)
        return G

    def graph_to_features(
        self, edge_list_df: pd.DataFrame, subject_id: str
    ) -> np.ndarray:
        """
        Compute graph-level GDV feature vector for one subject.

        Parameters
        ----------
        edge_list_df : pd.DataFrame  Full edge list
        subject_id   : str

        Returns
        -------
        np.ndarray  (15,) aggregated GDV feature vector
        """
        subj_df = edge_list_df[edge_list_df["subject_id"] == subject_id]
        G = self._build_graph(subj_df)
        gdv = compute_gdv_features(G)
        return aggregate_gdv(gdv)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(
        self, edge_list_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare graph-level GDV features for all subjects.

        Parameters
        ----------
        edge_list_df : pd.DataFrame

        Returns
        -------
        X : np.ndarray  (n_subjects, 15)
        y : np.ndarray  (n_subjects,)
        """
        subject_ids = edge_list_df["subject_id"].unique()
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for subj in subject_ids:
            feats = self.graph_to_features(edge_list_df, subj)
            X_list.append(feats)
            label = 1 if edge_list_df[edge_list_df["subject_id"] == subj].iloc[0]["diagnosis"] == "Diseased" else 0
            y_list.append(label)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        logger.info(
            "RFGDVBaseline features: X=%s y=%s (Healthy=%d Diseased=%d)",
            X.shape, y.shape, int((y == 0).sum()), int((y == 1).sum()),
        )
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFGDVBaseline":
        """Fit StandardScaler → RandomForestClassifier pipeline."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )),
        ])
        self.pipeline.fit(X, y)
        self._is_fitted = True
        logger.info("RFGDVBaseline fitted on %d samples", len(y))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> MetricsReport:
        """Evaluate on test set."""
        y_pred   = self.predict(X_test)
        y_scores = self.predict_proba(X_test)
        report = MetricsReport.from_arrays(y_test, y_pred, y_scores, split="test")
        logger.info("RFGDVBaseline evaluation:\n%s", report)
        return report

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def dry_run(self) -> None:
        """Print config summary without fitting."""
        print("=" * 50)
        print("RFGDVBaseline — Dry Run Summary")
        print("=" * 50)
        print(f"  Estimators   : {self.n_estimators}")
        print(f"  Max depth    : {self.max_depth}")
        print(f"  Class weight : {self.class_weight}")
        print(f"  Random state : {self.random_state}")
        print(f"  GDV orbits   : 5 (degree, clustering, triangles, squares, wedges)")
        print(f"  Feature dim  : 15 (5 orbits × 3 aggregators: mean, std, max)")
        print(f"  Pipeline     : StandardScaler → RandomForestClassifier")
        print("  Status       : NOT fitted (dry run)")
        print("=" * 50)

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("RFGDVBaseline must be fitted before inference.")
