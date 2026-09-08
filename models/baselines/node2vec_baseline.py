"""
node2vec_baseline.py
====================
Baseline 3: node2vec embeddings + Logistic Regression.

Graph Archetype: Random Walk Pathways — node2vec maps each brain region's
neighbourhood structure into a low-dimensional embedding by maximising the
likelihood of observed random walk co-occurrences. Graph-level embeddings
are obtained by mean-pooling node embeddings.

Author: basakanimesh14
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.metrics import MetricsReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# node2vec import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from node2vec import Node2Vec as _Node2Vec
    _HAS_NODE2VEC = True
    logger.debug("node2vec library available.")
except ImportError:
    _HAS_NODE2VEC = False
    logger.warning(
        "node2vec library not found. Falling back to networkx-based random walk simulation. "
        "Install with: pip install node2vec"
    )


def _random_walk_embedding(
    G: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 50,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Fallback: simulate node embeddings using random walk co-occurrence statistics.

    This is a simplified, non-optimised fallback that uses spectral embedding
    when node2vec is unavailable.

    Parameters
    ----------
    G          : nx.Graph
    dimensions : int
    walk_length: int
    num_walks  : int
    seed       : int

    Returns
    -------
    dict[node_label → np.ndarray (dimensions,)]
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)

    if n == 0:
        return {}

    # Use spectral embedding as approximation
    try:
        from sklearn.manifold import SpectralEmbedding
        adj = nx.to_numpy_array(G, nodelist=nodes)
        # Add small epsilon to avoid zero-row issues
        adj += 1e-6
        embed_dims = min(dimensions, n - 1)
        se = SpectralEmbedding(n_components=embed_dims, random_state=seed)
        coords = se.fit_transform(adj)
        if embed_dims < dimensions:
            # Pad with zeros
            pad = np.zeros((n, dimensions - embed_dims))
            coords = np.hstack([coords, pad])
        return {node: coords[i] for i, node in enumerate(nodes)}
    except Exception as exc:
        logger.warning("Spectral embedding failed (%s) — using random embeddings", exc)
        return {node: rng.standard_normal(dimensions) for node in nodes}


# ---------------------------------------------------------------------------
# Node2Vec Baseline
# ---------------------------------------------------------------------------

class Node2VecBaseline:
    """
    node2vec graph embedding + Logistic Regression classifier.

    For each subject:
      1. Build networkx.Graph from edge list
      2. Learn node embeddings via node2vec (or fallback spectral embedding)
      3. Mean-pool node embeddings → graph-level embedding
      4. Fit Logistic Regression

    Parameters
    ----------
    config : dict
        node2vec config (from config.yaml `baselines.node2vec` section).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        n2v_cfg = config.get("node2vec", config)
        self.dimensions   = n2v_cfg.get("dimensions", 64)
        self.walk_length  = n2v_cfg.get("walk_length", 30)
        self.num_walks    = n2v_cfg.get("num_walks", 200)
        self.p            = float(n2v_cfg.get("p", 1.0))
        self.q            = float(n2v_cfg.get("q", 1.0))
        self.workers      = n2v_cfg.get("workers", 4)
        self.seed         = n2v_cfg.get("seed", 42)

        lr_cfg = n2v_cfg.get("logistic_regression", {})
        self.lr_max_iter    = lr_cfg.get("max_iter", 1000)
        self.lr_class_weight= lr_cfg.get("class_weight", "balanced")

        self.pipeline: Optional[Pipeline] = None
        self._is_fitted = False

        logger.info(
            "Node2VecBaseline initialised | dim=%d p=%.1f q=%.1f n2v=%s",
            self.dimensions, self.p, self.q,
            "native" if _HAS_NODE2VEC else "fallback",
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, subj_df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        for _, row in subj_df.iterrows():
            w = float(row["weight"])
            if w != 0.0:
                G.add_edge(row["source"], row["target"], weight=w)
        return G

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def fit_embeddings(self, G: nx.Graph) -> dict[str, np.ndarray]:
        """
        Learn node embeddings for a brain graph.

        Uses node2vec if available; falls back to spectral embedding.

        Parameters
        ----------
        G : nx.Graph

        Returns
        -------
        dict[node_label → np.ndarray (dimensions,)]
        """
        if len(G.nodes()) == 0:
            return {}

        if _HAS_NODE2VEC:
            try:
                n2v = _Node2Vec(
                    G,
                    dimensions=self.dimensions,
                    walk_length=self.walk_length,
                    num_walks=self.num_walks,
                    p=self.p,
                    q=self.q,
                    workers=self.workers,
                    seed=self.seed,
                    quiet=True,
                )
                model = n2v.fit(window=10, min_count=1, batch_words=4)
                return {str(node): model.wv[str(node)] for node in G.nodes()}
            except Exception as exc:
                logger.warning("node2vec fit failed (%s) — using fallback", exc)

        return _random_walk_embedding(
            G,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            seed=self.seed,
        )

    def graph_to_features(
        self, edge_list_df: pd.DataFrame, subject_id: str
    ) -> np.ndarray:
        """
        Compute graph-level embedding for one subject via mean-pooling.

        Parameters
        ----------
        edge_list_df : pd.DataFrame
        subject_id   : str

        Returns
        -------
        np.ndarray  (dimensions,)
        """
        subj_df = edge_list_df[edge_list_df["subject_id"] == subject_id]
        G = self._build_graph(subj_df)
        embeddings = self.fit_embeddings(G)
        if not embeddings:
            return np.zeros(self.dimensions, dtype=np.float32)
        emb_matrix = np.stack(list(embeddings.values()), axis=0)
        return emb_matrix.mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(
        self, edge_list_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare graph-level embeddings for all subjects.

        Parameters
        ----------
        edge_list_df : pd.DataFrame

        Returns
        -------
        X : np.ndarray  (n_subjects, dimensions)
        y : np.ndarray  (n_subjects,)
        """
        subject_ids = edge_list_df["subject_id"].unique()
        X_list: list[np.ndarray] = []
        y_list: list[int] = []

        for idx, subj in enumerate(subject_ids):
            if (idx + 1) % 10 == 0:
                logger.info("  Embedding subject %d / %d ...", idx + 1, len(subject_ids))
            feats = self.graph_to_features(edge_list_df, subj)
            X_list.append(feats)
            label = 1 if edge_list_df[edge_list_df["subject_id"] == subj].iloc[0]["diagnosis"] == "Diseased" else 0
            y_list.append(label)

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        logger.info(
            "Node2VecBaseline features: X=%s y=%s (Healthy=%d Diseased=%d)",
            X.shape, y.shape, int((y == 0).sum()), int((y == 1).sum()),
        )
        return X, y

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Node2VecBaseline":
        """Fit StandardScaler → LogisticRegression pipeline."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr",     LogisticRegression(
                max_iter=self.lr_max_iter,
                class_weight=self.lr_class_weight,
                random_state=self.seed,
            )),
        ])
        self.pipeline.fit(X, y)
        self._is_fitted = True
        logger.info("Node2VecBaseline fitted on %d samples", len(y))
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
        y_pred   = self.predict(X_test)
        y_scores = self.predict_proba(X_test)
        report = MetricsReport.from_arrays(y_test, y_pred, y_scores, split="test")
        logger.info("Node2VecBaseline evaluation:\n%s", report)
        return report

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def dry_run(self) -> None:
        """Print config summary without fitting."""
        print("=" * 50)
        print("Node2VecBaseline — Dry Run Summary")
        print("=" * 50)
        print(f"  Dimensions   : {self.dimensions}")
        print(f"  Walk length  : {self.walk_length}")
        print(f"  Num walks    : {self.num_walks}")
        print(f"  p (return)   : {self.p}")
        print(f"  q (in-out)   : {self.q}")
        print(f"  Workers      : {self.workers}")
        print(f"  node2vec lib : {'Available ✓' if _HAS_NODE2VEC else 'Fallback (spectral)'}")
        print(f"  LR max_iter  : {self.lr_max_iter}")
        print(f"  Pipeline     : StandardScaler → LogisticRegression")
        print("  Status       : NOT fitted (dry run)")
        print("=" * 50)

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Node2VecBaseline must be fitted before inference.")
