# PROJECT_CONTEXT.md — Antigravity IDE Master Reference

> **Project Name:** Neuro-Neo4j  
> **Author / User ID:** `basakanimesh14` (GitHub: `AnimeshBasak-14`)  
> **Workspace Path:** `c:\Users\basak\Desktop\Neuro-Neo4j`  
> **GitHub Repo:** [AnimeshBasak-14/Neuro-Neo4j](https://github.com/AnimeshBasak-14/Neuro-Neo4j) (Private)  
> **Primary Purpose:** Guide for Antigravity IDE and AI Agents working in this repository.

---

## 1. Project High-Level Description

**Neuro-Neo4j** is an end-to-end clinical machine learning framework that formulates patient resting-state fMRI scans as **relational brain graphs** rather than Euclidean image grids. 

* **Clinical Goal:** Detect early biomarker signatures of neurodegenerative diseases (Alzheimer's, Parkinson's, Dementia) before massive cellular loss occurs.
* **Task Formulation:** **Graph-level Binary Classification** (Healthy Control `0` vs. Neurodegenerative `1`) + **Post-hoc Subgraph Explainability**.
* **Atlas:** Automated Anatomical Labeling (**AAL-116**) with 116 anatomical brain regions.
* **Nodes ($V$):** 116 brain regions with 5 graph-theoretic features (Degree, PageRank, Clustering Coeff, Betweenness, Eigenvector).
* **Edges ($E$):** Functional connectivity derived from Pearson correlations of BOLD time series, thresholded at $|r| \ge 0.3$.

---

## 2. Core Architecture Stack

```
fMRI Connectomes (116x116 Pearson Correlation)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Storage & Graph Analytics: Neo4j 5.x + GDS Plugin         │
│    - Cypher schema for (:BrainRegion) & [:CONNECTED_TO]     │
│    - Computes 5 centrality features per region              │
│    - Mock fallback engine for offline / dry-run mode        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Deep Learning Layer: PyTorch Geometric (PyG 2.8)         │
│    - BrainConnectomeDataset: Graph-level data pipeline      │
│    - BrainGNNModel: GCN + GATv2 (dynamic attention, 4 heads)│
│    - Global Readout -> SwiGLU Gated Feed-Forward Head       │
│    - Threshold calibration via Youden's J-statistic         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Explainability Layer: GNNExplainer                        │
│    - Mutual Information maximization on soft edge/node masks│
│    - Top-k subgraph extraction (biomarker circuits)         │
│    - Jaccard stability evaluation across subjects           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Serving & Monitoring: FastAPI + Drift Detection          │
│    - Endpoints: /health, /ready, /metrics, /predict, /explain│
│    - Prometheus scraping metrics                            │
│    - Sliding window Macro-F1 drift monitor + retrain trigger│
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Critical Agent Guidelines & Codebase Quirks

Any Antigravity Agent modifying or executing code in this codebase **MUST** follow these rules:

### A. Python Version Constraints
* **Always use Python 3.11**: `uv venv .venv --python 3.11`.
* **Never use Python 3.14**: PyTorch and scientific binary wheels are not yet compiled for 3.14 and will fail.

### B. Package Name Shadowing: `neo4j` vs `neo4j_layer`
* There was a folder originally named `neo4j/`. In Python, having a local folder `neo4j/` **shadows** the official `pip install neo4j` driver package.
* **Resolution:** Local project code lives under `neo4j_layer/` (e.g. `from neo4j_layer.etl.transform import ...`).
* The official Neo4j driver is imported as `from neo4j import GraphDatabase`.

### C. Dataset Inheritance: Do NOT Subclass `InMemoryDataset`
* `BrainConnectomeDataset` in `models/gnn/dataset.py` uses a custom `__len__` and `__getitem__` interface rather than subclassing PyG's `InMemoryDataset`.
* **Reason:** PyG 2.8's `InMemoryDataset.__init__` internally resets `_data_list` to `None` during automated download/processing triggers.

### D. GNNExplainer API in PyG >= 2.4
* `GNNExplainer` runs its own inner optimization loop via backward passes.
* **Do NOT wrap GNNExplainer calls in `torch.no_grad()`** — it requires gradients for mask learning.
* For `explanation_type='model'`, do **not** pass `target=...` (PyG >= 2.4 prohibits target for model explanations).

### E. Hardware Roles (Strict User Constraint)
* **Tesla V100 / AGX Orin / DGX Spark:** Dedicated strictly for **TRAINING and TESTING ONLY**.
* **Serving / Hosting:** FastAPI serving runs on standard local/clinical compute, **never** deployed as a server on training GPUs.
* **Precision by Hardware:**
  * **V100:** Volta (SM 7.0) $\to$ **FP16 only** with `GradScaler` (no BF16). `pin_memory=True`.
  * **AGX Orin:** Ampere (SM 8.7) $\to$ **FP16**. Unified memory $\to$ `pin_memory=False`.
  * **DGX Spark:** Blackwell (SM 10.0) $\to$ **BF16 native**.

### F. Windows Console Character Encoding
* When executing dry-run tests or output scripts on Windows consoles, stdout must be UTF-8 safe or use ASCII indicators (`PASS` / `FAIL`) to avoid `UnicodeEncodeError: 'charmap' codec` with emoji characters (`\u274c`).

---

## 4. Repository File Map

| Path | Purpose |
|---|---|
| `config/config.yaml` | Base central configuration (AAL-116, default hyperparams). |
| `config/agx_config.yaml` | AGX Orin optimized config (FP16, 128-dim, unified memory). |
| `config/v100_config.yaml` | Tesla V100 optimized config (FP16, 192-dim, dedicated VRAM, DDP). |
| `config/dgx_spark_config.yaml` | DGX Spark Blackwell config (BF16, 256-dim). |
| `data/synthetic/generate_synthetic.py` | Generates synthetic AAL-116 fMRI data with neurodegenerative perturbations. |
| `neo4j_layer/etl/` | Connectome transformation (regression, z-score, thresholding, edge list generation). |
| `neo4j_layer/gds/` | GDS Cypher queries and NetworkX mock feature projection engine. |
| `models/metrics.py` | `MetricsReport`, Accuracy, Macro-F1, ROC-AUC, Jaccard Index. |
| `models/baselines/` | Benchmark baselines: RBF-SVM, RF-GDV (5 orbits), node2vec (with spectral fallback). |
| `models/gnn/gcn_gat_model.py` | `BrainGNNModel` (Hybrid GCN + GAT + BatchNorm + Residuals). |
| `models/gnn/dataset.py` | `BrainConnectomeDataset` (PyG DataLoader compatible). |
| `models/gnn/train.py` | `GNNTrainer` (AMP FP16/BF16, cosine warmup, early stopping, checkpointing). |
| `models/gnn/explainer.py` | `BrainGNNExplainer` (extracts top-k biomarker subgraphs). |
| `api/main.py` | FastAPI application factory, lifespan, CORS. |
| `api/routers/` | `/predict`, `/explain`, `/health`, `/metrics` (Prometheus). |
| `monitoring/drift_monitor.py` | Sliding window Macro-F1 drift detection. |
| `validation/subject_split.py` | Subject-level stratification (prevents data leakage). |
| `dry_run/dry_run_all.py` | Master smoke test suite (10/10 tests, offline, CPU-only). |
| `scripts/train_launcher.py` | Unified launcher that auto-detects GPU and executes full pipeline. |
| `scripts/train_v100.sh` | Shell launcher for single or multi-GPU V100 servers. |
| `scripts/eval_agx.sh` | Shell launcher for AGX Orin training or evaluation. |

---

## 5. Quick Command Reference

```bash
# 1. Environment setup (Windows/Linux)
uv venv .venv --python 3.11
uv sync

# 2. Run master smoke test (Zero GPU / Zero DB required - 10/10 PASS)
uv run python dry_run/dry_run_all.py

# 3. Launch full training pipeline (Auto-detects GPU & config)
python scripts/train_launcher.py

# 4. Run API locally
uv run uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```
