# Neuro-Neo4j

**Hybrid Graph Database and Explainable GNN Pipeline for Brain Biomarker Discovery in Neurodegenerative Disorders**

> Author: **basakanimesh14**

---

## Overview

Neuro-Neo4j treats every patient's fMRI scan as a **relational graph** — brain regions are nodes, Pearson correlations are edges — and learns to classify neurodegeneration by reading the *topology* rather than the raw signal.

| Stage | Technology |
|-------|-----------|
| Knowledge Graph Storage | Neo4j 5.x + GDS plugin |
| Feature Engineering | GDS PageRank, Degree, Betweenness |
| Baseline Models | SVM · RF+GDV · node2vec+LR |
| Deep Model | Hybrid GCN + GAT (PyTorch Geometric) |
| Explainability | GNNExplainer → biomarker subgraphs |
| Serving | FastAPI (`/predict`, `/explain`, `/health`, `/metrics`) |
| Drift Monitoring | Macro-F1 sliding window + auto-retrain trigger |

---

## Atlas

Synthetic data uses the **AAL-116** atlas (116 anatomical brain regions). Change to Schaefer-100 or any other atlas in `config/config.yaml`.

---

## Quick Start

### 1. Start Neo4j (Docker)
```bash
docker compose up -d
# Browser UI: http://localhost:7474  (neo4j / neuro_password)
```

### 2. Set Up Python Environment
```powershell
.\setup_env.ps1
.venv\Scripts\Activate.ps1
```

### 3. Generate Synthetic Data
```powershell
uv run python data\synthetic\generate_synthetic.py
```

### 4. Run Dry-Run (no training, no GPU needed)
```powershell
uv run python dry_run\dry_run_all.py
```

### 5. Start API
```powershell
uv run uvicorn api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

---

## Project Structure

```
Neuro-Neo4j/
├── config/               # Central YAML configuration
├── data/
│   ├── raw/              # Raw .mat / .csv correlation matrices
│   ├── processed/        # Edge lists after preprocessing
│   └── synthetic/        # AAL-116 synthetic fMRI data (for dry-run)
├── neo4j/
│   ├── schema/           # Cypher constraints & indexes
│   ├── etl/              # Transform → Edge List → LOAD CSV
│   └── gds/              # GDS graph projection & centrality features
├── models/
│   ├── baselines/        # SVM, RF+GDV, node2vec+LR
│   ├── gnn/              # Hybrid GCN+GAT, Trainer, GNNExplainer
│   └── metrics.py        # Accuracy, Macro-F1, ROC-AUC, Jaccard
├── api/
│   ├── routers/          # /predict, /explain, /health, /metrics
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic request/response models
│   └── neo4j_client.py   # Neo4j driver wrapper
├── monitoring/           # Drift detection & retraining trigger
├── validation/           # Subject-level splits, domain-shift analysis
└── dry_run/              # Smoke tests — no training required
```

---

## Metrics

| Metric | Definition | Rationale |
|--------|-----------|-----------|
| **Accuracy** | Correct / Total | Overall performance snapshot |
| **Macro-F1** | Harmonic mean of per-class F1 | Clinical robustness — penalises missed diseased cases |
| **ROC-AUC** | Area under ROC curve | Threshold-independent discrimination |
| **Jaccard** | \|A∩B\| / \|A∪B\| | Biomarker subgraph stability across runs |

---

## Model Architecture

```
Input node features (5-dim)
        │
  [GCN Layer 1]  global feature aggregation
  [GCN Layer 2]
        │
  [GAT Layer 1]  attention-weighted aggregation (4 heads)
        │
  [Global Mean Pool]  → graph-level embedding
        │
  [MLP Classifier]  → Healthy / Diseased
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness + Neo4j + model status |
| GET | `/ready` | Readiness probe |
| GET | `/metrics` | Current Macro-F1, accuracy, drift status |
| GET | `/metrics/prometheus` | Prometheus-format metrics |
| POST | `/predict` | Single subject classification |
| POST | `/predict/batch` | Batch classification |
| POST | `/explain` | GNNExplainer biomarker subgraph |

---

## License

MIT
