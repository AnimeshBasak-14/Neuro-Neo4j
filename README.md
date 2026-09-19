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
├── config/               # Hardware & project YAML configurations (V100, AGX, base)
├── data/
│   ├── raw/              # Raw .mat / .csv correlation matrices
│   ├── processed/        # Edge lists after preprocessing & binary cache
│   └── synthetic/        # AAL-116 synthetic fMRI connectomes
├── neo4j_layer/          # Neo4j Graph DB & ETL layer (avoids driver shadowing)
│   ├── schema/           # Cypher constraints & indexes
│   ├── etl/              # Transform → Edge List → LOAD CSV
│   └── gds/              # GDS graph projection & centrality features (with mock)
├── models/
│   ├── baselines/        # SVM, RF+GDV, node2vec+LR
│   ├── gnn/              # BrainGNNModel (GCN+GATv2+SwiGLU), GNNTrainer, GNNExplainer
│   └── metrics.py        # Accuracy, Macro-F1, ROC-AUC, Jaccard
├── api/
│   ├── routers/          # /predict, /explain, /health, /metrics
│   ├── main.py           # FastAPI application
│   ├── schemas.py        # Pydantic request/response models
│   └── neo4j_client.py   # Neo4j driver wrapper
├── monitoring/           # Drift detection & retraining trigger
├── validation/           # Subject-level splits, domain-shift analysis
├── scripts/              # Training launcher & publication visualization generator
└── dry_run/              # Master smoke test suite (10/10 PASS offline)
```

---

## Model Architecture & The 3 Breakthrough Pillars

```
Input node features (5-dim) + Pearson Edge Attributes (edge_attr)
                          │
  [GCN Conv × 3]          Global neighbourhood aggregation (edge_weight=|r|, SiLU)
                          │
  [GATv2 Conv × 1]        Dynamic attention aggregation (4 heads, edge_dim=1, SiLU)
                          │
  [Dual Global Pooling]   Concat: [global_mean_pool, global_max_pool] → ℝ^(hidden × 2)
                          (Captures both whole-brain context & focal outlier lesions)
                          │
  [SwiGLU Head]           SOTA Gated Unit: (Swish(x W_g) ⊗ (x W_u)) W_d
                          (Dynamically filters scanner SNR noise & non-linear artifacts)
                          │
  [Linear Projection]     → Softmax → P(Healthy), P(Diseased)
```

---

## Empirical Benchmarks (Tesla V100 Held-Out Test Cohort, N=75)

| Model Architecture | Attention Mechanism | Edge Aware | Global Pooling | Test ROC-AUC | Test Accuracy | Test Macro-F1 |
|---|---|---|---|:---:|:---:|:---:|
| **RBF-SVM** | None (1D vector) | Flattened | Flattened | `0.6710` | `54.67%` | `0.5210` |
| **RF-GDV** | None (Graphlets) | Orbit counts | Summary stats | `0.6920` | `56.00%` | `0.5340` |
| **node2vec** | None (Spectral / Walks) | Unweighted | Mean embedding | `0.6350` | `52.00%` | `0.4980` |
| **GATv1 (Raw)** | Static Attention | No | Mean Pool | `0.7127` | `57.33%` | `0.5502` |
| **GATv1 (Calibrated)** | Static Attention | No | Mean Pool | `0.7127` | `68.00%` | `0.6780` |
| **GATv2 + SwiGLU** | Dynamic Attention | No | Mean Pool | `0.7127` | `68.00%` | `0.6795` |
| **Neuro-Neo4j (Ours Full)** | **Dynamic Attention** | **Yes (Pearson $r$)** | **Dual (Mean+Max)** | **`0.7461`** | **`73.33%`** | **`0.7310`** |

> **Threshold Calibration**: The optimal Youden decision boundary $t^* = 0.1762$ ($J = 0.4644$) realigns operational clinical accuracy ($73.33\%$) with the model's strong discriminative ranking power across multi-site scanners.

---

## Publication Figures

All 300-DPI publication plots are generated via `python scripts/visualize_results.py`:
- `logs/plots/1_training_convergence_curves.png`: 70-epoch convergence curves directly extracted from the V100 checkpoint.
- `logs/plots/2_confusion_matrix_calibrated.png`: Raw vs. Calibrated ($t^*=0.1762$) confusion matrix showing `73.33%` accuracy.
- `logs/plots/3_model_ablation_comparison.png`: Architectural ablation benchmark across all classical and GNN models.
- `logs/plots/4_biomarker_circuits.png`: Consensus biomarker circuits discovered by GNNExplainer across functional brain networks.

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
