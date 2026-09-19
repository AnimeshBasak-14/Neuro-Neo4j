# Solution

## Neuro-Neo4j: Architecture, Design Decisions, and Implementation

**Author:** basakanimesh14  
**Repository:** [AnimeshBasak-14/Neuro-Neo4j](https://github.com/AnimeshBasak-14/Neuro-Neo4j)

---

## 1. Solution Overview

Neuro-Neo4j addresses the three research gaps with a unified pipeline combining a **persistent relational knowledge graph** (Neo4j), a **hybrid graph neural network** (GCN + GAT), and a **post-hoc explainability layer** (GNNExplainer).

```
fMRI Connectomes
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Knowledge Graph Storage (Neo4j + GDS)             │
│  • Persistent graph of 116 brain regions per subject        │
│  • Queryable: "which subjects show loss of DMN coherence?"  │
│  • GDS plugin computes centrality features in Cypher        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Feature Engineering                               │
│  • 5 node features per region (graph-theoretic)             │
│  • Degree · PageRank · Clustering · Betweenness · Eigenvec  │
│  • Threshold + z-score normalise edge weights               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Hybrid GCN + GAT Classifier                       │
│  • GCNConv layers: aggregate neighbourhood structure        │
│  • GATConv layers: attention-weighted aggregation           │
│  • Global mean pooling → graph-level embedding              │
│  • MLP head → Healthy / Diseased                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4: Explainability (GNNExplainer)                     │
│  • Learns soft mask over edges and node features            │
│  • Top-k subgraph = candidate biomarker circuit             │
│  • Jaccard stability score across subjects                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 5: Monitoring & API                                  │
│  • FastAPI: /predict /explain /health /metrics              │
│  • Drift monitor: sliding-window Macro-F1 degradation alert │
│  • Auto-retrain trigger when F1 drops below threshold       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1 — Knowledge Graph (Neo4j)

### 2.1 Why Neo4j

Functional connectivity data is inherently relational. Neo4j stores this as a **native property graph** where:
- **Nodes** = brain regions (`BrainRegion`, `Subject`)
- **Relationships** = `CONNECTED_TO {weight, z_score, diagnosis}`
- **Graph algorithms** run in-database via the GDS plugin (no Python round-trip)

### 2.2 Schema

```cypher
// Constraints (constraints.cypher)
CREATE CONSTRAINT region_unique IF NOT EXISTS
  FOR (r:BrainRegion) REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT subject_unique IF NOT EXISTS
  FOR (s:Subject) REQUIRE s.patient_id IS UNIQUE;

// Node structure
(:BrainRegion {name, lobe, hemisphere, aal_index})
(:Subject {patient_id, diagnosis, age, scanner_site, label})

// Edge structure
(:BrainRegion)-[:CONNECTED_TO {
    weight,        // Pearson correlation
    z_score,       // normalised weight
    subject_id,    // links to Subject
    diagnosis      // Healthy | Diseased
}]->(:BrainRegion)
```

### 2.3 GDS Feature Projection

The Graph Data Science plugin projects the stored graph into memory and computes centrality measures in Cypher:

```cypher
CALL gds.graph.project('brain_graph', 'BrainRegion', 'CONNECTED_TO',
  {relationshipProperties: 'weight'})

CALL gds.pageRank.stream('brain_graph')      // PageRank centrality
CALL gds.degree.stream('brain_graph')         // Degree centrality
CALL gds.betweenness.stream('brain_graph')    // Betweenness centrality
CALL gds.localClusteringCoefficient.stream('brain_graph')
CALL gds.eigenvector.stream('brain_graph')    // Eigenvector centrality
```

These 5 features form the node feature matrix $X \in \mathbb{R}^{116 \times 5}$ fed into the GNN.

> **Mock mode:** When Neo4j is unavailable (dry-run, AGX training), all GDS calls are intercepted by `FeatureProjectionEngine(mock=True)`, which computes equivalent features using NetworkX — no Neo4j instance required.

---

## 3. Layer 2 — Preprocessing Pipeline

### 3.1 ETL Steps

| Step | Operation | Implementation |
|---|---|---|
| **Global signal regression** | Remove shared variance across all regions | `ConnectomeTransformer.regress_global_signal()` |
| **Z-score normalisation** | Mean=0, Std=1 per connection across subjects | `ConnectomeTransformer.normalize(method='zscore')` |
| **Sparsification** | Keep edges where \|r\| > 0.3 (Pearson) | `ConnectomeTransformer.threshold(0.3)` |
| **Symmetrisation** | Enforce $A = A^T$ | Done at matrix creation |
| **Edge list export** | Write to CSV for PyG DataLoader | `to_edge_list()` |

### 3.2 Threshold Choice: 0.3

A Pearson threshold of 0.3 is standard in the fMRI literature (Power et al., 2011; Bullmore & Sporns, 2009). It:
- Removes noise-dominated weak correlations
- Preserves anatomically meaningful connections
- Keeps graphs sparse enough for tractable computation (~7–15% edge density for AAL-116)

---

## 4. Layer 3 — Hybrid GCN + GAT Architecture

### 4.1 Design Rationale

Two complementary message-passing mechanisms are combined:

| Component | What it captures | Why needed |
|---|---|---|
| **GCNConv** | Isotropic neighbourhood aggregation | Structural connectivity patterns, hub regions |
| **GATv2Conv** | Dynamic attention-weighted aggregation | Fixes static attention limitation (Brody et al., 2021); selectively focuses on disrupted functional pathways |
| **SwiGLU Head** | Multiplicative gated linear unit | SOTA activation (Shazeer, 2020; LLaMA 3); dynamically filters scanner SNR fluctuations and non-linear artifacts |

GCN alone misses the importance weighting of connections. GATv2 provides strictly expressive dynamic attention over static GAT. Combining them gives structural stability from GCN and selective pathway focus from GATv2. The SwiGLU classification readout head stabilizes predictions across multi-site variance.

### 4.2 Architecture

```
Input: X ∈ ℝ^(116 × 5)    [116 regions, 5 node features each]
       A ∈ ℝ^(116 × 116)   [adjacency / edge index]

────────────────────────────────────────────────────────
 GCN Block × num_gcn_layers:
   GCNConv(in → hidden)
   BatchNorm1d(hidden)
   ReLU
   Dropout(p=0.2)
   + Residual connection (projection if dims differ)

 GATv2 Block × num_gat_layers:
   GATv2Conv(hidden → hidden // H, heads=H, concat=True)
   BatchNorm1d(hidden)
   ELU
   Dropout(p=0.2)
   + Residual connection

 Global Readout (Mean Pooling):
   h_G = (1/|V|) Σ h_v        [graph-level embedding, ℝ^hidden]

 SwiGLU Classifier Head:
   SwiGLU: (Swish(h_G · W_gate) ⊗ (h_G · W_up)) · W_down  →  ℝ^(hidden//2)
   Dropout(p=0.2)
   Linear(hidden//2 → 2)      [logits for 2 classes]
────────────────────────────────────────────────────────
Output: logits ∈ ℝ^2  →  Softmax  →  P(Healthy), P(Diseased)
```

### 4.3 Threshold Calibration: Youden's J-Statistic

In multi-site clinical deployments, scanner noise distributions cause an apparent gap between **ROC-AUC (~0.71–0.80)** and raw accuracy (~57%) under an arbitrary fixed $p = 0.5$ threshold. We incorporate **Youden's J-statistic** threshold optimization:

$$J(t) = \text{Sensitivity}(t) + \text{Specificity}(t) - 1 = \text{TPR}(t) - \text{FPR}(t)$$

The optimal decision boundary $t^* = \arg\max_t J(t)$ is calibrated on validation probabilities and applied during inference ($\hat{y} = \mathbb{I}(P(\text{Diseased}) \ge t^*)$), realigning operational accuracy and Macro-F1 with the model's true discriminative ranking power.

### 4.4 Model Sizes by Hardware

| Hardware | Hidden | GCN | GATv2 | Heads | Readout | Precision |
|---|---|---|---|---|---|---|
| AGX Orin | 128 | 3 | 1 | 4 | SwiGLU | FP16 |
| V100 Server | 192 | 3 | 1 | 4 | SwiGLU | FP16 |
| DGX Spark (future) | 256 | 4 | 2 | 8 | SwiGLU | BF16 |

> For 116-node brain graphs with ~250–500 subjects, a 100K–170K parameter model provides the ideal inductive bias, avoiding the severe overfitting seen with multi-million parameter dense networks.

### 4.5 Training Configuration

| Setting | Value | Rationale |
|---|---|---|
| Loss function | CrossEntropyLoss + label smoothing 0.1 | Prevents overconfident predictions on small datasets |
| Optimiser | AdamW (lr=0.001, wd=0.0001) | Weight decay regularises sparse graph features |
| Scheduler | Linear warmup (5 ep) + cosine decay | Prevents early overfitting, smooth convergence |
| Activation | SwiGLU in classification head | Multiplicative gating filters scanner noise |
| Attention | GATv2Conv (dynamic self-attention) | Overcomes standard GAT rank collapse |
| Early stopping | Patience = 25 epochs on val Macro-F1 | Stops when validation F1 plateaus |
| Gradient clipping | max norm = 1.0 | Stabilises GATv2 attention gradient flow |
| Mixed precision | FP16 (V100/AGX) / BF16 (Blackwell) | 2× throughput, same accuracy |

### 4.5 Validation Strategy

- **Subject-level split** (70/15/15): Subjects are split — not graphs — preventing data leakage where multiple scans from one subject appear in both train and test
- **5-fold cross-validation**: Reported metrics are averaged across 5 folds, with standard deviation
- **Stratification**: Each fold maintains the same class ratio

---

## 5. Layer 4 — GNNExplainer: Biomarker Subgraph Extraction

### 5.1 How GNNExplainer Works

GNNExplainer (Ying et al., 2019) learns a **soft mask** $m_E \in [0,1]^{|E|}$ over edges and $m_F \in [0,1]^{|V| \times F}$ over node features that maximally preserve the model's prediction when the masked subgraph is substituted for the full graph:

$$\arg\max_{m_E, m_F} \; \text{MI}\big(\hat{y}, (G \odot m_E, X \odot m_F)\big)$$

This is solved by gradient ascent on $m_E$ and $m_F$ while keeping model weights frozen.

### 5.2 Clinical Output

For each test subject, GNNExplainer returns:

1. **Top-$k$ edges** — ranked functional connections most responsible for the prediction
2. **Top-$k$ regions** — brain areas with highest feature importance
3. **Biomarker subgraph** — the minimal subgraph that reproduces the classification
4. **Jaccard stability** — consistency of the subgraph across similar subjects

Example output for a Diseased subject:
```
Top Biomarker Regions:
  1. Hippocampus_L       (importance: 0.94)
  2. Frontal_Sup_L       (importance: 0.88)
  3. Precuneus_L         (importance: 0.81)
  4. Cingulum_Post_L     (importance: 0.79)
  5. Amygdala_L          (importance: 0.72)

Top Disrupted Connections:
  Hippocampus_L ↔ Precuneus_L              (weight: 0.82)
  Frontal_Sup_L ↔ Cingulum_Post_L          (weight: 0.77)
```

> These regions correspond to the **Default Mode Network** — exactly the circuit known to show early disconnection in Alzheimer's disease.

---

## 6. Layer 5 — API, Monitoring & Deployment

### 6.1 FastAPI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness: checks Neo4j + model checkpoint |
| `/ready` | GET | Readiness probe (Kubernetes-compatible) |
| `/predict` | POST | Single-subject classification |
| `/predict/batch` | POST | Batch classification |
| `/explain` | POST | GNNExplainer biomarker subgraph |
| `/metrics` | GET | Current Macro-F1, accuracy, drift status |
| `/metrics/prometheus` | GET | Prometheus-compatible metrics scrape |

### 6.2 Drift Detection

The `DriftMonitor` tracks Macro-F1 on a sliding window of recent predictions:

```
F1 baseline = 0.80   (set at deployment)
Window size = 50 predictions
Alert threshold = 5% drop (F1 < 0.76)
Retrain trigger = 3 consecutive windows below threshold
```

When drift is detected:
1. `RetrainTrigger` emits an alert to the log
2. Operator is notified (extensible: Slack, email, PagerDuty)
3. New training run is initiated with accumulated data

### 6.3 Deployment Note

> **Training hardware (V100 / AGX Orin / DGX Spark) is used for training and testing ONLY.**  
> FastAPI inference runs on separate, always-on clinical hardware.  
> Checkpoints (`.pt` files) are the artefact transferred between training and serving environments.

---

## 7. Baseline Comparisons

Three baselines are included for rigorous evaluation:

| Baseline | Features | Model | Complexity |
|---|---|---|---|
| **SVM** | Flattened upper-triangle (6,670 features) | RBF-SVM | $O(n^2)$ — slow for large $N$ |
| **RF + GDV** | 5 graphlet degree vectors × 3 aggregators = 15 features | Random Forest (200 trees) | $O(n \log n)$ |
| **node2vec + LR** | 64-dim node embeddings, mean-pooled | Logistic Regression | $O(n \cdot W \cdot L)$ |
| **Neuro-Neo4j GNN** | 5-dim node features (GDS centrality) | GCN + GAT | $O(n \cdot |E| \cdot H)$ |

The GNN is expected to outperform all baselines because it simultaneously learns *which structural patterns* are discriminative rather than relying on fixed feature engineering.

---

## 8. Training & Testing Time Estimates

### 8.1 Dataset Scale

| Scale | Subjects | Graphs | Edges (total) | Recommended Hardware |
|---|---|---|---|---|
| Small (default) | 250 | 250 | ~1.2M | AGX Orin |
| Medium | 500 | 500 | ~2.5M | V100 Server |
| Large | 1,000 | 1,000 | ~5M | V100 (multi) / DGX Spark |

### 8.2 AGX Orin — 250 Subjects, 200 Epochs

| Phase | Duration | Notes |
|---|---|---|
| Data generation | 2–3 min | 250 connectomes, AAL-116 |
| ETL (edge list) | 4–6 min | 250 CSV files → 1 edge list |
| Dry run | 30 sec | 10 smoke tests, CPU only |
| Training (200 epochs) | **20–35 min** | FP16, batch=32, ~3–5s/epoch |
| Evaluation (test set, 38 subjects) | < 5 sec | Single forward pass |
| GNNExplainer (38 subjects) | **15–25 min** | 1000 optimisation steps per subject |
| **Total** | **~45–70 min** | |

### 8.3 V100 Server — 500 Subjects, 200 Epochs

| Phase | Duration | Notes |
|---|---|---|
| Data generation | 4–6 min | 500 connectomes |
| ETL | 8–12 min | Larger edge list |
| Training (200 epochs) | **8–15 min** | FP16, 32GB VRAM, ~2–4s/epoch |
| Evaluation | < 3 sec | |
| GNNExplainer (75 test subjects) | **10–20 min** | Faster GPU = faster mask optimisation |
| **Total** | **~30–55 min** | |

**With 4× V100 DDP:**

| Phase | Duration |
|---|---|
| Training (200 epochs) | **3–5 min** |
| Total pipeline | **~25–35 min** |

### 8.4 DGX Spark — 500 Subjects, 300 Epochs (When Available)

| Phase | Duration | Notes |
|---|---|---|
| Data generation | 3–5 min | |
| ETL | 6–10 min | |
| Training (300 epochs) | **10–20 min** | BF16, 128GB unified, ~2–4s/epoch |
| Evaluation | < 2 sec | |
| GNNExplainer (75 subjects) | **8–15 min** | |
| **Total** | **~30–55 min** | |

### 8.5 Summary Table

| Hardware | Subjects | Epochs | Training | GNNExplainer | **Full Pipeline** |
|---|---|---|---|---|---|
| AGX Orin | 250 | 200 | ~30 min | ~20 min | **~45–70 min** |
| V100 (1×) | 500 | 200 | ~12 min | ~15 min | **~35–55 min** |
| V100 (4×) | 500 | 200 | ~4 min | ~15 min | **~30–40 min** |
| DGX Spark | 500 | 300 | ~15 min | ~12 min | **~30–55 min** |

> [!NOTE]
> GNNExplainer is the computational bottleneck — it runs an independent optimisation
> loop (1,000 gradient steps) per subject. Time scales linearly with test set size.
> On AGX Orin, you can reduce `explainer.epochs: 500` in `config/agx_config.yaml`
> to halve explainer time with minor mask quality reduction.

---

## 9. Expected Performance (Synthetic Data)

> [!NOTE]
> The following are expected ranges for the **synthetic AAL-116 dataset**.
> Performance on real clinical data will differ. Synthetic data has controlled
> class-discriminative signal; real data has higher noise and more subtle effects.

| Baseline | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| SVM (RBF kernel) | 0.65–0.75 | 0.63–0.73 | 0.70–0.80 |
| RF + GDV | 0.68–0.78 | 0.66–0.76 | 0.73–0.83 |
| node2vec + LR | 0.62–0.72 | 0.60–0.70 | 0.68–0.78 |
| **Neuro-Neo4j GNN** | **0.80–0.90** | **0.80–0.88** | **0.87–0.95** |

---

## 10. Limitations and Future Work

| Limitation | Planned Solution |
|---|---|
| Synthetic data only | Adapter for ADNI, PPMI, HCP real datasets |
| AAL-116 atlas hardcoded | Configurable atlas (Schaefer-100, Destrieux-148) |
| Binary classification only | Multi-class (AD vs PD vs FTD vs HC) |
| Single timepoint per subject | Longitudinal graph sequence modelling |
| No structural MRI fusion | Multi-modal: fMRI + DTI tractography |
| GNNExplainer is slow | Replace with faster gradient-based methods (Integrated Gradients) |
| No federated learning | Multi-site training without raw data sharing |

---

## 11. Project Structure Reference

```
Neuro-Neo4j/
├── PROBLEM_STATEMENT.md      ← Problem definition
├── SOLUTION.md               ← This document
├── config/
│   ├── config.yaml           ← Base config (CPU / default)
│   ├── agx_config.yaml       ← AGX Orin (FP16, 128-dim)
│   ├── v100_config.yaml      ← V100 server (FP16, 192-dim)
│   └── dgx_spark_config.yaml ← DGX Spark (BF16, 256-dim)
├── data/synthetic/           ← AAL-116 fMRI data generator
├── neo4j/                    ← Schema, ETL, GDS Cypher scripts
├── neo4j_layer/              ← Python package (neo4j pip alias)
├── models/
│   ├── baselines/            ← SVM, RF-GDV, node2vec
│   ├── gnn/                  ← GCN+GAT, Trainer, GNNExplainer, Dataset
│   └── metrics.py            ← Accuracy, Macro-F1, ROC-AUC, Jaccard
├── api/                      ← FastAPI service
├── monitoring/               ← Drift detection + retrain trigger
├── validation/               ← Subject-level split, domain shift
├── dry_run/                  ← 10 smoke tests (no GPU required)
└── scripts/
    ├── train_launcher.py     ← Auto-detecting pipeline runner
    ├── train_v100.sh         ← V100 launch script
    ├── train_dgx_spark.sh    ← DGX Spark launch script
    └── eval_agx.sh           ← AGX train/evaluate script
```
