# STUDY.MD — Viva, Presentation & Cross-Examination Cheat Sheet

> **Project:** Neuro-Neo4j: Hybrid Graph Database and Explainable GNN Pipeline for Brain Biomarker Discovery  
> **Author:** basakanimesh14  
> **Target Audience:** Quick study guide for slide presentations, project defenses, and professor/examiner Q&A.

---

## 1. Core Definitions (Flashcards)

### Graph Basics
* **Node (Vertex $V$):** An entity in the network. In our project: **One of 116 anatomical brain regions** (AAL-116 atlas, e.g., *Left Hippocampus*).
* **Edge ($E$):** A relationship/connection between two nodes. In our project: **Functional connectivity** (Pearson correlation between BOLD signals of two brain regions).
* **Graph ($G = (V, E, X)$):** The entire interconnected network. Here: **1 patient = 1 complete brain graph** (116 nodes).
* **Adjacency Matrix ($A$):** A $116 \times 116$ grid where cell $(i, j)$ represents connection strength between region $i$ and region $j$.

### Prediction Levels
* **Node-Level Task:** Predicting a property for each individual node (e.g., "Is region $X$ damaged?").
* **Edge-Level Task:** Predicting if a link exists (e.g., "Will region $A$ disconnect from region $B$?").
* **Graph-Level Task (OUR PROJECT):** Taking the **entire graph as a single input** and outputting one label for the whole graph (**Healthy vs. Diseased**).

### Graph Metrics & Features (Node Features $X \in \mathbb{R}^{116 \times 5}$)
1. **Degree Centrality:** Count/sum of connections a node has. Measures local connectedness/hubness.
2. **PageRank:** Importance score based on whether neighbors are also important.
3. **Clustering Coefficient:** Tendency of a node's neighbors to connect to each other ("triangles"). Measures local density.
4. **Betweenness Centrality:** How often a node lies on the shortest path between all pairs of nodes. High = crucial bridge/traffic hub.
5. **Eigenvector Centrality:** Influence score proportional to the sum of influence of its neighbors.

### Random Walks & Graph Embeddings
* **Simple Random Walk:** Starting at node $u$, jump to any neighboring node with equal probability. Used to capture local neighborhood structures.
* **Biased Random Walk (node2vec):** Uses two parameters ($p$ and $q$):
  * **$p$ (Return parameter):** Likelihood of immediately returning to the previous node (local exploration / clustering).
  * **$q$ (In-out parameter):** Likelihood of moving outward to unseen nodes (global exploration / structural roles).
* **Graphlet Degree Vector (GDV):** Counts occurrences of small interconnected subgraphs (triangles, squares, wedges) around a node.

### Deep Learning & GNN Concepts
* **Message Passing:** The core mechanism of GNNs. Nodes send their feature vectors to neighbors, aggregate incoming vectors, and update their own representations.
* **GCN (Graph Convolutional Network):** Averages information from all neighbors equally using normalized adjacency. Great for broad structural aggregation.
* **GAT (Graph Attention Network) & GATv2:** Learns an **attention weight $\alpha_{ij}$** for each edge. GATv2 (Brody et al., 2021) provides strictly dynamic attention, avoiding static rank collapse.
* **SwiGLU (Swish Gated Linear Unit):** SOTA feed-forward activation function (used in LLaMA 3, PaLM; Shazeer, 2020). Multiplies a Swish-activated projection with a linear gating projection: $\text{SwiGLU}(x) = (\text{Swish}(x W_1) \otimes x W_2) W_3$. Acts as a dynamic filter suppressing scanner SNR noise.
* **Global Pooling (Readout):** Compresses 116 node vectors into **one single graph vector** using mean pooling ($\frac{1}{|V|} \sum h_v$) so an MLP can classify it.
* **GNNExplainer:** Optimization method that freezes model weights and learns a mask on edges ($m_E$) and node features ($m_F$) to find the minimal subgraph that produces the disease prediction.

---

## 2. The 60-Second Elevator Pitch

> *"Traditional neuroimaging flattens 3D brain scans into Euclidean grids or voxels, missing long-range network breakdowns. Neuro-Neo4j models patient fMRI scans as **relational graphs** using the AAL-116 atlas. We store and analyze these connectomes in **Neo4j** using GDS centrality features, classify whole brain graphs using a **Hybrid GCN+GAT** architecture, and explain clinical predictions using **GNNExplainer** to extract patient-specific biomarker subgraphs (e.g. Default Mode Network disconnections). The entire pipeline is productionized with a FastAPI service and drift monitoring."*

---

## 3. Architecture & Data Flow (What Happens Under the Hood)

```
[fMRI BOLD Signal / Correlation Matrix]
                │
                ▼
1. Preprocessing (Regress global signal → Z-score → Threshold |r| ≥ 0.3)
                │
                ▼
2. Neo4j Knowledge Graph (Store (:BrainRegion) & [:CONNECTED_TO])
   └── Compute 5 centrality features per node (GDS / NetworkX mock)
                │
                ▼
3. PyG BrainConnectomeDataset (116 nodes × 5 features + edge_index)
                │
                ▼
4. Hybrid GNN Model:
   ├── GCN Layers (Structural neighborhood encoding)
   ├── GAT Layers (Attention-weighted edge refinement)
   ├── Global Mean Pooling (116 node vectors → 1 graph vector)
   └── MLP Classifier → [P(Healthy), P(Diseased)]
                │
                ▼
5. GNNExplainer:
   └── Extracts Top-k disrupted edges & regions as clinical biomarkers
```

---

## 4. Why Did We Choose X? (Justifications)

| Decision | Why? | Alternatives Considered |
|---|---|---|
| **AAL-116 Atlas** | 116 anatomically validated regions covering cerebrum & cerebellum; standard in neuroimaging literature. | Schaefer-100, Destrieux-148. |
| **Pearson Threshold 0.3** | Removes noise/spurious correlations while keeping meaningful network pathways (~7-15% edge density). | 0.2 (too noisy/dense), 0.5 (disconnected islands). |
| **GCN + GAT Hybrid** | GCN handles overall neighborhood topology; GAT learns specific attention weights on critical pathways without high compute cost. | Pure GCN (no attention weighting), Pure GAT (overfitting on small graphs). |
| **Neo4j + GDS** | Brain connectivity is naturally a graph. Enables Cypher queries across clinical cohorts and in-database graph algorithms. | Flat `.csv`/SQL tables (slow relational queries, no native graph algorithms). |
| **Macro-F1 over Accuracy** | Clinical cohorts can be imbalanced; Macro-F1 equally penalizes missed disease cases and false alarms. | Standard Accuracy (misleading when classes are skewed). |

---

## 5. Top 15 Cross-Questioning (Viva / Examiner Q&A)

### Q1: Why is this formulated as graph-level classification and not node-level?
> **Answer:** "Neurodegeneration affects widespread distributed neural networks, not just isolated cells. The question is: *'Does this patient have Alzheimer's?'* That diagnosis belongs to the patient's entire brain graph, not an individual region."

### Q2: Why did you use Pearson correlation and threshold at 0.3?
> **Answer:** "Pearson correlation measures functional co-activation between regions over time. Weak correlations ($|r| < 0.3$) represent acquisition noise or physiological artifacts. Thresholding at 0.3 is standard in literature (e.g. Bullmore & Sporns) to retain sparse, topologically meaningful connectomes."

### Q3: How do you prevent data leakage in your train/val/test splits?
> **Answer:** "We perform **subject-level stratified splitting**. If a patient has multiple scans, all scans for that patient remain strictly inside the train set or test set. We never shuffle at the graph slice level."

### Q4: Why combine GCN and GAT instead of using just one?
> **Answer:** "GCN is fast and aggregates isotropic neighborhood structure. GAT adds multi-head attention to learn anisotropic importance weights on edges. Combining them gives structural stability from GCN and selective pathway focus from GAT without explosive parameter counts."

### Q5: What are the 5 node features and why did you pick them?
> **Answer:** "Degree, PageRank, Local Clustering Coefficient, Betweenness Centrality, and Eigenvector Centrality. They capture both local clustering (segregation) and global bridging (integration), which are known to degrade in neurodegenerative diseases."

### Q6: How does GNNExplainer work mathematically?
> **Answer:** "It freezes the trained GNN weights and maximizes Mutual Information between the model's prediction and a masked graph. It optimizes a soft continuous mask $m_E \in [0, 1]^{|E|}$ and thresholding isolates the most influential edges."

### Q7: What baseline models did you compare against?
> **Answer:** "1. **RBF-SVM** on flattened upper-triangle correlations (6,670 features).  
> 2. **Random Forest + Graphlet Degree Vectors (RF-GDV)** capturing 5 orbit motifs.  
> 3. **node2vec + Logistic Regression** using biased random walks."

### Q8: Why did node2vec need a fallback in your setup?
> **Answer:** "The standard `node2vec` pip package relies on `gensim` C-extensions that fail compilation on newer Python environments (e.g., Python 3.14). We built a spectral/NetworkX fallback so the pipeline remains resilient on any environment."

### Q9: What hardware constraints did you design for?
> **Answer:** "AGX Orin (unified memory, 64GB, FP16, `pin_memory=False`), Tesla V100 (dedicated VRAM, FP16 via AMP GradScaler, `pin_memory=True`), and DGX Spark (Blackwell, BF16 native). Both AGX and V100 are strictly dedicated to training/testing, not hosting."

### Q10: Why can't we use BF16 on the V100 server?
> **Answer:** "The Tesla V100 is based on Volta architecture (Compute Capability 7.0), which lacks hardware instructions for bfloat16. BF16 requires Ampere (SM 8.0+) or newer. Therefore, V100 uses FP16 with `torch.cuda.amp.GradScaler`."

### Q11: What is the Jaccard stability metric in your evaluation?
> **Answer:** "$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$. It measures the overlap of top-20 biomarker edges identified across different test subjects with the same diagnosis. High Jaccard means the biomarkers are consistent and reproducible, not random noise."

### Q12: How does the monitoring and retrain trigger work?
> **Answer:** "A sliding window of recent predictions monitors Macro-F1. If Macro-F1 drops by more than 5% relative to the 0.80 baseline across 3 consecutive windows, a drift alert is logged and an automated retraining trigger is flagged."

### Q13: Why does your model have only ~50K to 100K parameters? Isn't that small?
> **Answer:** "Brain connectomes have 116 nodes. With cohorts of 250–500 subjects, models with millions of parameters (like LLMs or large CNNs) would immediately overfit. A lightweight 50K–100K parameter model provides the right inductive bias and prevents memorization."

### Q14: How does Neo4j interact with PyTorch Geometric?
> **Answer:** "Neo4j acts as the persistent knowledge store. Graph Data Science (GDS) projections compute centrality features. These are exported to an edge-list and metadata format, which PyG loads into memory as `torch_geometric.data.Data` objects."

### Q15: What are the main limitations of this study?
> **Answer:** "Currently evaluated on synthetic AAL-116 fMRI data with controlled perturbations. Next steps are validation on clinical databases (ADNI, PPMI), multi-class classification (AD vs. PD vs. Controls), and multi-modal integration with DTI structural tractography."

### Q16: What is SwiGLU and why did you use it in the GNN readout head?
> **Answer:** "SwiGLU (Swish Gated Linear Unit) is the state-of-the-art feed-forward activation function from modern LLMs (LLaMA 3, PaLM). Traditional activations like ReLU suffer from dead neurons and have no gating mechanism. SwiGLU introduces a multiplicative gate $(\text{Swish}(x W_1) \otimes x W_2) W_3$ that dynamically suppresses noisy dimensional fluctuations from scanner site SNR shifts while allowing subtle disease signals to pass through."

### Q17: On the held-out test set, ROC-AUC was ~0.713 but Accuracy was ~57%. Why the gap, and how did you fix it?
> **Answer:** "Default accuracy uses an uncalibrated $p = 0.5$ threshold. Multi-site scanner variance shifts the predicted probability distribution. The high ROC-AUC of 0.713 proves the model ranks patients correctly. We applied **Youden's J-statistic** ($J = \text{TPR} - \text{FPR}$) on validation predictions to find the optimal decision boundary $t^*$, which realigns operational accuracy and Macro-F1 with the true 71%+ discriminative AUC."

### Q18: How do your empirical results from GATv1 compare to the enhanced GNN architecture?
> **Answer:** "Our baseline empirical run on the Tesla V100 with unweighted GATv1 + standard ReLU achieved a peak Validation ROC-AUC of 0.7966 and a held-out Test ROC-AUC of 0.7127 (calibrated accuracy 68.00%). Upgrading to **edge-weight aware message passing** (`edge_attr`), **dual global pooling** (`[mean_pool, max_pool]`), **smooth continuous SiLU non-linearities**, and **SwiGLU gating** pushed peak Validation ROC-AUC to **`0.8329`** (Val Macro-F1: `0.7652`), held-out Test ROC-AUC to **`0.7461`** (+3.34%), Calibrated Test Accuracy to **`73.33%`** (+5.33%), and Calibrated Macro-F1 to **`0.7310`** (+5.15%)."

### Q19: Why was there previously a plateau in raw gain, and how did edge attributes + dual pooling break through it?
> **Answer:** "Upgrading only the 2-layer classifier head with SwiGLU operated downstream of graph feature extraction. The underlying graph convolutions were still discarding all functional connection correlation weights (`edge_attr`), treating connectomes as binary graphs. Furthermore, `global_mean_pool` averaged 116 regional embeddings, diluting focal circuit breakdowns across the healthy majority. By passing Pearson $|r|$ to GCN and $r$ to GATv2, the convolutions directly learn connection degradation. Concatenating `[mean_pool, max_pool]` preserves both whole-brain connectivity background and focal outlier lesions."

---

## 6. Slide-by-Slide Presentation Structure (from ppt.md)

* **Slide 1: Title & Executive Summary:** Overview, clinical goal, peak 0.833 Val AUC / 0.746 Test AUC, 73.33% calibrated test accuracy, V100 optimization.
* **Slide 2: Clinical Problem & Preprocessing:** 500-subject multi-site cohort, AAL-116 atlas, $|r| \ge 0.3$, zero data leakage split.
* **Slide 3: Hardware & Engineering Stack:** Tesla V100 FP16 mixed precision, PCIe pinned memory, dataset caching ($16\text{ min} \to 0.05\text{s}$).
* **Slide 4: Deep Learning Architecture (3 Breakthrough Pillars):** Edge-weight awareness (`edge_attr`), Dual Pooling (`[mean, max]`), Smooth SiLU + SwiGLU.
* **Slide 5: Training Dynamics:** Convergence curve on V100 (1.33s/epoch), peak Val AUC 0.8329 at epoch 70 with early stopping.
* **Slide 6: Held-Out Test Evaluation:** Test ROC-AUC 0.7461, Youden's J calibration ($t^* = 0.1762$, $J = 0.4644 \to 73.33\%$ Acc, 0.7310 F1).
* **Slide 7: Architectural Ablation Study:** Comprehensive benchmark table: SVM vs. RF-GDV vs. node2vec vs. GATv1 vs. GATv2+SwiGLU vs. Ours Full.
* **Slide 8: Explainable AI:** GNNExplainer module running natively on GPU across 75 test subjects.
* **Slide 9: Discovered Biomarkers:** 100% consensus Cerebellar Vermis network, Hippocampus, Amygdala, and fronto-temporal circuits.
* **Slide 10: System Deployment:** Decoupled V100 training cluster vs. FastAPI CPU clinical serving with Prometheus monitoring.
* **Slide 11: Summary & Next Steps:** Key achievements and clinical roadmap (ADNI, multi-GPU DDP).
