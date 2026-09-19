# Problem Statement

## Neuro-Neo4j: Hybrid Graph Database and Explainable GNN Pipeline for Brain Biomarker Discovery in Neurodegenerative Disorders

**Author:** basakanimesh14  
**Domain:** Computational Neuroscience · Graph Machine Learning · Clinical AI  
**Task Type:** Graph-level Binary Classification + Explainability

---

## 1. Background

Neurodegenerative disorders — including Alzheimer's disease (AD), Parkinson's disease (PD), and Frontotemporal Dementia (FTD) — collectively affect over **55 million people worldwide** and represent one of the fastest-growing global health crises. By 2050, this number is projected to exceed 150 million due to ageing populations.

Despite decades of research, neurodegenerative diseases remain:

- **Diagnosed late** — clinical symptoms appear only after >50% neuronal loss
- **Poorly understood mechanistically** — the network-level breakdown preceding cell death is largely unmapped
- **Resistant to treatment** — partly because we cannot identify *which* connections break down *when*

Functional MRI (fMRI) captures the brain's resting-state activity by measuring blood-oxygen-level-dependent (BOLD) signals across anatomical regions. When processed into **functional connectivity matrices** — pairwise Pearson correlations between regional BOLD time series — these scans encode the brain's communication topology at a given moment.

The central hypothesis of this project:

> **Neurodegenerative disorders are fundamentally disorders of connectivity, not just cellular loss — and their earliest signatures exist in the topology of functional brain graphs, before clinical symptoms manifest.**

---

## 2. Why Traditional Approaches Fail

### 2.1 Voxel-wise Analysis (Mass Univariate)

The dominant paradigm in neuroimaging applies statistical tests at each of the ~100,000+ brain voxels independently. This approach:

| Limitation | Consequence |
|---|---|
| No relational modelling | Cannot capture inter-regional interaction breakdown |
| Massive multiple comparison burden | Requires aggressive correction (Bonferroni, FDR), destroying statistical power |
| Ignores spatial topology | Treats spatially adjacent voxels as independent observations |
| Not interpretable as a network | Cannot answer: *which connections* are most predictive? |

### 2.2 Standard Deep Learning (CNNs, RNNs)

Convolutional and recurrent neural networks were designed for **Euclidean data** — grids and sequences. Brain functional connectivity data is fundamentally **non-Euclidean**: a graph with no fixed node ordering, variable edge densities, and topology-dependent semantics.

Applying CNNs to flattened connectivity matrices:
- Destroys the graph structure (nodes are reordered arbitrarily)
- Cannot generalise across subjects with different parcellation schemes
- Produces predictions with no mechanistic interpretability

### 2.3 Classical Graph Metrics (Handcrafted Features)

Graph-theoretic measures (global efficiency, clustering coefficient, small-worldness) capture some structural properties but:
- Require manual feature engineering
- Capture only scalar summaries of rich topological information
- Cannot learn task-relevant subgraph patterns from data

---

## 3. Research Gap

Three critical gaps remain unaddressed in existing literature:

### Gap 1: Relational Knowledge is Not Stored Relationally
Neuroimaging studies typically store connectivity matrices as flat files (`.mat`, `.csv`). The **relational structure** — which regions connect, with what strength, across which patient groups — is computed on the fly and discarded. No persistent, queryable knowledge graph of brain connectivity exists.

### Gap 2: Graph Neural Networks Lack Clinical Explainability
Recent GNN-based neuroimaging classifiers achieve strong discriminative performance but are black boxes. Clinicians cannot accept a diagnostic tool that says "diseased" without identifying *which neural pathways* drove that decision.

### Gap 3: Domain Shift Across Scanner Sites is Ignored
fMRI acquisition varies significantly across scanner manufacturers, field strengths (1.5T vs 3T vs 7T), and acquisition protocols. Models trained on one site systematically fail on others — a problem that is rarely measured and almost never corrected for in neuroimaging ML studies.

---

## 4. Problem Formulation

### 4.1 Formal Definition

**Given:** A dataset of $N$ subjects, each represented as a functional brain graph:

$$G_i = (V, E_i, X_i)$$

where:
- $V$ = set of $|V| = 116$ brain regions (AAL-116 atlas)
- $E_i \subseteq V \times V$ = functional connectivity edges (thresholded Pearson correlations)
- $X_i \in \mathbb{R}^{|V| \times F}$ = node feature matrix ($F = 5$ graph-theoretic features per region)
- $y_i \in \{0, 1\}$ = diagnostic label (0 = Healthy Control, 1 = Neurodegenerative)

**Learn:** A graph-level classifier $f: G \to [0,1]$ that predicts disease status, along with an explanation function $e: G \to \mathcal{M}$ that identifies the most diagnostically relevant subgraph $\mathcal{M} \subseteq G$.

### 4.2 Why Graph-Level (Not Node-Level)

The clinical question is: **"Is this patient's brain connectivity pattern consistent with neurodegeneration?"** This is inherently a graph-level question — the pathology involves the *global pattern* of connectivity breakdown, not changes in any single region in isolation.

### 4.3 Constraints

| Constraint | Reason |
|---|---|
| Subject-level data split (no graph leakage) | Multiple scans from one subject are not independent |
| Site-stratified evaluation | Models must generalise across scanner sites |
| Clinically interpretable output | Must identify specific regions and connections, not just class probabilities |
| No data augmentation by label mixing | Clinical ground truth labels are not interpolatable |

---

## 5. Specific Challenges

### Challenge 1: Small Sample Size
Neuroimaging datasets are expensive to collect. A typical multi-site study has 50–500 subjects. Standard deep learning requires thousands. Our GNN must learn from limited data without overfitting — requiring strong regularisation, k-fold cross-validation, and architecture constraints.

### Challenge 2: High Dimensionality, Low Signal
A 116-region connectome has $\binom{116}{2} = 6{,}670$ possible edges. With 50 subjects, the sample-to-feature ratio is ~0.0075 — severely underdetermined. Feature selection (via GDS centrality measures) and graph sparsification (thresholding) are essential preprocessing steps.

### Challenge 3: Class Imbalance
Clinical cohorts are rarely balanced. Diseased patients are harder to recruit and retain. Standard accuracy metrics are misleading; **Macro-F1** and **ROC-AUC** are primary metrics.

### Challenge 4: Multi-Site Heterogeneity
Scanner site acts as a confound that can inflate apparent classification accuracy. A model trained on Site A and tested on Site A may simply learn scanner artifacts rather than disease biology. Domain shift detection and correction are essential.

### Challenge 5: Explainability vs Performance Trade-off
GNNExplainer optimises a mask over edges and node features to explain a specific prediction. For clinical use, this explanation must be:
- Anatomically plausible (top-weighted edges should correspond to known disease circuits)
- Stable across similar patients (Jaccard similarity of top-$k$ subgraphs)
- Actionable (specific enough to guide biomarker hypothesis generation)

---

## 6. Clinical Significance

If solved successfully, this pipeline enables:

1. **Early detection** — identifying neurodegenerative connectivity signatures before cognitive symptoms
2. **Biomarker discovery** — which specific brain circuits (hippocampal-frontal, default mode network) break down first
3. **Treatment monitoring** — tracking connectivity restoration in response to intervention
4. **Multi-site validation** — a knowledge graph that accumulates evidence across institutions

---

## 7. Evaluation Criteria

| Metric | Definition | Clinical Rationale |
|---|---|---|
| **Macro-F1** | Harmonic mean of per-class precision and recall | Equally penalises missing diseased and healthy cases |
| **ROC-AUC** | Area under receiver-operating-characteristic curve | Threshold-independent discrimination |
| **Accuracy** | Correct predictions / total | Overall performance snapshot |
| **Jaccard Stability** | $\|A \cap B\| / \|A \cup B\|$ | Reproducibility of identified biomarker subgraphs |
| **Site Generalisation** | AUC drop across scanner sites | Real-world deployment robustness |

**Target performance (synthetic data baseline):**

| Metric | Minimum | Target |
|---|---|---|
| Macro-F1 | > 0.75 | > 0.85 |
| ROC-AUC | > 0.80 | > 0.90 |
| Jaccard (top-20 edges) | > 0.40 | > 0.60 |
