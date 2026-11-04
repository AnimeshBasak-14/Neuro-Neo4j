# Presentation 2: Slide-by-Slide Content

**Deck:** Explainable GNNs & Hybrid Graph Databases for Brain Biomarker Discovery, Part 2 (Execution & Results)
**Template:** `IITR_template_pptx(19062016)`, same layout as Presentation 1

**Layout rules copied from deck 1.** Black two-line bold title at top left with the blue rule under it. IITR logo top right. Content sits in rounded rectangles with a centred bold navy heading ("A. …", "B. …"). Body text is 12–14 pt Calibri. Footer: "IIT ROORKEE" bar and page number. Use a **red** border on boxes that state a problem or limitation and a **green** border on boxes that state a result or fix, as in deck 1.

**Equations:** paste the LaTeX below into PowerPoint via *Insert → Equation → LaTeX* (Office 365). Each equation is written to fit one line at 14 pt.

**Every number on these slides comes from `presentation2/figures/verified_numbers.json`, `logs/test_metrics.json` or the checkpoint history.** Rerun `presentation2/make_figures.py` to regenerate them. Rows marked † come from the README and have no run log. Re-run them before the presentation (see Appendix C) or keep the † marker.

---

## Slide 1: Title

**INDIAN INSTITUTE OF TECHNOLOGY ROORKEE** (gradient header bar, as in deck 1)

**Explainable Graph Neural Networks & Hybrid Graph Databases for Brain Biomarker Discovery**
*Part 2: Pipeline Execution and Results*

Animesh Basak (26911001)
Department of Computer Science & Engineering
Indian Institute of Technology Roorkee

(Keep the campus photo strip from slide 1 of deck 1.)

---

## Slide 2: Recap, Problem, Idea and Initial Features

**Box A (blue border). Recap: Problem Statement**
- Functional disconnection in resting-state fMRI appears years before structural atrophy. The goal is to catch it from the connectome alone.
- Task: graph-level binary classification $P(y_k=1\mid G_k)=\mathrm{softmax}(f_\theta(A_k,X_k))_1$, Healthy (0) vs Neurodegenerative (1), plus a post-hoc explanation subgraph $G_s\subseteq G_k$.
- Each subject is a graph $G_k=(V,E_k,X_k)$ on the AAL-116 atlas: $|V|=116$ regions. Edges are thresholded functional correlations.
- Core idea from deck 1: edge-weight-aware GCN, then GATv2 dynamic attention with the edge weight as an attribute, then dual pooling, a SwiGLU head and GNNExplainer for biomarker subgraphs.

**Box B (green border). Initial Features Selected and Initial Feature Engineering**

| Level | Feature | How it is computed |
|---|---|---|
| Node (5-dim) | Degree | $k_i/\max_j k_j$, the binary neighbour count scaled to [0,1] |
| | PageRank | power iteration on $\lvert z_{ij}\rvert$-weighted graph, damping 0.85 |
| | Clustering coeff. | weighted (Onnela) geometric-mean triangle intensity |
| | Betweenness | normalised shortest-path betweenness |
| | Eigenvector centrality | leading eigenvector of $\lvert W\rvert$, scaled by its max |
| Edge (1-dim) | $z_{ij}$ | signed z-scored correlation (GATv2); $\lvert z_{ij}\rvert$ is the GCN edge weight |

Initial engineering: global-signal removal → z-scoring of off-diagonal entries → pruning of $\lvert z_{ij}\rvert<0.3$ → symmetric COO edge list (both directions) → NetworkX/Neo4j-GDS centralities.

*Speaker note:* Deck 1 wrote the threshold as $\lvert r_{ij}\rvert\ge 0.3$. In the code it is applied **after** z-scoring, so it is $\lvert z_{ij}\rvert\ge 0.3$. Say this out loud; it explains why each graph keeps 76% of all possible edges.

---

## Slide 3: Pipeline Step 1, Data Collection

**Box A (blue). What we used and why**
- Clinical cohorts (ADNI, PPMI) need signed data-use agreements, which are still pending. To build and debug the whole pipeline end to end, we generated a **synthetic AAL-116 cohort with a planted, known disease signature**.
- $N=500$ subjects: 250 Healthy, 250 Diseased (labels alternate by index), 3 scanner sites assigned round-robin, ages 50–84.
- Because the ground truth is planted, we can check the **explanations** against it as well as the accuracy. Real data cannot give us that check.

**Box B (green). Generative model (`data/synthetic/generate_synthetic.py`)**

$$\Sigma = BB^\top + nI,\quad B_{ij}\sim\mathcal N(0,1),\quad R_{ij}=\frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}$$

Disease perturbation (Diseased only):

$$R_{i,i+1}\leftarrow 0.4\,R_{i,i+1}\ \ (\text{45 homotopic L–R pairs}),\qquad R_{ij}\leftarrow \min(1,\,1.3\,R_{ij})\ \ (i,j\ \text{same lobe})$$

Scanner noise: $R\leftarrow \mathrm{clip}\big(R+\tfrac12(\varepsilon+\varepsilon^\top),-1,1\big)$, $\varepsilon_{ij}\sim\mathcal N(0,\sigma_s^2)$, $\sigma_s\in\{0.02,0.04,0.06\}$ for sites A/B/C.

**Visual:** small 3-column table (build in PowerPoint): *Site | σ | Subjects*: A 0.02 167, B 0.04 167, C 0.06 166.

*Speaker note:* off-diagonal $R_{ij}$ has a standard deviation of about 0.047, so the site noise (0.02–0.06) is as large as the signal. That is deliberate.

---

## Slide 4: Pipeline Step 2, Data Cleaning and Preprocessing

**Box A (blue). Four transforms per subject (`neo4j_layer/etl/transform.py`)**

1. Global signal removal: $\tilde R = R-\mu_g\mathbf 1\mathbf 1^\top,\ \ \mu_g=\frac{1}{n(n-1)}\sum_{i\ne j}R_{ij}$, then the diagonal is reset to 1.
2. Z-scoring of off-diagonal entries: $z_{ij}=\dfrac{\tilde R_{ij}-\mu}{\sigma}$, with $\mu,\sigma$ taken over all $i\neq j$ of that subject; diagonal set to 0.
3. Sparsification: $W_{ij}=z_{ij}\cdot\mathbb 1[\lvert z_{ij}\rvert\ge 0.3]$.
4. Edge list: upper triangle $i<j$ with $W_{ij}\ne 0$ → `edge_list.csv` (2,545,839 rows) → Neo4j `LOAD CSV`.

**Box B (green). Leakage control**
- Subject-level stratified split, seed 42: **350 / 75 / 75** (train / val / test), class ratio kept in every split.
- Rows are mapped by `PatientID`, so no subject appears in two splits. This was checked with an assertion.
- All statistics in steps 1–3 are **per subject**, so nothing from the test set enters training.

**Visual:** flowchart, 4 boxes left to right (see Appendix B, Flowchart 1).

---

## Slide 5: Pipeline Step 3, Exploratory Data Analysis

**Box A (blue). Graph statistics**
- Edges kept per subject: **5,092 ± 34** of 6,670 possible (76.3% density). The two classes overlap almost completely (Fig 1, right).
- Edge weights: mean ≈ 0, SD 1.14, 50.0% negative. The distributions of the two classes look the same by eye (Fig 1, left).

**Box B (red). Where the class signal actually lives**
- Group-difference matrix $\Delta_{ij}=\overline{\lvert W_{ij}\rvert}_{D}-\overline{\lvert W_{ij}\rvert}_{H}$ (Fig 3): red blocks on every within-lobe square and a blue line on the homotopic L–R pairs. This is exactly the planted signature.
- One scalar per subject, $s_k=\mathrm{mean}_{(i,j)\in\text{intra-lobe}}\lvert W_{ij}\rvert$, separates the test set perfectly: **AUC = 1.000** (Fig 2).
- Consequence for modelling: the signal is a **block-level magnitude pattern that depends on region identity**. A model has to know which lobe each node belongs to before it can see it.

**Visuals:** Fig 1 (`fig1_eda_edges.png`), Fig 2 (`fig2_eda_signal_location.png`), Fig 3 (`fig3_group_difference_matrix.png`). If space is tight, use Fig 3 alone; it tells the whole story.

---

## Slide 6: Pipeline Step 4, Feature Engineering and Selection

**Box A (blue). Node descriptors** ($w_{ij}=\lvert W_{ij}\rvert$, $k_i=\lvert\mathcal N(i)\rvert$)

$$x_i^{\deg}=\frac{k_i}{\max_j k_j},\qquad \pi = \frac{1-d}{n}\mathbf 1 + d\,P^\top\pi,\ \ P_{ij}=\frac{w_{ij}}{\sum_l w_{il}},\ d=0.85$$

$$C_i=\frac{1}{k_i(k_i-1)}\sum_{j,h}\big(\hat w_{ij}\hat w_{ih}\hat w_{jh}\big)^{1/3},\qquad b_i=\frac{2}{(n-1)(n-2)}\sum_{s\ne i\ne t}\frac{\sigma_{st}(i)}{\sigma_{st}},\qquad \lambda_1 e = We$$

**Box B (red). Selection audit**
- All five features are **scale-free**: each is divided by a maximum, sums to one, or is a ratio of path counts. Multiplying every weight inside a lobe by 1.3 barely moves them, so the planted signal is mostly removed **before** the GNN ever sees it.
- NetworkX reads `weight` as a **distance** in betweenness, so strong correlations count as long paths. The feature is inverted relative to its usual meaning.
- What carries the signal: the edge attribute $z_{ij}$ (kept), plus region identity (missing; see slide 10).

*Speaker note:* this slide is where you show you understand *why* the results come out the way they do. Do not skip it.

---

## Slide 7: Pipeline Step 5, Model Selection

**Box A (blue). Architecture: 521,378 parameters, $d=192$**

$$h_i^{(0)}=\mathrm{SiLU}(W_{\text{in}}x_i+b_{\text{in}})\in\mathbb R^{192}$$

GCN ×3 with residual:
$$h_i^{(l+1)}=h_i^{(l)}+\mathrm{Drop}\Big(\mathrm{SiLU}\big(\mathrm{BN}\big(\textstyle\sum_{j\in\mathcal N(i)\cup\{i\}}\frac{\tilde w_{ij}}{\sqrt{\tilde d_i\tilde d_j}}\,\Theta^{(l)}h_j^{(l)}\big)\big)\Big),\quad \tilde d_i=1+\sum_j w_{ij}$$

GATv2 ×1, $H=4$ heads of 48:
$$e_{ij}^{k}=a_k^\top\mathrm{LeakyReLU}_{0.2}\!\big(W_s^k h_i+W_t^k h_j+W_e^k z_{ij}\big),\qquad \alpha_{ij}^{k}=\frac{\exp e_{ij}^{k}}{\sum_{l\in\mathcal N(i)}\exp e_{il}^{k}}$$
$$h_i'=h_i+\mathrm{Drop}\Big(\mathrm{SiLU}\big(\mathrm{BN}\big(\big\Vert_{k=1}^{4}\textstyle\sum_{j}\alpha_{ij}^{k}W_t^k h_j\big)\big)\Big)$$

Readout and head:
$$h_G=\Big[\tfrac{1}{116}\textstyle\sum_i h_i'\ \Big\Vert\ \max_i h_i'\Big]\in\mathbb R^{384},\quad u=\big(\mathrm{SiLU}(h_GW_g)\odot h_GW_u\big)W_d\in\mathbb R^{96},\quad z=W_c\,\mathrm{Drop}(u)\in\mathbb R^{2}$$

**Box B (green). Why each block (one line each)**
- GCN: cheap isotropic smoothing that uses the continuous coupling strength $w_{ij}$.
- GATv2: the nonlinearity sits between $W_sh_i$ and $a$, so neighbour ranking can change with the query node $i$ (GATv1 cannot do this; Brody et al. 2022).
- Mean ‖ max: mean keeps the global level and max keeps single-hub extremes.
- SwiGLU: multiplicative gate. It is a design choice and needs an ablation before we claim it removes scanner noise.

**Visual:** architecture diagram (Appendix B, Flowchart 2).

---

## Slide 8: Pipeline Step 6, Model Training

**Box A (blue). Objective and optimiser**

$$\mathcal L(\theta)=-\frac1{|\mathcal B|}\sum_{k\in\mathcal B}\sum_{c\in\{0,1\}}q_{kc}\log\mathrm{softmax}(z_k)_c,\qquad q_{kc}=(1-\epsilon)\,\mathbb 1[c=y_k]+\tfrac{\epsilon}{2},\ \ \epsilon=0.1$$

AdamW ($\eta_0=10^{-3}$, $\lambda=10^{-4}$, decoupled):
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\ \ v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\ \ \theta_{t+1}=\theta_t-\eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}+\lambda\theta_t\Big)$$

Schedule: $\eta_t=\eta_0\frac{t+1}{5}$ for $t<5$, then $\eta_t=\eta_0\cdot\tfrac12\big(1+\cos\pi\frac{t-5}{195}\big)$.

**Box B (green). V100 run settings and outcome**
- FP16 autocast + `GradScaler` (Volta has no BF16), grad-norm clip 1.0, batch 32, `pin_memory=True`.
- Early stopping on **validation Macro-F1**, patience 25. Best checkpoint at **epoch 70**: val F1 0.765, val AUC 0.822, train F1 0.634.
- Train F1 stays below val F1 for the whole run, so the model **underfits**: heavy dropout, label smoothing and scale-free inputs together. It does not overfit.

**Visual:** Fig 5 (`fig5_training_curves.png`).

---

## Slide 9: Pipeline Step 7, Model Evaluation and Tuning

**Box A (blue). Metrics on the held-out test set ($N=75$)**

$$\mathrm{AUC}=P\big(\hat p_{k^+}>\hat p_{k^-}\big),\qquad \mathrm{F1}_{\text{macro}}=\tfrac12\textstyle\sum_{c}\frac{2\,\mathrm{TP}_c}{2\,\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c},\qquad J(t)=\mathrm{TPR}(t)-\mathrm{FPR}(t)$$

| Metric | Value |
|---|---|
| **Test ROC-AUC (headline)** | **0.746** |
| Validation ROC-AUC | 0.823 |
| Test accuracy / macro-F1 at default threshold 0.5 | 57.33% / 0.520 |

AUC is threshold-free, so it is the main metric. The test set is small (75 subjects), so accuracy and F1 move a lot with the threshold.

**Box B (green). Tuning: threshold calibration**
- Youden's $J$ picks the threshold $t^*$ that maximises TPR − FPR on the ROC curve. It is the standard way to set an operating point for a screening tool.
- Test scores sit lower than validation scores (mean $\hat p=0.27$). The ranking is fine (AUC 0.746) but the probability scale shifts between splits.
- Next step: fit $t^*$ on validation only, add temperature scaling, and run 5-fold CV.

*Speaker note (do not put on slide):* the 73.33% accuracy in the README used $t^*$ chosen on the test set, so do not quote it. If asked, say threshold calibration on a 75-subject test set is optimistic and that AUC is the reliable number.

**Visual:** Fig 6 (`fig6_roc_thresholds.png`): ROC with the two thresholds marked.

---

## Slide 10: Results, Explainability and Comparison with the 4 Base Papers

**Box A. Comparison on the same 75 test subjects** (build as a native PowerPoint table)

| Model (base paper) | Graph-aware | Test AUC | Test Acc | Macro-F1 | Source |
|---|---|---|---|---|---|
| RBF-SVM on flattened connectome (Abraham 2017) | No | 0.530 | 53.3% | 0.533 | re-run, this split |
| L2 logistic regression, flattened | No | 0.509 | 49.3% | 0.493 | re-run, this split |
| RF on graphlet degree vectors (Pržulj 2007) | Motifs | 0.692† | 56.0%† | 0.534† | README, no log |
| node2vec + classifier (Grover 2016) | Walks | 0.635† | 52.0%† | 0.498† | README, no log |
| GATv1, mean pool (Veličković 2018) | Static attn. | 0.713† | 57.3%† | 0.550† | README, no log |
| **Ours: edge-aware GCN + GATv2 + dual pool + SwiGLU** | **Dynamic attn.** | **0.746** | **57.3%** | **0.520** | `test_metrics.json` |
| Sanity check: 1 scalar, mean $\lvert z\rvert$ intra-lobe | (oracle) | 1.000 | 100% | 1.000 | re-run, this split |

*† Baseline numbers are from our earlier README runs. Only the SVM / logistic regression / one-scalar rows were re-run on this exact split.*

**Box B (green). What improved and why**
- Against the flattened SVM, the GNN gains **+0.216 AUC** (0.530 → 0.746). The SVM sees 6,670 independent coordinates from only 350 training subjects. The GNN shares $\Theta$ across all edges and nodes, so its parameter count does not grow with $n^2$.
- The edge attribute matters: $z_{ij}$ enters both the GCN normalisation and the GATv2 score, and it is the only input that still holds the magnitude signal after the scale-free node features.

**Box C (blue). Explainability: what the pipeline produces**
- GNNExplainer (500 epochs, lr 0.01) learns a soft edge mask and a feature mask per subject with the model frozen. We keep the top-10 edges and top regions per subject and aggregate them into a consensus circuit across subjects.
- The data is **synthetic with a planted disease pattern** (weaker left–right coupling, stronger within-lobe coupling), so no real biomarker has been discovered. What we built is the extraction pipeline.
- Because the planted pattern is known, we can score the explanations: **15.3%** of top-10 edges are within-lobe, close to the 16.0% expected by chance. So the explainer does not yet recover the planted pattern.
- Why: the model has no region identity (its five node features do not say which lobe a region belongs to), so it cannot use the within-lobe pattern, and a faithful explainer cannot report what the model does not use.
- Future work: add region/lobe identity as a positional feature, then re-check explanation precision. On real data (ADNI/PPMI) the same pipeline would be used.

**Visuals:** Fig 7 (`fig7_model_comparison.png`) and Fig 8 (`fig8_explainer_vs_ground_truth.png`). If there is only room for one, use Fig 8.

*Speaker note:* lead with the +0.216 AUC over the SVM, then show Fig 8 yourself before anyone asks. A panel that hears the chance-level explainer result from you will trust the rest of the talk more.

---

## Slide 11 (optional): Thank You

Same as deck 1.

---

# Appendix A: Graph Visualisations (Neo4j Cypher + Python)

## A.1 Python figures (already generated)

```bash
CUDA_VISIBLE_DEVICES= uv run python presentation2/make_figures.py
# → presentation2/figures/fig1..fig8 (.png at 300 DPI and .svg) + verified_numbers.json
```

| File | Use on slide |
|---|---|
| `fig1_eda_edges` | 5 |
| `fig2_eda_signal_location` | 5 |
| `fig3_group_difference_matrix` | 5 (main) |
| `fig4_brain_graph_subject` | 2 or 3 (a picture of "a brain as a graph") |
| `fig5_training_curves` | 8 |
| `fig6_roc_thresholds` | 9 |
| `fig7_model_comparison` | 10 |
| `fig8_explainer_vs_ground_truth` | 10 |

Insert the `.svg` files where you can (Office 365 keeps them sharp and lets you recolour them).

## A.2 Neo4j: load the graph

Create the region file that `load_csv.cypher` expects:

```python
# presentation2: run once from repo root
import pandas as pd
from data.synthetic.generate_synthetic import AAL116_REGIONS
pd.DataFrame({
    "atlasId": range(116),
    "anatomicalLabel": AAL116_REGIONS,
    "hemisphere": [n[-1] if n[-2] == "_" and n[-1] in "LR" else "M" for n in AAL116_REGIONS],
}).to_csv("data/processed/brain_regions.csv", index=False)
```

Copy `subjects.csv`, `brain_regions.csv` and `edge_list.csv` into `<neo4j>/import/data/`, then run `neo4j_layer/schema/constraints.cypher` and `neo4j_layer/etl/load_csv.cypher`. Tip: load only the test subjects if you just need visuals, since 2.5 M relationships takes a while.

## A.3 Cypher queries for exportable visuals (Neo4j Browser or Bloom)

**Q1. One subject's strongest connections (Browser graph view, then the export SVG button)**
```cypher
MATCH (a:BrainRegion)-[e:CONNECTED_TO {subjectId: 'sub_0001'}]->(b:BrainRegion)
WHERE abs(e.weight) >= 2.0
RETURN a, e, b;
```
Style in Browser: node caption `anatomicalLabel`, relationship thickness by `weight`.

**Q2. Group-difference table (the data behind Fig 3, exportable to CSV/Excel)**
```cypher
MATCH (s:Subject)
WITH s.patientId AS pid, s.diagnosis AS dx
MATCH (a:BrainRegion)-[e:CONNECTED_TO {subjectId: pid}]->(b:BrainRegion)
WITH a.anatomicalLabel AS src, b.anatomicalLabel AS tgt, dx, avg(abs(e.weight)) AS meanAbs
WITH src, tgt,
     sum(CASE dx WHEN 'Diseased' THEN meanAbs ELSE 0 END) AS dMean,
     sum(CASE dx WHEN 'Healthy'  THEN meanAbs ELSE 0 END) AS hMean
RETURN src, tgt, round(dMean - hMean, 4) AS delta
ORDER BY abs(delta) DESC
LIMIT 50;
```

**Q3. Hub regions per subject with GDS PageRank**
```cypher
CALL gds.graph.project.cypher(
  'sub_0001_graph',
  'MATCH (r:BrainRegion) RETURN id(r) AS id',
  'MATCH (a:BrainRegion)-[e:CONNECTED_TO {subjectId: "sub_0001"}]->(b:BrainRegion)
   RETURN id(a) AS source, id(b) AS target, abs(e.weight) AS weight'
);
CALL gds.pageRank.stream('sub_0001_graph', {relationshipWeightProperty: 'weight'})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).anatomicalLabel AS region, round(score, 4) AS pagerank
ORDER BY pagerank DESC LIMIT 15;
CALL gds.graph.drop('sub_0001_graph');
```

**Q4. Write explainer edges back, then view them as a biomarker overlay**

Python (writes `logs/biomarker_explanations.json` into Neo4j):
```python
import json
from neo4j import GraphDatabase
drv = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neuro_password"))
rows = [{"sid": s["subject_id"], "src": e["source"], "tgt": e["target"], "imp": e["importance"]}
        for s in json.load(open("logs/biomarker_explanations.json")) for e in s["top_edges"]]
with drv.session() as ses:
    ses.run("""
      UNWIND $rows AS r
      MATCH (a:BrainRegion {anatomicalLabel: r.src}), (b:BrainRegion {anatomicalLabel: r.tgt})
      MERGE (a)-[x:EXPLAINS {subjectId: r.sid}]->(b)
      SET x.importance = r.imp
    """, rows=rows)
drv.close()
```
Cypher (consensus edges chosen by the explainer in at least 2 subjects):
```cypher
MATCH (a:BrainRegion)-[x:EXPLAINS]->(b:BrainRegion)
WITH a, b, count(x) AS nSubjects, avg(x.importance) AS meanImp
WHERE nSubjects >= 2
RETURN a, b, nSubjects, meanImp
ORDER BY nSubjects DESC, meanImp DESC;
```

**Q5. Ground-truth check inside the database (precision of the explainer)**
```cypher
MATCH (a:BrainRegion)-[x:EXPLAINS]->(b:BrainRegion)
WITH x,
     CASE WHEN a.atlasId < 28 THEN 0 WHEN a.atlasId < 42 THEN 1 WHEN a.atlasId < 56 THEN 2
          WHEN a.atlasId < 70 THEN 3 WHEN a.atlasId < 80 THEN 4 WHEN a.atlasId < 90 THEN 5 ELSE 6 END AS la,
     CASE WHEN b.atlasId < 28 THEN 0 WHEN b.atlasId < 42 THEN 1 WHEN b.atlasId < 56 THEN 2
          WHEN b.atlasId < 70 THEN 3 WHEN b.atlasId < 80 THEN 4 WHEN b.atlasId < 90 THEN 5 ELSE 6 END AS lb
RETURN count(x) AS explainedEdges,
       sum(CASE WHEN la = lb THEN 1 ELSE 0 END) AS intraLobe,
       round(toFloat(sum(CASE WHEN la = lb THEN 1 ELSE 0 END)) / count(x), 3) AS precision;
// expected: 750, 115, 0.153  (chance = 0.160)
```

---

# Appendix B: Flowcharts, Diagrams and Tables (build natively)

## Flowchart 1: Data pipeline (slide 4). PowerPoint SmartArt "Basic Process", or shapes

| # | Shape | Text (line 1 bold / line 2 small) | Fill |
|---|---|---|---|
| 1 | Rounded rectangle | **Synthetic cohort** / 500 × (116×116) R, 3 sites | light blue `#DCEBFA` |
| 2 | Rounded rectangle | **Global signal removal** / R − μ_g | light blue |
| 3 | Rounded rectangle | **Z-score** / per subject, off-diagonal | light blue |
| 4 | Rounded rectangle | **Prune \|z\| < 0.3** / 5,092 edges kept | light blue |
| 5 | Can (cylinder) | **Neo4j** / (:BrainRegion)-[:CONNECTED_TO]-> | light green `#DDF2E4` |
| 6 | Rounded rectangle | **GDS / NetworkX features** / 5 per node | light green |
| 7 | Rounded rectangle | **PyG Data** / x ∈ R^{116×5}, edge_attr ∈ R^{E×1} | light green |

Connectors: block arrows, 1.5 pt, navy `#1F3864`. Put a dashed rectangle around 2–4 labelled "per-subject (no leakage)". Under shape 7 add a three-way split arrow labelled "350 / 75 / 75 (stratified, by PatientID)".

## Flowchart 2: Model architecture (slide 7). Vertical stack, top to bottom

| # | Shape | Text | Output shape |
|---|---|---|---|
| 1 | Parallelogram | Input graph: x (116×5), z_ij | — |
| 2 | Rectangle | Linear 5→192 + SiLU + Dropout 0.25 | 116×192 |
| 3 | Rectangle ×3 (stacked with a "×3" badge) | GCNConv(\|z\|) → BN → SiLU → Dropout, **+ residual** | 116×192 |
| 4 | Rectangle | GATv2Conv, 4 heads × 48, edge_dim=1 (z_ij) → BN → SiLU, **+ residual** | 116×192 |
| 5 | Two side-by-side rectangles merging | Mean pool ‖ Max pool | 384 |
| 6 | Rectangle | SwiGLU 384→384→96 | 96 |
| 7 | Rectangle | Dropout → Linear 96→2 → softmax | 2 |
| 8 | Rounded rectangle (green) | GNNExplainer: edge mask M_e, feature mask M_f (θ frozen) | top-k edges |

Draw a curved arrow from the side of block 3 back to its own input to show the residual. Put block 8 to the right of 4–7 with a dotted arrow, since it runs after training.

## Flowchart 3: Explainer loop (optional, slide 10)

`Trained model f_θ (frozen)` → `Init masks M_e ∈ R^E, M_f ∈ R^5` → `Masked forward f_θ(A⊙σ(M_e), X⊙σ(M_f))` → `Loss = CE + λ₁‖σ(M)‖₁ + λ₂ Ent(σ(M))` → `∂L/∂M only` → back to the masked forward (500 iterations) → `Top-10 edges by σ(M_e)` → `Precision vs planted edges`.

## Tables for Excel (paste as-is)

**Table T1 (slide 9), threshold comparison**
```
Threshold rule,t,Test Accuracy,Test Macro-F1
Default 0.5,0.500,0.5733,0.520
Youden on validation,0.503,0.5733,0.520
Youden on test (leaky),0.176,0.7333,0.731
```

**Table T2 (slide 10), model comparison** (Excel bar chart: Category = Model, Value = AUC; add a reference line at 0.5)
```
Model,Test AUC,Test Accuracy,Test Macro-F1,Source
RBF-SVM (flattened),0.530,0.533,0.533,re-run
L2 LogReg (flattened),0.509,0.493,0.493,re-run
RF-GDV,0.692,0.560,0.534,README (unverified)
node2vec,0.635,0.520,0.498,README (unverified)
GATv1 mean-pool,0.713,0.573,0.550,README (unverified)
Ours,0.746,0.573,0.520,test_metrics.json
One scalar (oracle),1.000,1.000,1.000,re-run
```

**Table T3 (slide 8), training history every 10 epochs**
```
Epoch,Train loss,Val loss,Train F1,Val F1,Val AUC
1,3.397,0.693,0.460,0.336,0.571
11,0.694,0.695,0.568,0.529,0.563
21,0.677,0.610,0.613,0.658,0.791
31,0.682,0.721,0.648,0.540,0.720
41,0.671,0.620,0.651,0.655,0.763
51,0.645,0.596,0.648,0.702,0.788
61,0.634,0.671,0.654,0.719,0.818
70,0.668,0.670,0.634,0.765,0.822
```

**Table T4 (slide 10), explainer audit**
```
Metric,Value,Chance
Top-10 edges that are intra-lobe,15.3%,16.0%
Top-10 edges that are homotopic L-R,3 / 750,~5 / 750
Node-feature importance (all 75 subjects),0.0,—
Most frequent edge (subjects),3 / 75,—
```

---

# Appendix C: Re-running the † rows before the talk

The GATv1 row is the same model with three flags switched off. Copy `config/v100_config.yaml` to `config/v100_gatv1.yaml` and set:
```yaml
model:
  use_gatv2: false
  use_edge_attr: false
  use_dual_pooling: false
  use_swiglu: false
training:
  best_model_filename: best_model_gatv1.pt
```
Then:
```bash
python scripts/train_launcher.py --config config/v100_gatv1.yaml
```
Save `logs/test_metrics.json` under a new name after each run so it is not overwritten. RF-GDV and node2vec live in `models/baselines/`; evaluate them on the test PatientIDs from `subject_level_split(..., random_seed=42)`. Note that `node2vec_baseline.py` silently falls back to a spectral embedding when the `node2vec` package is missing, so log which path ran.
