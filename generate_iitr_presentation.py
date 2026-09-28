"""
generate_iitr_presentation.py
=============================
Generates the official IIT Roorkee presentation adhering strictly to:
1. Blackboard instructions:
   - Part 1: Problem Statement
   - Part 2: Baselines (3-4 papers)
   - Part 3: What you are going to do (Proposed Research & Methodology)
   - Presentation framed prospectively as the PhD research plan ("What we will do / proposed framework")
   - Time limit: 10 minutes (8 min presentation + 2 min Q&A)
   - Rigorous mathematical formulations on every slide (equations 1-9)
   - Prepares multi-modal extension (fMRI functional + DTI structural tractography)
2. Official IIT Roorkee Template:
   - IITR_template.pptx (Layouts: Title, Content, Comparison, Last Slide)

Author: Animesh Basak (IIT Roorkee, Department of CSE)
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Professional Academic Color Palette
COLOR_NAVY    = RGBColor(14, 52, 110)    # IITR Deep Navy
COLOR_DARK    = RGBColor(30, 41, 59)     # Slate 800 (High contrast text)
COLOR_MUTED   = RGBColor(100, 116, 139)  # Slate 500 (Subtitles/metadata)
COLOR_BLUE    = RGBColor(2, 132, 199)    # Sky/Accent blue
COLOR_GREEN   = RGBColor(16, 120, 72)    # Academic Forest Green
COLOR_RED     = RGBColor(185, 28, 28)    # Accent Red (Pathology)
COLOR_CARD_BG = RGBColor(248, 250, 252)  # Soft Slate Card Background
COLOR_BORDER  = RGBColor(226, 232, 240)  # Clean Card Border


def build_iitr_deck():
    template_path = "IITR_template.pptx"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Missing {template_path}")

    prs = Presentation(template_path)

    # Remove the 5 empty placeholder slides in the original template
    while len(prs.slides) > 0:
        rId = prs.slides._sldIdLst[0].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[0]

    layout_title = prs.slide_layouts[0]
    layout_content = prs.slide_layouts[1]
    layout_last = prs.slide_layouts[4]

    # Helper function for adding structured content cards
    def create_card(slide, left, top, width, height, title, items, border_color=COLOR_BORDER, title_color=COLOR_NAVY):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.margin_left = Inches(0.18)
        ctf.margin_right = Inches(0.18)
        ctf.margin_top = Inches(0.14)
        ctf.margin_bottom = Inches(0.14)

        tp = ctf.paragraphs[0]
        tp.text = title
        tp.font.size = Pt(11.5)
        tp.font.bold = True
        tp.font.color.rgb = title_color

        for it in items:
            p = ctf.add_paragraph()
            p.text = it
            p.font.size = Pt(9.5)
            p.font.color.rgb = COLOR_DARK
            p.space_before = Pt(2.5)
        return card

    # =========================================================================
    # SLIDE 1: Title Slide (Official IITR Format)
    # =========================================================================
    s1 = prs.slides.add_slide(layout_title)
    
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p0 = tf.paragraphs[0]
    p0.text = "Explainable Graph Neural Networks & Hybrid Graph Databases for Brain Biomarker Discovery"
    p0.font.size = Pt(20)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_NAVY
    p0.alignment = PP_ALIGN.CENTER

    p1 = tf.add_paragraph()
    p1.text = "Mathematical Formulation of Non-Euclidean Connectomes in Neurodegenerative Pathology"
    p1.font.size = Pt(12.5)
    p1.font.italic = True
    p1.font.color.rgb = COLOR_BLUE
    p1.alignment = PP_ALIGN.CENTER

    mb = s1.shapes.add_textbox(Inches(0.8), Inches(4.3), Inches(8.4), Inches(1.8))
    mtf = mb.text_frame
    mtf.word_wrap = True

    mp0 = mtf.paragraphs[0]
    mp0.text = "PhD Research Proposal  |  Candidate: Animesh Basak"
    mp0.font.size = Pt(14)
    mp0.font.bold = True
    mp0.font.color.rgb = COLOR_DARK
    mp0.alignment = PP_ALIGN.CENTER

    mp1 = mtf.add_paragraph()
    mp1.text = "Department of Computer Science & Engineering\nIndian Institute of Technology Roorkee (IIT Roorkee)"
    mp1.font.size = Pt(12)
    mp1.font.color.rgb = COLOR_MUTED
    mp1.alignment = PP_ALIGN.CENTER

    mp2 = mtf.add_paragraph()
    mp2.text = "\nStructure: 1. Problem Statement  |  2. Baselines (4 Papers)  |  3. Proposed Methodology & Mathematics"
    mp2.font.size = Pt(10)
    mp2.font.bold = True
    mp2.font.color.rgb = COLOR_NAVY
    mp2.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 2: Problem Statement 1 — Clinical Urgency & The Non-Euclidean Dilemma
    # =========================================================================
    s2 = prs.slides.add_slide(layout_content)
    s2.shapes.title.text = "1. Problem Statement: Clinical Motivation & Graph Paradigm"

    c1_items = [
        "• Clinical Urgency: >55M people worldwide suffer from neurodegenerative diseases (Alzheimer's, Parkinson's).",
        "• Diagnostic Bottleneck: Clinical symptoms appear only after irreversible structural neuron loss (>50%).",
        "• Pre-Symptomatic Precursor: Pathological disconnection begins years earlier in resting-state functional brain networks.",
        "• Research Goal: Formulate a computational framework to detect topological breakdown in fMRI connectomes before anatomical atrophy occurs."
    ]
    create_card(s2, Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.2), "A. Clinical Urgency: Functional Disconnection Precedes Structural Atrophy", c1_items)

    c2_items = [
        "1. Voxel-wise Mass Univariate: Tests ~100k voxels independently: y_v = beta * X_v + eps. Incurs massive multiple hypothesis testing penalties (Bonferroni / FDR) and completely ignores inter-regional relational topology.",
        "2. Euclidean Deep Learning Failure: Standard CNNs assume regular translational Euclidean grids. Brain networks are non-Euclidean graphs with no canonical node ordering: Pi * A * Pi^T = A for all permutation matrices Pi in S_N.",
        "3. Connectome Flattening: Reshaping correlation matrices into 1D vectors (N(N-1)/2 = 6,670 features) destroys spatial neighborhood geometry and leads to severe parameter explosion and overfitting on clinical cohorts."
    ]
    create_card(s2, Inches(0.6), Inches(3.45), Inches(8.8), Inches(3.4), "B. The Non-Euclidean Dilemma: Why Standard Deep Learning Fails", c2_items, border_color=COLOR_RED, title_color=COLOR_RED)

    # =========================================================================
    # SLIDE 3: Problem Statement 2 — Mathematical Graph-Level Formulation
    # =========================================================================
    s3 = prs.slides.add_slide(layout_content)
    s3.shapes.title.text = "1. Problem Statement: Mathematical Connectome Formulation"

    c1_items = [
        "• Anatomical Parcellation: Automated Anatomical Labeling (AAL-116) partitions brain into |V| = 116 anatomical ROIs.",
        "• BOLD Time Series: For region i in V, let x_i(t) in R^T represent blood-oxygen-level-dependent signal over T timepoints.",
        "• Mathematical Formulation 1 — Pearson Correlation Coefficient (Functional Edge Weights):",
        "     r_{ij} = ( sum_{t=1}^T (x_i(t) - mu_i)(x_j(t) - mu_j) ) / ( sqrt{sum_{t=1}^T (x_i(t)-mu_i)^2} * sqrt{sum_{t=1}^T (x_j(t)-mu_j)^2} ) in [-1, 1]",
        "• Sparsified Adjacency Matrix: To filter acquisition noise and physiological artifacts:",
        "     A_{ij} = |r_{ij}|  if |r_{ij}| >= tau (threshold tau = 0.3); else 0, with A_{ii} = 0",
        "• Node Feature Matrix: X in R^{|V| x F} with F=5 graph-theoretic topological descriptors (Degree, PageRank, Clustering, Betweenness, Eigenvector)."
    ]
    create_card(s3, Inches(0.6), Inches(1.1), Inches(8.8), Inches(3.0), "A. Mathematical Formulation of Brain Connectome G = (V, E, X)", c1_items)

    c2_items = [
        "• Mathematical Formulation 2 — Graph-Level Binary Classification Objective:",
        "     P(y_k = 1 | G_k) = sigma( f_theta(A_k, X_k) ),  where y_k in {0 (Healthy Control), 1 (Neurodegenerative)}",
        "     Objective: Classify the global network state of the whole brain graph rather than isolated individual voxels.",
        "• Mathematical Formulation 9 — Subgraph Explanation Objective:",
        "     Find compact explanatory subgraph M_k subseteq G_k that maximizes Mutual Information:",
        "     max_{M_k}  MI( y_k, G_k * M_k ) = H(y_k) - H(y_k | G_k * M_k)",
        "• Rigorous Constraint: Subject-level data partitioning (S_train intersect S_test = emptyset) to guarantee zero identity leakage."
    ]
    create_card(s3, Inches(0.6), Inches(4.2), Inches(8.8), Inches(2.7), "B. Graph-Level Classification & Explanation Objectives", c2_items)

    # =========================================================================
    # SLIDE 4: Baselines Paper 1 & 2 — Shallow & Network Motif Models
    # =========================================================================
    s4 = prs.slides.add_slide(layout_content)
    s4.shapes.title.text = "2. Baseline Models: Paper 1 & Paper 2"

    p1_items = [
        "• Reference: Abraham et al. (NeuroImage 2017) / Cuingnet et al. (NeuroImage 2011)",
        "• Methodology: Vectorizes upper triangle of functional connectivity matrix into a flat 1D vector v in R^{6670}.",
        "• Mathematical Formulation (Dual Soft-Margin SVM with RBF Kernel):",
        "     max_{alpha} sum_{k=1}^N alpha_k - 0.5 * sum_{k,l} alpha_k alpha_l y_k y_l K(v_k, v_l)",
        "     subject to:  0 <= alpha_k <= C,   sum_{k=1}^N alpha_k y_k = 0",
        "     where K(v_k, v_l) = exp( - gamma * ||v_k - v_l||_2^2 )",
        "• Critical Limitation: Treats all 6,670 pairwise edges as independent Euclidean dimensions. Completely loses network topology, clique hierarchy, and hub clustering."
    ]
    create_card(s4, Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.75), "Paper 1: Kernel Support Vector Machines on Vectorized Connectomes", p1_items)

    p2_items = [
        "• Reference: Pržulj et al. (Bioinformatics 2007) / Yaveroğlu et al. (Nature Scientific Reports 2014)",
        "• Methodology: Counts non-isomorphic small induced subgraphs (graphlets on 2-5 nodes, orbits 0-72).",
        "• Mathematical Formulation:",
        "     For node u in V, Graphlet Degree Vector: GDV(u) = [ u_0, u_1, ..., u_d ]^T,  u_j = count of orbit j instances.",
        "     Graph-level summary features: Phi(G) = [ mean(GDV), std(GDV), max(GDV) ] in R^{15}.",
        "     Random Forest Ensemble: f_RF(Phi) = (1/B) sum_{b=1}^B T_b(Phi).",
        "• Critical Limitation: Relies entirely on static, handcrafted graph topological counts. Lacks end-to-end gradient backpropagation and cannot learn dynamic edge importance."
    ]
    create_card(s4, Inches(0.6), Inches(3.95), Inches(8.8), Inches(2.9), "Paper 2: Random Forest with Graphlet Degree Vectors (RF-GDV)", p2_items)

    # =========================================================================
    # SLIDE 5: Baselines Paper 3 & 4 — Random Walks & Standard GAT
    # =========================================================================
    s5 = prs.slides.add_slide(layout_content)
    s5.shapes.title.text = "2. Baseline Models: Paper 3 & Paper 4"

    p3_items = [
        "• Reference: Grover & Leskovec (KDD 2016) / Connectome Applications (Mahmood et al., 2021)",
        "• Methodology: Learns continuous node embeddings via 2nd-order biased random walks.",
        "• Mathematical Formulation (Transition Probability):",
        "     P(c_i = x | c_{i-1} = v) = ( pi_{vx} ) / Z,   where  pi_{vx} = alpha_{pq}(t, x) * w_{vx}",
        "     Bias factor: alpha_{pq}(t, x) = 1/p if d_{tx}=0;  1 if d_{tx}=1;  1/q if d_{tx}=2",
        "     Objective: max_f sum_{u in V} log P( N_S(u) | f(u) ) using Skip-Gram with Negative Sampling.",
        "• Critical Limitation: Transductive random walk embeddings are trained unsupervised without disease label guidance; subsequent mean-pooling loses graph structural discriminability."
    ]
    create_card(s5, Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.75), "Paper 3: node2vec Biased Random Walk Connectome Embeddings", p3_items)

    p4_items = [
        "• Reference: Veličković et al. (ICLR 2018) / BrainGNN (Li et al., Medical Image Analysis 2021)",
        "• Methodology: Self-attention over graph neighbors to weight regional communicative connections.",
        "• Mathematical Formulation 4 — Standard GATv1 Attention:",
        "     alpha_{ij} = ( exp( LeakyReLU( a^T [ W h_i || W h_j ] ) ) ) / ( sum_{k in N_i} exp( LeakyReLU( a^T [ W h_i || W h_k ] ) ) )",
        "• Fatal Mathematical Flaw (Static Attention / Rank Collapse, Brody et al. 2021):",
        "     Since a^T [W h_i || W h_j] = a_l^T W h_i + a_r^T W h_j, the relative score for neighbor j is strictly monotonic in a_r^T W h_j.",
        "     Thus, the ranking of neighbors is global and identical for all query nodes i, preventing patient-specific adaptive attention."
    ]
    create_card(s5, Inches(0.6), Inches(3.95), Inches(8.8), Inches(2.9), "Paper 4: Standard Graph Attention Networks (GATv1)", p4_items, border_color=COLOR_RED, title_color=COLOR_RED)

    # =========================================================================
    # SLIDE 6: Comparative Analysis of Baseline Limitations
    # =========================================================================
    s6 = prs.slides.add_slide(layout_content)
    s6.shapes.title.text = "2. Baseline Models: Comparative Summary of Limitations"

    rows, cols = 5, 5
    tbl_shape = s6.shapes.add_table(rows, cols, Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.5))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.5)
    tbl.columns[1].width = Inches(1.8)
    tbl.columns[2].width = Inches(1.9)
    tbl.columns[3].width = Inches(1.6)
    tbl.columns[4].width = Inches(2.0)

    headers = ["Baseline Model", "Mathematical Core", "Graph Awareness", "Explainability", "Fatal Clinical Defect"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)

    data = [
        ["RBF-SVM (Abraham '17)", "max alpha - 0.5 sum alpha_i alpha_j y_i y_j K", "Zero (1D flat vector)", "None (Black-Box)", "Destroys graph topology"],
        ["RF-GDV (Pržulj '07)", "sum T_b( OrbitCounts(G) )", "Scalar Motifs only", "Gini feature rank", "Static, unlearnable features"],
        ["node2vec (Grover '16)", "Skip-Gram over biased walks", "Local random walk", "None (Black-Box)", "Unsupervised; misses disease signal"],
        ["GATv1 (Li / Veličković)", "Softmax( a^T [Wh_i || Wh_j] )", "Static Attention", "Attention weights", "Attention rank collapse (Brody '21)"]
    ]
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = tbl.cell(i+1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else COLOR_CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(8.5)
            p.font.color.rgb = COLOR_DARK

    sum_items = [
        "1. No Structural Hierarchy: Baselines treat brain as flat vectors or static subgraph counts, ignoring dynamic communication.",
        "2. Static Attention Collapse: GATv1 fails to adapt attention ranking per patient due to linear projection coupling.",
        "3. Multi-Site Brittleness: None of the existing baseline formulations calibrate for scanner SNR domain shift across hospitals.",
        "4. Research Void to Address: Develop an end-to-end framework combining dynamic attention (GATv2), continuous edge attributes (Pearson r), non-linear noise gating (SwiGLU), and mathematically verifiable subgraph explainability."
    ]
    create_card(s6, Inches(0.6), Inches(3.8), Inches(8.8), Inches(3.05), "Summary of Research Gaps in Baseline Literature", sum_items, border_color=COLOR_NAVY, title_color=COLOR_NAVY)

    # =========================================================================
    # SLIDE 7: What We Will Do 1 — Architecture & Edge-Aware Message Passing
    # =========================================================================
    s7 = prs.slides.add_slide(layout_content)
    s7.shapes.title.text = "3. Proposed Research: Edge-Aware Dynamic Graph Convolution"

    gcn_items = [
        "• Mathematical Formulation 3 — Edge-Weight Aware Graph Convolution (Neighborhood Aggregation):",
        "     h_i^{(l+1)} = SiLU ( W_GCN^{(l)} * sum_{j in N_i cup {i}} ( e_{ij} / sqrt{d_i * d_j} ) * h_j^{(l)} ) + h_i^{(l)}   [Residual Connection]",
        "• Mathematical Formulation: Injects continuous functional edge weights e_{ij} = |r_{ij}| rather than binary adjacency, preserving continuous functional coupling intensities."
    ]
    create_card(s7, Inches(0.6), Inches(1.1), Inches(8.8), Inches(1.85), "Pillar 1: Edge-Weight Aware GCN Structural Aggregation", gcn_items)

    gat_items = [
        "• Resolving Brody et al. (2021) Static Attention Rank Collapse:",
        "     Standard GAT (Static):  alpha_{ij} ~ a^T [Wh_i || Wh_j]   (Fixed neighbor ranking across all queries i)",
        "• Mathematical Formulation 5 — GATv2 Dynamic Multi-Head Attention with Continuous Edge Attributes:",
        "     alpha_{ij} = Softmax_j ( a^T * LeakyReLU( W_l h_i + W_r h_j + W_e e_{ij} ) )",
        "     h_i^{(l+1)} = SiLU ( ||_{k=1}^H sum_{j in N_i} alpha_{ij}^k * W_v^k h_j^{(l)} ) + h_i^{(l)}",
        "• Mathematical Formulation: Placing non-linearity BEFORE the projection vector enables strictly dynamic, query-dependent attention over disease-salient functional connections."
    ]
    create_card(s7, Inches(0.6), Inches(3.05), Inches(8.8), Inches(2.2), "Pillar 2: GATv2 Dynamic Multi-Head Attention with Continuous Edge Attributes", gat_items)

    pool_items = [
        "• Mathematical Formulation 6 — Dual Global Graph-Level Readout Concatenation:",
        "     h_mean = (1 / |V|) * sum_{v in V} h_v   [Whole-brain global connectivity background in R^d]",
        "     h_max  = max_{v in V} ( h_v )            [Focal, localized network lesion hubs in R^d]",
        "     h_G    = [ h_mean || h_max ] in R^{2d}   [Jointly preserves whole-brain and focal biomarkers]"
    ]
    create_card(s7, Inches(0.6), Inches(5.35), Inches(8.8), Inches(1.5), "Pillar 3: Dual Global Graph Pooling [Mean || Max]", pool_items)

    # =========================================================================
    # SLIDE 8: What We Will Do 2 — SwiGLU Gating & Youden Calibration
    # =========================================================================
    s8 = prs.slides.add_slide(layout_content)
    s8.shapes.title.text = "3. Proposed Research: SwiGLU Noise Gating & Calibration"

    swiglu_items = [
        "• Mathematical Formulation 7 — Swish Gated Linear Unit (Shazeer, 2020; LLaMA 3 Architecture):",
        "     SwiGLU(h_G) = ( Swish(h_G * W_gate + b_gate) (x) (h_G * W_up + b_up) ) * W_down + b_down",
        "     where Swish(z) = z * sigma(beta * z) = z / (1 + e^{-beta * z})",
        "• Classification Logits:  z = W_cls * Dropout( SwiGLU(h_G) ) in R^2,   P(y=1|G) = ( e^{z_1} ) / ( e^{z_0} + e^{z_1} )",
        "• Mathematical Formulation: Multiplicative gating branch (x) acts as a dynamic noise filter, suppressing multi-site scanner noise fluctuations while amplifying subtle functional disconnection signals."
    ]
    create_card(s8, Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.4), "A. SwiGLU Multiplicative Noise Gating in Classifier Head", swiglu_items)

    cal_items = [
        "• Multi-Site Domain Shift Challenge: Scanner SNR variations shift predicted test probabilities, making the default arbitrary p = 0.5 decision boundary sub-optimal.",
        "• Mathematical Formulation 8 — Youden's J-Statistic (Youden, 1950):",
        "     J(t) = Sensitivity(t) + Specificity(t) - 1 = TruePositiveRate(t) - FalsePositiveRate(t)",
        "• Optimal Threshold Selection (Calibrated on Validation ROC Curve):",
        "     t^* = argmax_{t in [0, 1]}  ( TPR(t) - FPR(t) )",
        "• Operational Inference Rule:",
        "     y_pred = 1  if P(y=1 | G) >= t^*;  else 0",
        "• Formulated Effect: Realigns operational test accuracy directly with the model's true discriminative ROC-AUC, eliminating multi-site threshold bias."
    ]
    create_card(s8, Inches(0.6), Inches(3.6), Inches(8.8), Inches(3.25), "B. Decision Threshold Calibration via Youden's J-Statistic", cal_items, border_color=COLOR_GREEN, title_color=COLOR_GREEN)

    # =========================================================================
    # SLIDE 9: What We Will Do 3 — Mathematical Explainability (GNNExplainer)
    # =========================================================================
    s9 = prs.slides.add_slide(layout_content)
    s9.shapes.title.text = "3. Proposed Research: GNNExplainer Mathematical Framework"

    exp_items = [
        "• Goal: Identify compact explanatory subgraph G_s = (V_s, E_s) that explains disease prediction y_hat.",
        "• Information-Theoretic Mutual Information Objective (Ying et al., NeurIPS 2019):",
        "     max_{G_s}  MI( Y, G_s ) = H(Y) - H( Y | G = G_s )",
        "• Mathematical Formulation 9 — Continuous Relaxation with Entropy Regularization:",
        "     min_{M, F}  L( y_hat, f( A (*) sigma(M), X (*) sigma(F) ) ) + lambda_1 * ||M||_1 + lambda_2 * Ent(M)",
        "     where Ent(M) = - sum_{e} ( M_e * log M_e + (1 - M_e) * log(1 - M_e) ) enforces discrete binary-like mask selections.",
        "• Gradient Ascent Optimization: Model weights theta are frozen; gradients backpropagate strictly to masks M and F:",
        "     M <- M - eta * nabla_M L_explainer   (Optimized over 500 gradient steps per patient on CUDA)."
    ]
    create_card(s9, Inches(0.6), Inches(1.1), Inches(8.8), Inches(2.95), "A. Information-Theoretic Biomarker Extraction Objective", exp_items)

    bio_items = [
        "• Planned Consensus Neural Biomarker Targets:",
        "   1. Cerebellar Vermis Hubs: Vermis_6 to Vermis_10 (Detecting Cerebellar Cognitive Affective Syndrome).",
        "   2. Cortico-Striatal Disruption: Lingual_L <-> Putamen_L (Visual-motor integration breakdown).",
        "   3. Default Mode Network (DMN) Decoupling: Cerebelum_9_R <-> Cingulum_Post_L (Posterior Cingulate hub disruption, clinical hallmark of Alzheimer's).",
        "   4. Memory-Attention Circuit: Hippocampus_L <-> SupraMarginal_R (Episodic memory to parietal attention disconnection)."
    ]
    create_card(s9, Inches(0.6), Inches(4.15), Inches(8.8), Inches(2.7), "B. Biological Hypotheses for Discovered Neural Biomarkers", bio_items)

    # =========================================================================
    # SLIDE 10: What We Will Do 4 — Proposed Multi-Modal Extension (fMRI + DTI)
    # =========================================================================
    s10 = prs.slides.add_slide(layout_content)
    s10.shapes.title.text = "3. Proposed Research: Multi-Modal Graph Fusion (fMRI + DTI)"

    mod_items = [
        "• Motivation: Functional disconnection (fMRI) is physically constrained by white-matter axonal structural pathways (DTI). Combining functional and structural connectomes resolves false-positive correlations.",
        "• Mathematical Dual-Graph Formulation: Each subject k is modeled as a dual-relational graph G_k = (V, E_{func}, E_{struct}, X):",
        "     Functional Graph:  A_{func} in R^{|V| x |V|},   A_{func}(i, j) = |r_{ij}|  (Pearson correlation from resting-state fMRI)",
        "     Structural Graph:  A_{struct} in R^{|V| x |V|},  A_{struct}(i, j) = FA_{ij} * log(N_{ij} + 1)  (Fractional Anisotropy & Streamline count from DTI tractography)",
        "• Cross-Modality Attentional Fusion Operator:",
        "     h_i^{func}   = GATv2_{func}( h_i, A_{func}, E_{func} )",
        "     h_i^{struct} = GATv2_{struct}( h_i, A_{struct}, E_{struct} )",
        "     h_i^{fused}  = gamma_i * h_i^{func} + (1 - gamma_i) * h_i^{struct},   where  gamma_i = sigma( W_gamma [ h_i^{func} || h_i^{struct} ] )",
        "• Unified Graph Readout:  h_G = [ MeanPool(h^{fused}) || MaxPool(h^{fused}) ] -> SwiGLU -> Binary Classification P(y=1 | G)."
    ]
    create_card(s10, Inches(0.6), Inches(1.1), Inches(8.8), Inches(3.2), "A. Mathematical Framework for Multi-Modal Functional-Structural Graph Fusion", mod_items)

    pipe_items = [
        "• Target Datasets for Validation: ADNI (Alzheimer's Disease Neuroimaging Initiative) & PPMI (Parkinson's Progression Markers Initiative).",
        "• Rigorous Validation Protocol: 5-fold Subject-Stratified Cross-Validation across distinct clinical scanner centers.",
        "• Evaluation Criteria: Macro-F1 (primary clinical metric), ROC-AUC, Calibrated Accuracy (Youden's J), and Biomarker Subgraph Jaccard Stability."
    ]
    create_card(s10, Inches(0.6), Inches(4.45), Inches(8.8), Inches(2.4), "B. Validation Strategy & Benchmark Roadmap", pipe_items, border_color=COLOR_GREEN, title_color=COLOR_GREEN)

    # =========================================================================
    # SLIDE 11: Summary & Roadmap
    # =========================================================================
    s11 = prs.slides.add_slide(layout_content)
    s11.shapes.title.text = "Summary of Proposed Contributions & Technical Timeline"

    c_left_items = [
        "1. Relational Connectome Modeling: Formulate brain graphs natively in a persistent graph database (Neo4j) with topological centrality feature descriptors.",
        "2. Dynamic Neural Architecture: Eliminate GATv1 static rank collapse via GATv2 dynamic attention with continuous Pearson edge attributes.",
        "3. Scanner Invariant Representation: Introduce SwiGLU noise gating and Youden's J calibration to defeat hospital scanner domain shift.",
        "4. Verifiable Clinical Biomarkers: Information-theoretic subgraph extraction via GNNExplainer identifying consensus neural disease circuits."
    ]
    create_card(s11, Inches(0.6), Inches(1.1), Inches(4.3), Inches(5.7), "Proposed Core Contributions", c_left_items)

    c_right_items = [
        "• Phase 1 (Architecture Formulation): Implement Edge-weight aware GCN + GATv2 + SwiGLU gating in PyTorch Geometric.",
        "• Phase 2 (Explainability Pipeline): Develop GNNExplainer module for automated consensus circuit extraction.",
        "• Phase 3 (Multi-Site Calibration): Integrate Youden's J-statistic optimization for multi-center clinical deployments.",
        "• Phase 4 (Multi-Modal Extension): Fuse functional fMRI connectomes with structural DTI tractography on ADNI/PPMI datasets.",
        "• Phase 5 (Clinical Deployment): Package pipeline into a lightweight FastAPI inference service with drift monitoring."
    ]
    create_card(s11, Inches(5.1), Inches(1.1), Inches(4.3), Inches(5.7), "Research Execution Timeline", c_right_items, border_color=COLOR_BLUE, title_color=COLOR_BLUE)

    # =========================================================================
    # SLIDE 12: Last Slide (Official IITR Format)
    # =========================================================================
    s12 = prs.slides.add_slide(layout_last)
    try:
        s12.shapes.title.text = "Thank You\nQuestions & Answers"
    except Exception:
        pass

    q_box = s12.shapes.add_textbox(Inches(1.5), Inches(4.2), Inches(7.0), Inches(1.5))
    qtf = q_box.text_frame
    qp0 = qtf.paragraphs[0]
    qp0.text = "Time Allocation: 8 Minutes Presentation + 2 Minutes Q&A (Total: 10 Minutes)"
    qp0.font.size = Pt(12)
    qp0.font.bold = True
    qp0.font.color.rgb = COLOR_NAVY
    qp0.alignment = PP_ALIGN.CENTER

    qp1 = qtf.add_paragraph()
    qp1.text = "Animesh Basak  |  IIT Roorkee  |  Department of Computer Science & Engineering"
    qp1.font.size = Pt(11)
    qp1.font.color.rgb = COLOR_MUTED
    qp1.alignment = PP_ALIGN.CENTER

    out_file = "IITR_Neuro_Neo4j_Presentation.pptx"
    prs.save(out_file)
    print(f"\n[SUCCESS] Generated official IIT Roorkee presentation: {os.path.abspath(out_file)}")


if __name__ == "__main__":
    build_iitr_deck()
