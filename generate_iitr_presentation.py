"""
generate_iitr_presentation.py
=============================
Generates the official IIT Roorkee presentation adhering strictly to:
1. Blackboard instructions:
   - Part 1: Problem Statement
   - Part 2: Baselines (3-4 papers)
   - Part 3: What you are going to do (Proposed Research & Methodology)
   - Prospective PhD research plan ("What we will do / proposed framework")
   - Time limit: 10 minutes (8 min presentation + 2 min Q&A)
   - Rigorous mathematical formulations rendered directly as authentic LaTeX equations (assets/math/*.png)
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
    def create_card(slide, left, top, width, height, title, items, border_color=COLOR_BORDER, title_color=COLOR_NAVY, title_size=11.5, item_size=9.2):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        ctf = card.text_frame
        ctf.word_wrap = True
        ctf.margin_left = Inches(0.18)
        ctf.margin_right = Inches(0.18)
        ctf.margin_top = Inches(0.12)
        ctf.margin_bottom = Inches(0.12)

        tp = ctf.paragraphs[0]
        tp.text = title
        tp.font.size = Pt(title_size)
        tp.font.bold = True
        tp.font.color.rgb = title_color

        for it in items:
            p = ctf.add_paragraph()
            p.text = it
            p.font.size = Pt(item_size)
            p.font.color.rgb = COLOR_DARK
            p.space_before = Pt(2.0)
        return card

    # Helper function to add a LaTeX equation image with an optional caption
    def add_math_image(slide, img_path, left, top, width, height=None):
        if not os.path.exists(img_path):
            print(f"[WARN] Equation image not found: {img_path}")
            return None
        if height is not None:
            pic = slide.shapes.add_picture(img_path, left, top, width=width, height=height)
        else:
            pic = slide.shapes.add_picture(img_path, left, top, width=width)
        return pic

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
        "• Node Feature Matrix: X in R^{|V| x 5} with graph-theoretic descriptors (Degree, PageRank, Clustering, Betweenness, Eigenvector)."
    ]
    create_card(s3, Inches(0.6), Inches(1.05), Inches(8.8), Inches(2.85), "A. Mathematical Formulation 1: Brain Connectome Graph G = (V, E, X)", c1_items)
    add_math_image(s3, "assets/math/eq1_pearson.png", Inches(1.0), Inches(2.1), Inches(4.5))
    add_math_image(s3, "assets/math/eq1_sparsification.png", Inches(1.0), Inches(2.9), Inches(5.2))

    c2_items = [
        "• Graph-Level Binary Classification Objective: Classify the whole-brain network state rather than isolated voxels:",
        "",
        "• Subgraph Explanation Objective: Find explanatory subgraph M_k subseteq G_k maximizing Mutual Information:",
        "     max_{M_k}  MI( y_k, G_k * M_k ) = H(y_k) - H(y_k | G_k * M_k)",
        "• Patient-Stratified Cross-Validation: S_train intersect S_test = emptyset (guarantees zero subject identity leakage)."
    ]
    create_card(s3, Inches(0.6), Inches(4.05), Inches(8.8), Inches(2.8), "B. Mathematical Formulation 2: Graph Classification & Explanation Objectives", c2_items)
    add_math_image(s3, "assets/math/eq2_classification.png", Inches(1.0), Inches(4.7), Inches(4.6))

    # =========================================================================
    # SLIDE 4: Baselines Paper 1 & 2 — Shallow & Network Motif Models
    # =========================================================================
    s4 = prs.slides.add_slide(layout_content)
    s4.shapes.title.text = "2. Baseline Models: Paper 1 & Paper 2"

    p1_items = [
        "• Reference: Abraham et al. (NeuroImage 2017) / Cuingnet et al. (NeuroImage 2011)",
        "• Vectorizes upper triangle of functional connectivity matrix into a flat 1D vector v in R^{6670}:",
        "",
        "• Critical Limitation: Treats all 6,670 pairwise edges as independent Euclidean dimensions. Completely loses network topology, clique hierarchy, and hub clustering."
    ]
    create_card(s4, Inches(0.6), Inches(1.05), Inches(8.8), Inches(2.85), "Paper 1: Kernel Support Vector Machines on Vectorized Connectomes", p1_items)
    add_math_image(s4, "assets/math/base_svm.png", Inches(0.9), Inches(1.95), Inches(6.0))

    p2_items = [
        "• Reference: Pržulj et al. (Bioinformatics 2007) / Yaveroğlu et al. (Nature Scientific Reports 2014)",
        "• Counts non-isomorphic small induced subgraphs (graphlets on 2-5 nodes, orbits 0-72):",
        "",
        "• Critical Limitation: Relies entirely on static, handcrafted graph topological counts. Lacks end-to-end gradient backpropagation and cannot learn dynamic edge importance."
    ]
    create_card(s4, Inches(0.6), Inches(4.05), Inches(8.8), Inches(2.85), "Paper 2: Random Forest with Graphlet Degree Vectors (RF-GDV)", p2_items)
    add_math_image(s4, "assets/math/base_rf_gdv.png", Inches(0.9), Inches(4.95), Inches(6.0))

    # =========================================================================
    # SLIDE 5: Baselines Paper 3 & 4 — Random Walks & Standard GAT
    # =========================================================================
    s5 = prs.slides.add_slide(layout_content)
    s5.shapes.title.text = "2. Baseline Models: Paper 3 & Paper 4"

    p3_items = [
        "• Reference: Grover & Leskovec (KDD 2016) / Connectome Applications (Mahmood et al., 2021)",
        "• Learns continuous node embeddings via 2nd-order biased random walks:",
        "",
        "• Critical Limitation: Transductive random walk embeddings are trained unsupervised without disease label guidance; subsequent mean-pooling loses graph structural discriminability."
    ]
    create_card(s5, Inches(0.6), Inches(1.05), Inches(8.8), Inches(2.85), "Paper 3: node2vec Biased Random Walk Connectome Embeddings", p3_items)
    add_math_image(s5, "assets/math/base_node2vec.png", Inches(0.9), Inches(1.95), Inches(5.8))

    p4_items = [
        "• Reference: Veličković et al. (ICLR 2018) / BrainGNN (Li et al., Medical Image Analysis 2021)",
        "• Self-attention over graph neighbors to weight regional communicative connections:",
        "",
        "• Fatal Mathematical Flaw (Static Attention / Rank Collapse, Brody et al. 2021):",
        "",
        "• Consequence: The neighbor ranking is global and identical for all query nodes i, preventing patient-specific adaptive attention."
    ]
    create_card(s5, Inches(0.6), Inches(4.05), Inches(8.8), Inches(2.85), "Paper 4: Standard Graph Attention Networks (GATv1)", p4_items, border_color=COLOR_RED, title_color=COLOR_RED)
    add_math_image(s5, "assets/math/eq4_gatv1.png", Inches(0.9), Inches(4.7), Inches(5.0))
    add_math_image(s5, "assets/math/base_gatv1_defect.png", Inches(0.9), Inches(5.6), Inches(5.8))

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
        "",
        "• Formulation: Injects continuous functional edge weights e_{ij} = |r_{ij}| rather than binary adjacency, preserving continuous functional coupling intensities with residual connections."
    ]
    create_card(s7, Inches(0.6), Inches(1.05), Inches(8.8), Inches(1.9), "Pillar 1: Edge-Weight Aware GCN Structural Aggregation", gcn_items)
    add_math_image(s7, "assets/math/eq3_edge_gcn.png", Inches(1.0), Inches(1.48), Inches(5.2))

    gat_items = [
        "• Mathematical Formulation 5 — GATv2 Dynamic Multi-Head Attention with Continuous Edge Attributes:",
        "",
        "• Multi-Head Aggregation with Residual Stream:",
        "",
        "• Mathematical Formulation: Placing non-linearity BEFORE projection eliminates static attention collapse, enabling strictly dynamic, query-dependent attention over disease-salient functional connections."
    ]
    create_card(s7, Inches(0.6), Inches(3.05), Inches(8.8), Inches(2.6), "Pillar 2: GATv2 Dynamic Multi-Head Attention with Continuous Edge Attributes", gat_items)
    add_math_image(s7, "assets/math/eq5_gatv2.png", Inches(1.0), Inches(3.48), Inches(5.2))
    add_math_image(s7, "assets/math/eq5_gatv2_update.png", Inches(1.0), Inches(4.3), Inches(5.0))

    pool_items = [
        "• Mathematical Formulation 6 — Dual Global Graph-Level Readout Concatenation:",
        "",
        "• Mathematical Formulation: Preserves both whole-brain background connectivity (Mean) and focal lesion hubs (Max)."
    ]
    create_card(s7, Inches(0.6), Inches(5.75), Inches(8.8), Inches(1.25), "Pillar 3: Dual Global Graph Pooling [Mean || Max]", pool_items)
    add_math_image(s7, "assets/math/eq6_dual_pool.png", Inches(1.0), Inches(6.15), Inches(5.4))

    # =========================================================================
    # SLIDE 8: What We Will Do 2 — SwiGLU Gating & Youden Calibration
    # =========================================================================
    s8 = prs.slides.add_slide(layout_content)
    s8.shapes.title.text = "3. Proposed Research: SwiGLU Noise Gating & Calibration"

    swiglu_items = [
        "• Mathematical Formulation 7 — Swish Gated Linear Unit (Shazeer, 2020; LLaMA 3 Architecture):",
        "",
        "• Classification Logits:  z = W_cls * Dropout( SwiGLU(h_G) ) in R^2,   P(y=1|G) = ( e^{z_1} ) / ( e^{z_0} + e^{z_1} )",
        "• Mathematical Formulation: Multiplicative gating branch (x) acts as a dynamic noise filter, suppressing multi-site scanner noise fluctuations while amplifying subtle functional disconnection signals."
    ]
    create_card(s8, Inches(0.6), Inches(1.05), Inches(8.8), Inches(2.6), "A. SwiGLU Multiplicative Noise Gating in Classifier Head", swiglu_items)
    add_math_image(s8, "assets/math/eq7_swiglu.png", Inches(1.0), Inches(1.65), Inches(6.0))

    cal_items = [
        "• Multi-Site Domain Shift Challenge: Scanner SNR variations shift predicted test probabilities, making the default arbitrary p = 0.5 decision boundary sub-optimal.",
        "• Mathematical Formulation 8 — Youden's J-Statistic Optimization (Youden, 1950):",
        "",
        "• Optimal Calibration Rule on Validation ROC Curve:",
        "",
        "• Formulated Effect: Realigns operational test accuracy directly with the model's true discriminative ROC-AUC, eliminating multi-site threshold bias."
    ]
    create_card(s8, Inches(0.6), Inches(3.8), Inches(8.8), Inches(3.05), "B. Decision Threshold Calibration via Youden's J-Statistic", cal_items, border_color=COLOR_GREEN, title_color=COLOR_GREEN)
    add_math_image(s8, "assets/math/eq8_youden.png", Inches(1.0), Inches(4.7), Inches(5.2))
    add_math_image(s8, "assets/math/eq8_youden_opt.png", Inches(1.0), Inches(5.45), Inches(5.2))

    # =========================================================================
    # SLIDE 9: What We Will Do 3 — Mathematical Explainability (GNNExplainer)
    # =========================================================================
    s9 = prs.slides.add_slide(layout_content)
    s9.shapes.title.text = "3. Proposed Research: GNNExplainer Mathematical Framework"

    exp_items = [
        "• Goal: Identify compact explanatory subgraph G_s = (V_s, E_s) that explains disease prediction y_hat.",
        "• Information-Theoretic Mutual Information Objective (Ying et al., NeurIPS 2019):  max_{G_s} MI( Y, G_s ) = H(Y) - H( Y | G = G_s )",
        "• Mathematical Formulation 9 — Continuous Relaxation with Entropy Regularization:",
        "",
        "• Entropy Term Enforces Crisp Binary-like Edge Masks:",
        "",
        "• Gradient Optimization: Model weights theta are frozen; gradients backpropagate strictly to masks: M <- M - eta * nabla_M L_explainer."
    ]
    create_card(s9, Inches(0.6), Inches(1.05), Inches(8.8), Inches(3.25), "A. Information-Theoretic Biomarker Extraction Objective", exp_items)
    add_math_image(s9, "assets/math/eq9_gnnexplainer.png", Inches(1.0), Inches(2.2), Inches(6.0))
    add_math_image(s9, "assets/math/eq9_entropy.png", Inches(1.0), Inches(2.95), Inches(5.6))

    bio_items = [
        "• Planned Consensus Neural Biomarker Targets:",
        "   1. Cerebellar Vermis Hubs: Vermis_6 to Vermis_10 (Detecting Cerebellar Cognitive Affective Syndrome).",
        "   2. Cortico-Striatal Disruption: Lingual_L <-> Putamen_L (Visual-motor integration breakdown).",
        "   3. Default Mode Network (DMN) Decoupling: Cerebelum_9_R <-> Cingulum_Post_L (Posterior Cingulate hub disruption, clinical hallmark of Alzheimer's).",
        "   4. Memory-Attention Circuit: Hippocampus_L <-> SupraMarginal_R (Episodic memory to parietal attention disconnection)."
    ]
    create_card(s9, Inches(0.6), Inches(4.45), Inches(8.8), Inches(2.45), "B. Biological Hypotheses for Discovered Neural Biomarkers", bio_items)

    # =========================================================================
    # SLIDE 10: What We Will Do 4 — Proposed Multi-Modal Extension (fMRI + DTI)
    # =========================================================================
    s10 = prs.slides.add_slide(layout_content)
    s10.shapes.title.text = "3. Proposed Research: Multi-Modal Graph Fusion (fMRI + DTI)"

    mod_items = [
        "• Motivation: Functional disconnection (fMRI) is physically constrained by white-matter axonal structural pathways (DTI). Combining functional and structural connectomes resolves false-positive correlations.",
        "• Mathematical Dual-Graph Formulation: Each subject k is modeled as a dual-relational graph G_k = (V, E_{func}, E_{struct}, X):",
        "     Functional Graph:  A_{func} in R^{|V| x |V|},   A_{func}(i, j) = |r_{ij}|  (Pearson correlation from resting-state fMRI)",
        "     Structural Graph:  A_{struct} in R^{|V| x |V|},  A_{struct}(i, j) = FA_{ij} * log(N_{ij} + 1)  (Fractional Anisotropy & Streamline count from DTI)",
        "• Proposed Multi-Modal Gating Operator (Functional-Structural Fusion):",
        "",
        "• Unified Graph Readout:  h_G = [ MeanPool(h^{fused}) || MaxPool(h^{fused}) ] -> SwiGLU -> Binary Classification P(y=1 | G)."
    ]
    create_card(s10, Inches(0.6), Inches(1.05), Inches(8.8), Inches(3.45), "A. Mathematical Framework for Multi-Modal Functional-Structural Graph Fusion", mod_items)
    add_math_image(s10, "assets/math/ext_multimodal.png", Inches(1.0), Inches(2.9), Inches(6.0))

    pipe_items = [
        "• Target Datasets for Validation: ADNI (Alzheimer's Disease Neuroimaging Initiative) & PPMI (Parkinson's Progression Markers Initiative).",
        "• Rigorous Validation Protocol: 5-fold Subject-Stratified Cross-Validation across distinct clinical scanner centers.",
        "• Evaluation Criteria: Macro-F1 (primary clinical metric), ROC-AUC, Calibrated Accuracy (Youden's J), and Biomarker Subgraph Jaccard Stability."
    ]
    create_card(s10, Inches(0.6), Inches(4.65), Inches(8.8), Inches(2.25), "B. Validation Strategy & Benchmark Roadmap", pipe_items, border_color=COLOR_GREEN, title_color=COLOR_GREEN)

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
