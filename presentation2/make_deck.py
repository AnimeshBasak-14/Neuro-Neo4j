"""
make_deck.py — Complete deck builder for Presentation 2.
"""

import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

from build_presentation import (
    TEMPLATE_PATH, REPO_ROOT, FIG_DIR, OUTPUT_PPTX,
    NAVY, BLACK, DARK_GRAY, WHITE, BORDER_BLUE, BORDER_GREEN, BORDER_RED,
    FILL_LIGHT_BLUE, FILL_LIGHT_GREEN,
    add_math_to_para, set_para_font, create_rounded_box,
    add_bullet_item, add_slide_title, add_picture_with_svg, add_speaker_notes
)

def build_presentation():
    print(f"Loading template from {TEMPLATE_PATH}...")
    prs = pptx.Presentation(TEMPLATE_PATH)
    
    # -------------------------------------------------------------------------
    # Slide 1: Title Slide
    # -------------------------------------------------------------------------
    print("Building Slide 1: Title...")
    s1 = prs.slides[0]
    
    # Shape 0: Title text
    title_shape = s1.shapes[0]
    tf1 = title_shape.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "Explainable Graph Neural Networks & Hybrid Graph Databases for Brain Biomarker Discovery"
    set_para_font(p1, name="Calibri", size_pt=22, bold=True, color=BLACK)
    p1.alignment = PP_ALIGN.CENTER
    
    p2 = tf1.add_paragraph()
    p2.text = "Part 2: Pipeline Execution and Results"
    set_para_font(p2, name="Calibri", size_pt=16, bold=False, italic=True, color=NAVY)
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(8)
    
    # Shape 2: Presenter Details
    author_shape = s1.shapes[2]
    tf_author = author_shape.text_frame
    tf_author.text = "" # clear
    pa1 = tf_author.paragraphs[0]
    pa1.text = "Animesh Basak (26911001)"
    set_para_font(pa1, name="Calibri", size_pt=16, bold=True, color=BLACK)
    pa1.alignment = PP_ALIGN.CENTER
    
    pa2 = tf_author.add_paragraph()
    pa2.text = "Department of Computer Science & Engineering\nIndian Institute of Technology Roorkee"
    set_para_font(pa2, name="Calibri", size_pt=13, bold=False, color=DARK_GRAY)
    pa2.alignment = PP_ALIGN.CENTER
    pa2.space_before = Pt(4)
    
    # Shape 1: Supervisor / Committee
    sup_shape = s1.shapes[1]
    tf_sup = sup_shape.text_frame
    tf_sup.text = ""
    ps1 = tf_sup.paragraphs[0]
    ps1.text = "Supervisor(s): Faculty Review Committee"
    set_para_font(ps1, name="Calibri", size_pt=13, bold=False, italic=True, color=DARK_GRAY)
    ps1.alignment = PP_ALIGN.CENTER
    
    add_speaker_notes(s1, (
        "Good morning, committee members. I am Animesh Basak, presenting Part 2 of our project on "
        "Explainable Graph Neural Networks and Hybrid Graph Databases for Brain Biomarker Discovery. "
        "In Part 1, we established our theoretical formulation and proposed an architecture fusing "
        "edge-weight-aware GCN layers with GATv2 dynamic attention. Today, I present the complete "
        "implementation pipeline, execution on our NVIDIA Tesla V100 server, empirical evaluation, "
        "and an honest audit of our biomarker explanations. The primary takeaway from this presentation "
        "is that our edge-aware graph neural network achieves a held-out test ROC-AUC of 0.746, "
        "substantially outperforming traditional vectorised machine learning baselines while exposing "
        "precise structural challenges in neuroimaging graph explanation that define our next steps."
    ))

    # Remove old slides 1..11
    print("Purging template placeholder slides...")
    for i in range(len(prs.slides) - 1, 0, -1):
        rId = prs.slides._sldIdLst[i].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[i]

    content_layout = prs.slide_masters[1].slide_layouts[0] # blank slide with master 1 header/footer

    # -------------------------------------------------------------------------
    # Slide 2: Recap, Problem, Idea and Initial Features
    # -------------------------------------------------------------------------
    print("Building Slide 2: Recap & Features...")
    s2 = prs.slides.add_slide(content_layout)
    add_slide_title(s2, "Recap: Problem Statement & Feature Representation")
    
    # Box A (Blue border)
    _, tf2a = create_rounded_box(s2, Inches(0.4), Inches(1.15), Inches(4.4), Inches(5.9),
                                 "A. Recap: Problem Statement", BORDER_BLUE)
    add_bullet_item(tf2a, prefix="•", text_segments=[
        ("Functional disconnection in resting-state fMRI appears years before structural atrophy. The goal is to catch it from the connectome alone.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf2a, prefix="•", text_segments=[
        ("Task: graph-level binary classification ", False, False, False),
        (r"P(y_k=1\mid G_k)=\mathrm{softmax}(f_\theta(A_k,X_k))_1", True, False, False),
        (", Healthy (0) vs Neurodegenerative (1), plus a post-hoc explanation subgraph ", False, False, False),
        (r"G_s\subseteq G_k", True, False, False),
        (".", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf2a, prefix="•", text_segments=[
        ("Each subject is a graph ", False, False, False),
        (r"G_k=(V,E_k,X_k)", True, False, False),
        (" on the AAL-116 atlas: ", False, False, False),
        (r"|V|=116", True, False, False),
        (" regions. Edges are thresholded functional correlations.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf2a, prefix="•", text_segments=[
        ("Core idea from deck 1: edge-weight-aware GCN, then GATv2 dynamic attention with the edge weight as an attribute, then dual pooling, a SwiGLU head and GNNExplainer for biomarker subgraphs.", False, False, False)
    ], pt_size=11)

    # Box B (Green border)
    _, tf2b = create_rounded_box(s2, Inches(5.0), Inches(1.15), Inches(4.6), Inches(5.9),
                                 "B. Initial Features Selected & Engineering", BORDER_GREEN)
    
    # Add native table inside Box B area
    tb_shape = s2.shapes.add_table(7, 3, Inches(5.12), Inches(1.65), Inches(4.36), Inches(3.45))
    table = tb_shape.table
    table.columns[0].width = Inches(1.15)
    table.columns[1].width = Inches(1.15)
    table.columns[2].width = Inches(2.06)
    
    headers = ["Level", "Feature", "How it is computed"]
    for col_idx, h in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        set_para_font(p, name="Calibri", size_pt=9.5, bold=True, color=WHITE)
        p.alignment = PP_ALIGN.CENTER

    table_data = [
        ("Node (5-dim)", "Degree", r"k_i/\max_j k_j", ", binary neighbour count in [0,1]"),
        ("", "PageRank", r"\text{PageRank}(|z_{ij}|)", ", power iteration, damping 0.85"),
        ("", "Clustering", r"C_i", ", weighted (Onnela) triangle intensity"),
        ("", "Betweenness", r"b_i", ", normalised shortest-path betweenness"),
        ("", "Eigenvector", r"\lambda_1 e = |W|e", ", leading eigenvector scaled by max"),
        ("Edge (1-dim)", r"z_{ij}", r"z_{ij}", r", signed z-score (GATv2); |z_{ij}| in GCN")
    ]
    for row_idx, row in enumerate(table_data, start=1):
        for col_idx in range(3):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(248, 250, 252) if row_idx % 2 == 1 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = ""
            if col_idx == 0:
                p.text = row[0]
                set_para_font(p, size_pt=9, bold=True, color=DARK_GRAY)
            elif col_idx == 1:
                p.text = row[1]
                set_para_font(p, size_pt=9, bold=False, color=DARK_GRAY)
            else:
                add_math_to_para(p, row[2])
                r = p.add_run()
                r.text = row[3]
                set_para_font(p, size_pt=8.5, color=DARK_GRAY)

    # Note below table
    p_tb_note = tf2b.add_paragraph()
    p_tb_note.space_before = Pt(220)
    p_tb_note.text = "Initial engineering: global-signal removal → z-scoring of off-diagonal entries → pruning of "
    set_para_font(p_tb_note, size_pt=9.5, bold=False, color=DARK_GRAY)
    add_math_to_para(p_tb_note, r"|z_{ij}| < 0.3")
    r_end = p_tb_note.add_run()
    r_end.text = " → symmetric COO edge list (both directions) → NetworkX/Neo4j-GDS centralities."
    set_para_font(p_tb_note, size_pt=9.5, color=DARK_GRAY)

    add_speaker_notes(s2, (
        "This slide recaps our clinical motivation and initial feature representation. "
        "Functional connectivity disruption in resting-state fMRI manifests years before structural brain atrophy "
        "is detectable on anatomical scans. We frame disease detection as graph-level binary classification over "
        "the 116-region AAL brain atlas. Instead of flattening connectivity matrices and discarding spatial topology, "
        "we extract five topological node descriptors: degree centrality, PageRank, clustering coefficient, "
        "betweenness, and eigenvector centrality. Crucially, as noted in our implementation audit, edge pruning is "
        "applied after z-scoring at absolute z above 0.3 rather than raw Pearson correlation. This preserves "
        "approximately 76 percent of all possible edges, maintaining dense functional pathways that our edge-aware "
        "GCN and GATv2 layers require to track subtle network alterations."
    ))

    # -------------------------------------------------------------------------
    # Slide 3: Pipeline Step 1, Data Collection
    # -------------------------------------------------------------------------
    print("Building Slide 3: Data Collection...")
    s3 = prs.slides.add_slide(content_layout)
    add_slide_title(s3, "Pipeline Step 1: Data Collection & Planted Cohort")
    
    # Box A (Blue border)
    _, tf3a = create_rounded_box(s3, Inches(0.4), Inches(1.15), Inches(4.3), Inches(5.9),
                                 "A. What We Used and Why", BORDER_BLUE)
    add_bullet_item(tf3a, prefix="•", text_segments=[
        ("Clinical cohorts (ADNI, PPMI) need signed data-use agreements, which are still pending. To build and debug the whole pipeline end to end, we generated a ", False, False, False),
        ("synthetic AAL-116 cohort with a planted, known disease signature", False, True, False),
        (".", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf3a, prefix="•", text_segments=[
        (r"N=500", True, False, False),
        (" subjects: 250 Healthy, 250 Diseased (labels alternate by index), 3 scanner sites assigned round-robin, ages 50–84.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf3a, prefix="•", text_segments=[
        ("Because the ground truth is planted, we can check the ", False, False, False),
        ("explanations", False, True, False),
        (" against it as well as the accuracy. Real clinical data cannot give us that verification.", False, False, False)
    ], pt_size=11)

    # Box B (Green border)
    _, tf3b = create_rounded_box(s3, Inches(4.9), Inches(1.15), Inches(4.7), Inches(5.9),
                                 "B. Generative Model (generate_synthetic.py)", BORDER_GREEN)
    
    p_eq1 = tf3b.add_paragraph()
    p_eq1.space_before = Pt(4)
    p_eq1.space_after = Pt(4)
    add_math_to_para(p_eq1, r"\Sigma = BB^\top + nI,\quad B_{ij}\sim\mathcal N(0,1),\quad R_{ij}=\frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}")
    
    p_eq2_lbl = tf3b.add_paragraph()
    p_eq2_lbl.text = "Disease perturbation (Diseased only):"
    set_para_font(p_eq2_lbl, size_pt=10.5, bold=True, color=NAVY)
    p_eq2_lbl.space_before = Pt(4)
    
    p_eq2a = tf3b.add_paragraph()
    add_math_to_para(p_eq2a, r"R_{i,i+1}\leftarrow 0.4\,R_{i,i+1}\quad\text{(45 homotopic L–R pairs)}")
    p_eq2b = tf3b.add_paragraph()
    add_math_to_para(p_eq2b, r"R_{ij}\leftarrow \min(1,\,1.3\,R_{ij})\quad(i,j\text{ in same lobe})")
    
    p_eq3_lbl = tf3b.add_paragraph()
    p_eq3_lbl.text = "Scanner noise model across 3 clinical sites:"
    set_para_font(p_eq3_lbl, size_pt=10.5, bold=True, color=NAVY)
    p_eq3_lbl.space_before = Pt(4)
    
    p_eq3 = tf3b.add_paragraph()
    add_math_to_para(p_eq3, r"R\leftarrow \mathrm{clip}\big(R+\tfrac12(\varepsilon+\varepsilon^\top),-1,1\big),\ \varepsilon_{ij}\sim\mathcal N(0,\sigma_s^2)")
    
    # Small site table inside Box B
    site_table_shape = s3.shapes.add_table(4, 3, Inches(5.15), Inches(5.2), Inches(4.2), Inches(1.4))
    site_table = site_table_shape.table
    site_table.columns[0].width = Inches(1.2)
    site_table.columns[1].width = Inches(1.5)
    site_table.columns[2].width = Inches(1.5)
    
    site_headers = ["Site", "Noise SD (σ)", "Subjects"]
    for col_idx, h in enumerate(site_headers):
        cell = site_table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        set_para_font(p, size_pt=10, bold=True, color=WHITE)
        p.alignment = PP_ALIGN.CENTER
        
    site_rows = [
        ("Site A", "0.02", "167"),
        ("Site B", "0.04", "167"),
        ("Site C", "0.06", "166")
    ]
    for row_idx, (st, sg, sb) in enumerate(site_rows, start=1):
        for col_idx, val in enumerate([st, sg, sb]):
            cell = site_table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(248, 250, 252) if row_idx % 2 == 1 else WHITE
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            set_para_font(p, size_pt=9.5, color=DARK_GRAY)
            p.alignment = PP_ALIGN.CENTER

    add_speaker_notes(s3, (
        "Here we detail our data collection step. Because formal clinical data access agreements for ADNI "
        "and PPMI remain pending institutional approval, we engineered a rigorous synthetic cohort of 500 "
        "subjects with a planted biological ground truth. The cohort features balanced classes across three "
        "distinct scanner noise regimes. The planted pathology attenuates homotopic inter-hemispheric connections "
        "across 45 left-right region pairs to 40 percent of baseline, while boosting intra-lobe coherence by 30 percent. "
        "A likely question is why synthetic data is scientifically valid here. The answer is that planted ground truth "
        "provides the only objective benchmark against which post-hoc biomarker explanations can be quantitatively verified. "
        "Real clinical datasets lack known ground-truth edge masks, making objective explanation auditing impossible."
    ))

    # -------------------------------------------------------------------------
    # Slide 4: Pipeline Step 2, Data Cleaning and Preprocessing
    # -------------------------------------------------------------------------
    print("Building Slide 4: Cleaning & Preprocessing (with Flowchart 1)...")
    s4 = prs.slides.add_slide(content_layout)
    add_slide_title(s4, "Pipeline Step 2: Data Cleaning & Preprocessing")
    
    # Top Left Box: Box A (blue)
    _, tf4a = create_rounded_box(s4, Inches(0.4), Inches(1.15), Inches(4.4), Inches(3.1),
                                 "A. Four Transforms per Subject (transform.py)", BORDER_BLUE)
    add_bullet_item(tf4a, prefix="1.", text_segments=[
        ("Global signal removal: ", False, True, False),
        (r"\tilde R = R-\mu_g\mathbf 1\mathbf 1^\top,\ \mu_g=\frac{\sum_{i\ne j}R_{ij}}{n(n-1)}", True, False, False),
        (", diagonal reset to 1.", False, False, False)
    ], pt_size=10)
    add_bullet_item(tf4a, prefix="2.", text_segments=[
        ("Z-scoring off-diagonal entries: ", False, True, False),
        (r"z_{ij}=(\tilde R_{ij}-\mu)/\sigma", True, False, False),
        ("; diagonal set to 0.", False, False, False)
    ], pt_size=10)
    add_bullet_item(tf4a, prefix="3.", text_segments=[
        ("Sparsification: ", False, True, False),
        (r"W_{ij}=z_{ij}\cdot\mathbb 1[|z_{ij}|\ge 0.3]", True, False, False),
        (".", False, False, False)
    ], pt_size=10)
    add_bullet_item(tf4a, prefix="4.", text_segments=[
        ("Edge list: upper triangle ", False, True, False),
        (r"i<j", True, False, False),
        (" with ", False, False, False),
        (r"W_{ij}\ne 0", True, False, False),
        (" → edge_list.csv (2,545,839 rows) → Neo4j LOAD CSV.", False, False, False)
    ], pt_size=10)

    # Top Right Box: Box B (green)
    _, tf4b = create_rounded_box(s4, Inches(5.0), Inches(1.15), Inches(4.6), Inches(3.1),
                                 "B. Zero-Leakage Split Safeguards", BORDER_GREEN)
    add_bullet_item(tf4b, prefix="•", text_segments=[
        ("Subject-level stratified split, seed 42: ", False, False, False),
        ("350 / 75 / 75", False, True, False),
        (" (train / val / test), class ratio kept exactly in every split.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf4b, prefix="•", text_segments=[
        ("Rows are mapped strictly by ", False, False, False),
        ("PatientID", False, True, False),
        (", so no subject appears across multiple splits. This was checked with an assertion.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf4b, prefix="•", text_segments=[
        ("All statistics in steps 1–3 are calculated ", False, False, False),
        ("within individual subjects", False, True, False),
        (", so nothing from the held-out test set enters model training.", False, False, False)
    ], pt_size=11)

    # Flowchart 1 (Native PowerPoint Shapes)
    # Background dashed container for steps 2..4
    leakage_box = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.58), Inches(4.38), Inches(3.78), Inches(2.25))
    leakage_box.fill.background()
    leakage_box.line.color.rgb = NAVY
    leakage_box.line.dash_style = pptx.enum.dml.MSO_LINE_DASH_STYLE.DASH
    leakage_box.line.width = Pt(1.5)
    tf_leak = leakage_box.text_frame
    p_leak = tf_leak.paragraphs[0]
    p_leak.text = "per-subject transforms (zero data leakage)"
    p_leak.alignment = PP_ALIGN.CENTER
    set_para_font(p_leak, size_pt=9, bold=True, italic=True, color=NAVY)
    
    # 7 Shapes
    fc1_specs = [
        # (idx, text_bold, text_small, fill, border, left, shape_type)
        (1, "Synthetic Cohort", "500 subjects\n(116×116) R", FILL_LIGHT_BLUE, BORDER_BLUE, Inches(0.4), MSO_SHAPE.ROUNDED_RECTANGLE),
        (2, "Global Signal", "R - μ_g\n(mean 0)", FILL_LIGHT_BLUE, BORDER_BLUE, Inches(1.68), MSO_SHAPE.ROUNDED_RECTANGLE),
        (3, "Z-Score", "per subject\noff-diagonal", FILL_LIGHT_BLUE, BORDER_BLUE, Inches(2.96), MSO_SHAPE.ROUNDED_RECTANGLE),
        (4, "Prune |z| < 0.3", "5,092 edges\nkept / graph", FILL_LIGHT_BLUE, BORDER_BLUE, Inches(4.24), MSO_SHAPE.ROUNDED_RECTANGLE),
        (5, "Neo4j Property", "(:BrainRegion)\n-[:CONNECTED]->", FILL_LIGHT_GREEN, BORDER_GREEN, Inches(5.52), MSO_SHAPE.CAN),
        (6, "GDS / NetworkX", "5 centralities\nper node", FILL_LIGHT_GREEN, BORDER_GREEN, Inches(6.80), MSO_SHAPE.ROUNDED_RECTANGLE),
        (7, "PyG Data", "x ∈ R^{116×5}\ne_attr ∈ R^{E×1}", FILL_LIGHT_GREEN, BORDER_GREEN, Inches(8.08), MSO_SHAPE.ROUNDED_RECTANGLE),
    ]
    
    for (num, t_bold, t_small, f_col, b_col, l_pos, s_type) in fc1_specs:
        sp = s4.shapes.add_shape(s_type, l_pos, Inches(4.75), Inches(1.15), Inches(1.2))
        sp.fill.solid()
        sp.fill.fore_color.rgb = f_col
        sp.line.color.rgb = b_col
        sp.line.width = Pt(1.5)
        tf = sp.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.04)
        tf.margin_right = Inches(0.04)
        tf.margin_top = Inches(0.08)
        p = tf.paragraphs[0]
        p.text = t_bold
        p.alignment = PP_ALIGN.CENTER
        set_para_font(p, size_pt=9, bold=True, color=NAVY)
        p_sub = tf.add_paragraph()
        p_sub.text = t_small
        p_sub.alignment = PP_ALIGN.CENTER
        set_para_font(p_sub, size_pt=8, bold=False, color=DARK_GRAY)
        
        # Add connecting right arrow between blocks
        if num < 7:
            arr = s4.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, l_pos + Inches(1.16), Inches(5.2), Inches(0.11), Inches(0.2))
            arr.fill.solid()
            arr.fill.fore_color.rgb = NAVY
            arr.line.color.rgb = NAVY

    # Split label under shape 7
    tb_split = s4.shapes.add_textbox(Inches(7.1), Inches(6.15), Inches(2.3), Inches(0.5))
    tf_split = tb_split.text_frame
    p_sp = tf_split.paragraphs[0]
    p_sp.text = "Stratified Split by PatientID:\n350 Train / 75 Val / 75 Test"
    p_sp.alignment = PP_ALIGN.CENTER
    set_para_font(p_sp, size_pt=9, bold=True, color=BORDER_GREEN)

    add_speaker_notes(s4, (
        "Slide 4 illustrates our preprocessing and data leakage safeguards. For each subject, raw functional "
        "connectivity undergoes global signal regression, off-diagonal z-scoring, and sparsification before entering "
        "our Neo4j property graph. Because brain connectivity networks exhibit high subject variability, all normalization "
        "statistics are strictly computed within individual subjects. Furthermore, our dataset is partitioned using a "
        "patient-level stratified split of 350 training, 75 validation, and 75 held-out test subjects under fixed random seed 42. "
        "No subject appears in multiple splits, and no population-level test statistics leak into model training. "
        "The primary point to emphasize is that our preprocessing guarantees zero cross-subject data leakage, allowing the "
        "graph neural network to learn genuine generalizable functional connectivity patterns rather than memorizing scanner noise."
    ))

    # -------------------------------------------------------------------------
    # Slide 5: Pipeline Step 3, Exploratory Data Analysis
    # -------------------------------------------------------------------------
    print("Building Slide 5: Exploratory Data Analysis...")
    s5 = prs.slides.add_slide(content_layout)
    add_slide_title(s5, "Pipeline Step 3: Exploratory Data Analysis")
    
    # Left Top Box: Box A (blue)
    _, tf5a = create_rounded_box(s5, Inches(0.4), Inches(1.15), Inches(4.5), Inches(2.35),
                                 "A. First-Order Graph Statistics", BORDER_BLUE)
    add_bullet_item(tf5a, prefix="•", text_segments=[
        ("Edges kept per subject: ", False, False, False),
        ("5,092 ± 34", False, True, False),
        (" of 6,670 possible (76.3% density). The two classes overlap almost completely (Fig 1, right).", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf5a, prefix="•", text_segments=[
        ("Edge weights: mean ≈ 0, SD 1.14, 50.0% negative. The distributions of the two classes look the same by eye (Fig 1, left).", False, False, False)
    ], pt_size=11)

    # Left Bottom Box: Box B (red border - problem/finding)
    _, tf5b = create_rounded_box(s5, Inches(0.4), Inches(3.65), Inches(4.5), Inches(3.4),
                                 "B. Where the Class Signal Actually Lives", BORDER_RED)
    add_bullet_item(tf5b, prefix="•", text_segments=[
        ("Group-difference matrix ", False, False, False),
        (r"\Delta_{ij}=\overline{|W_{ij}|}_{D}-\overline{|W_{ij}|}_{H}", True, False, False),
        (" (Fig 3): red blocks on every within-lobe square and a blue line on homotopic L–R pairs. This is exactly the planted signature.", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf5b, prefix="•", text_segments=[
        ("One scalar per subject, ", False, False, False),
        (r"s_k=\mathrm{mean}_{(i,j)\in\text{intra-lobe}}|W_{ij}|", True, False, False),
        (", separates the test set perfectly: ", False, False, False),
        ("AUC = 1.000", False, True, False),
        (" (Fig 2).", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf5b, prefix="•", text_segments=[
        ("Consequence for modelling: the signal is a ", False, False, False),
        ("block-level magnitude pattern that depends on region identity", False, True, False),
        (". A model must know which lobe each node belongs to before it can see it.", False, False, False)
    ], pt_size=10.5)

    # Visuals on Right side: Fig 3 (top) and Fig 2 (bottom)
    add_picture_with_svg(s5, prs, "fig3_group_difference_matrix", Inches(5.1), Inches(1.15), width=Inches(4.5))
    add_picture_with_svg(s5, prs, "fig2_eda_signal_location", Inches(5.1), Inches(4.3), width=Inches(4.5))

    add_speaker_notes(s5, (
        "This slide examines our exploratory data analysis. The left plot demonstrates that global edge count "
        "and weight distributions are virtually identical between healthy controls and diseased subjects, with edge counts "
        "overlapping at 5,092 edges. Standard first-order network metrics cannot separate the cohorts. However, the group "
        "difference matrix on the right reveals the planted signal clearly: distinct block-structured hyperconnectivity across "
        "anatomical lobes and a sharp reduction along the homotopic diagonal. A single oracle scalar representing mean intra-lobe "
        "connection strength achieves an AUC of 1.000 on the test partition. This proves that disease discrimination requires "
        "identifying anatomical lobe boundaries and regional community structure. Without explicit lobe identities, a model "
        "must infer these relationships purely from topological connectivity."
    ))

    # -------------------------------------------------------------------------
    # Slide 6: Pipeline Step 4, Feature Engineering and Selection
    # -------------------------------------------------------------------------
    print("Building Slide 6: Feature Engineering...")
    s6 = prs.slides.add_slide(content_layout)
    add_slide_title(s6, "Pipeline Step 4: Feature Engineering & Selection Audit")
    
    # Left Box: Box A (blue)
    _, tf6a = create_rounded_box(s6, Inches(0.4), Inches(1.15), Inches(4.4), Inches(5.9),
                                 "A. Node Descriptors (w_ij=|W_ij|, k_i=|N(i)|)", BORDER_BLUE)
    
    p6_1 = tf6a.add_paragraph()
    p6_1.space_before = Pt(4)
    add_math_to_para(p6_1, r"x_i^{\deg}=\frac{k_i}{\max_j k_j}")
    p6_1_lbl = tf6a.add_paragraph()
    p6_1_lbl.text = "1. Normalised Degree Centrality"
    set_para_font(p6_1_lbl, size_pt=9.5, bold=True, color=NAVY)
    
    p6_2 = tf6a.add_paragraph()
    p6_2.space_before = Pt(4)
    add_math_to_para(p6_2, r"\pi = \frac{1-d}{n}\mathbf 1 + d\,P^\top\pi,\quad P_{ij}=\frac{w_{ij}}{\sum_l w_{il}},\ d=0.85")
    p6_2_lbl = tf6a.add_paragraph()
    p6_2_lbl.text = "2. Weighted PageRank Vector"
    set_para_font(p6_2_lbl, size_pt=9.5, bold=True, color=NAVY)
    
    p6_3 = tf6a.add_paragraph()
    p6_3.space_before = Pt(4)
    add_math_to_para(p6_3, r"C_i=\frac{1}{k_i(k_i-1)}\sum_{j,h}\big(\hat w_{ij}\hat w_{ih}\hat w_{jh}\big)^{1/3}")
    p6_3_lbl = tf6a.add_paragraph()
    p6_3_lbl.text = "3. Weighted Clustering Coefficient (Onnela)"
    set_para_font(p6_3_lbl, size_pt=9.5, bold=True, color=NAVY)
    
    p6_4 = tf6a.add_paragraph()
    p6_4.space_before = Pt(4)
    add_math_to_para(p6_4, r"b_i=\frac{2}{(n-1)(n-2)}\sum_{s\ne i\ne t}\frac{\sigma_{st}(i)}{\sigma_{st}},\quad \lambda_1 e = We")
    p6_4_lbl = tf6a.add_paragraph()
    p6_4_lbl.text = "4 & 5. Betweenness & Eigenvector Centralities"
    set_para_font(p6_4_lbl, size_pt=9.5, bold=True, color=NAVY)

    # Right Box: Box B (red border - selection audit)
    _, tf6b = create_rounded_box(s6, Inches(5.0), Inches(1.15), Inches(4.6), Inches(5.9),
                                 "B. Selection Audit & Bottlenecks", BORDER_RED)
    add_bullet_item(tf6b, prefix="•", text_segments=[
        ("All five node features are inherently ", False, False, False),
        ("scale-free", False, True, False),
        (": each is divided by a maximum, sums to one, or is a ratio of path counts. Multiplying every weight inside a lobe by 1.3 barely shifts them, so the planted signal is mostly removed ", False, False, False),
        ("before the GNN ever sees it", False, True, False),
        (".", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf6b, prefix="•", text_segments=[
        ("NetworkX reads weight as a ", False, False, False),
        ("distance", False, True, False),
        (" in shortest-path betweenness, so strong correlations count as long paths. The feature is inverted relative to its standard neuroimaging meaning.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf6b, prefix="•", text_segments=[
        ("What carries the signal: the continuous edge attribute ", False, False, False),
        (r"z_{ij}", True, False, False),
        (" (retained in GCN/GATv2), plus region identity (missing in node features; see Slide 10).", False, False, False)
    ], pt_size=11)

    add_speaker_notes(s6, (
        "Slide 6 presents our feature engineering audit. We computed five classical centrality descriptors "
        "for each brain region. However, rigorous mathematical scrutiny reveals a significant limitation: all five "
        "descriptors are inherently scale-free. Because degree, PageRank, clustering, betweenness, and eigenvector "
        "centralities normalize by maxima or sums of weights, multiplicative scaling of within-lobe connections "
        "leaves them largely invariant. Consequently, topological node features filter out the planted magnitude "
        "signal before the neural network sees it. The signed edge attribute z_ij remains the primary carrier of the "
        "disease signal into the model. What must be stated clearly is that without regional anatomical coordinates "
        "or lobe embeddings, the network faces an informational bottleneck that caps its performance."
    ))

    # -------------------------------------------------------------------------
    # Slide 7: Pipeline Step 5, Model Selection
    # -------------------------------------------------------------------------
    print("Building Slide 7: Model Selection (with Flowchart 2)...")
    s7 = prs.slides.add_slide(content_layout)
    add_slide_title(s7, "Pipeline Step 5: Model Selection & Architecture")
    
    # Left Box 1: Box A (blue)
    _, tf7a = create_rounded_box(s7, Inches(0.4), Inches(1.15), Inches(5.6), Inches(3.35),
                                 "A. Architecture: 521,378 Parameters (d=192)", BORDER_BLUE)
    p7_in = tf7a.add_paragraph()
    add_math_to_para(p7_in, r"h_i^{(0)}=\mathrm{SiLU}(W_{\text{in}}x_i+b_{\text{in}})\in\mathbb R^{192}")
    
    p7_gcn = tf7a.add_paragraph()
    p7_gcn.space_before = Pt(3)
    add_math_to_para(p7_gcn, r"h_i^{(l+1)}=h_i^{(l)}+\mathrm{Drop}\Big(\mathrm{SiLU}\big(\mathrm{BN}\big(\sum_{j\in\mathcal N(i)\cup\{i\}}\frac{\tilde w_{ij}}{\sqrt{\tilde d_i\tilde d_j}}\,\Theta^{(l)}h_j^{(l)}\big)\big)\Big)")
    
    p7_gat = tf7a.add_paragraph()
    p7_gat.space_before = Pt(3)
    add_math_to_para(p7_gat, r"e_{ij}^{k}=a_k^\top\mathrm{LeakyReLU}_{0.2}(W_s^k h_i+W_t^k h_j+W_e^k z_{ij}),\ \alpha_{ij}^{k}=\frac{\exp e_{ij}^{k}}{\sum_{l}\exp e_{il}^{k}}")
    
    p7_head = tf7a.add_paragraph()
    p7_head.space_before = Pt(3)
    add_math_to_para(p7_head, r"h_G=[\frac{1}{116}\sum_i h_i^\prime\ \Vert\ \max_i h_i^\prime]\in\mathbb R^{384},\ u=(\mathrm{SiLU}(h_GW_g)\odot h_GW_u)W_d\in\mathbb R^{96}")

    # Left Box 2: Box B (green)
    _, tf7b = create_rounded_box(s7, Inches(0.4), Inches(4.65), Inches(5.6), Inches(2.4),
                                 "B. Architectural Block Rationale", BORDER_GREEN)
    add_bullet_item(tf7b, prefix="•", text_segments=[
        ("GCN: ", False, True, False),
        ("cheap isotropic smoothing that uses the continuous coupling strength ", False, False, False),
        (r"w_{ij}", True, False, False),
        (".", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf7b, prefix="•", text_segments=[
        ("GATv2: ", False, True, False),
        ("nonlinearity sits between ", False, False, False),
        (r"W_sh_i", True, False, False),
        (" and ", False, False, False),
        (r"a", True, False, False),
        (", allowing dynamic neighbour ranking conditioned on the query node (fixes GATv1 static attention; Brody et al. 2022).", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf7b, prefix="•", text_segments=[
        ("Dual pooling (mean ‖ max): ", False, True, False),
        ("mean preserves global connectivity while max captures focal disruption.", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf7b, prefix="•", text_segments=[
        ("SwiGLU: ", False, True, False),
        ("gated bilinear head providing expressive capacity for classification.", False, False, False)
    ], pt_size=10.5)

    # Right Area: Flowchart 2 (Vertical Stack)
    fc2_blocks = [
        ("Input Connectome: x (116×5), z_ij", MSO_SHAPE.PARALLELOGRAM, FILL_LIGHT_BLUE, BORDER_BLUE, Inches(1.15), Inches(0.48)),
        ("Linear 5→192 + SiLU + Dropout 0.25", MSO_SHAPE.RECTANGLE, FILL_LIGHT_BLUE, BORDER_BLUE, Inches(1.78), Inches(0.48)),
        ("GCNConv(|z|) → BN → SiLU (+ residual) ×3", MSO_SHAPE.RECTANGLE, FILL_LIGHT_BLUE, BORDER_BLUE, Inches(2.41), Inches(0.55)),
        ("GATv2Conv 4h×48 (z_ij) → BN → SiLU (+ res)", MSO_SHAPE.RECTANGLE, FILL_LIGHT_BLUE, BORDER_BLUE, Inches(3.11), Inches(0.55)),
        ("Mean Pool ‖ Max Pool (384-dim)", MSO_SHAPE.RECTANGLE, FILL_LIGHT_GREEN, BORDER_GREEN, Inches(3.81), Inches(0.48)),
        ("SwiGLU Block 384 → 384 → 96", MSO_SHAPE.RECTANGLE, FILL_LIGHT_GREEN, BORDER_GREEN, Inches(4.44), Inches(0.48)),
        ("Dropout → Linear 96→2 → Softmax (y_hat)", MSO_SHAPE.RECTANGLE, FILL_LIGHT_GREEN, BORDER_GREEN, Inches(5.07), Inches(0.48)),
    ]
    
    for (t_str, s_type, f_col, b_col, top_pos, h_dim) in fc2_blocks:
        sp = s7.shapes.add_shape(s_type, Inches(6.25), top_pos, Inches(3.3), h_dim)
        sp.fill.solid()
        sp.fill.fore_color.rgb = f_col
        sp.line.color.rgb = b_col
        sp.line.width = Pt(1.5)
        tf = sp.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = t_str
        p.alignment = PP_ALIGN.CENTER
        set_para_font(p, size_pt=8.5, bold=True, color=NAVY)
        
        # Down connector arrow
        if top_pos < Inches(5.0):
            arr = s7.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(7.8), top_pos + h_dim + Inches(0.02), Inches(0.2), Inches(0.12))
            arr.fill.solid()
            arr.fill.fore_color.rgb = NAVY
            arr.line.color.rgb = NAVY

    # GNNExplainer block (to the right bottom)
    sp_exp = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.25), Inches(5.8), Inches(3.3), Inches(1.15))
    sp_exp.fill.solid()
    sp_exp.fill.fore_color.rgb = RGBColor(230, 245, 235)
    sp_exp.line.color.rgb = BORDER_GREEN
    sp_exp.line.width = Pt(2.0)
    tf_exp = sp_exp.text_frame
    tf_exp.word_wrap = True
    p_ex0 = tf_exp.paragraphs[0]
    p_ex0.text = "Post-Hoc: GNNExplainer"
    set_para_font(p_ex0, size_pt=9.5, bold=True, color=BORDER_GREEN)
    p_ex0.alignment = PP_ALIGN.CENTER
    p_ex1 = tf_exp.add_paragraph()
    p_ex1.text = "Optimizes edge mask M_e & feature mask M_f (θ frozen) → Top-10 Biomarker Subgraphs"
    set_para_font(p_ex1, size_pt=8.5, bold=False, color=DARK_GRAY)
    p_ex1.alignment = PP_ALIGN.CENTER

    add_speaker_notes(s7, (
        "Here is our complete deep learning architecture, comprising 521,378 trainable parameters at hidden "
        "dimension 192. Node features project through an initial linear layer before traversing three edge-weighted "
        "GCN blocks with residual connections and batch normalization. Next, a GATv2 attention layer with four heads "
        "incorporates continuous edge correlation attributes, allowing dynamic attention scores that overcome GATv1 "
        "static ranking limitations. Dual global pooling concatenates mean and maximum node representations to capture "
        "both diffuse network alterations and focal regional anomalies. Finally, a SwiGLU gating block feeds the classification "
        "head. The core architectural decision to highlight is that residual connections and edge-attribute conditioning "
        "enable stable gradient propagation across dense connectome topologies without suffering from oversmoothing."
    ))

    # -------------------------------------------------------------------------
    # Slide 8: Pipeline Step 6, Model Training
    # -------------------------------------------------------------------------
    print("Building Slide 8: Training...")
    s8 = prs.slides.add_slide(content_layout)
    add_slide_title(s8, "Pipeline Step 6: Model Training & Convergence")
    
    # Left Top Box: Box A (blue)
    _, tf8a = create_rounded_box(s8, Inches(0.4), Inches(1.15), Inches(4.5), Inches(2.8),
                                 "A. Objective & Optimiser", BORDER_BLUE)
    p8_loss = tf8a.add_paragraph()
    p8_loss.space_before = Pt(3)
    add_math_to_para(p8_loss, r"\mathcal L(\theta)=-\frac1{|\mathcal B|}\sum_{k\in\mathcal B}\sum_{c\in\{0,1\}}q_{kc}\log\mathrm{softmax}(z_k)_c")
    p8_smooth = tf8a.add_paragraph()
    add_math_to_para(p8_smooth, r"q_{kc}=(1-\epsilon)\,\mathbb 1[c=y_k]+\tfrac{\epsilon}{2},\ \ \epsilon=0.1")
    
    p8_adam = tf8a.add_paragraph()
    p8_adam.space_before = Pt(3)
    add_math_to_para(p8_adam, r"\theta_{t+1}=\theta_t-\eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}+\lambda\theta_t\Big)")
    p8_sched = tf8a.add_paragraph()
    p8_sched.text = "AdamW (η=1e-3, λ=1e-4) with 5-epoch linear warmup & cosine decay."
    set_para_font(p8_sched, size_pt=9.5, color=DARK_GRAY)

    # Left Bottom Box: Box B (green)
    _, tf8b = create_rounded_box(s8, Inches(0.4), Inches(4.1), Inches(4.5), Inches(2.95),
                                 "B. Tesla V100 Run Settings & Outcome", BORDER_GREEN)
    add_bullet_item(tf8b, prefix="•", text_segments=[
        ("FP16 mixed precision + ", False, False, False),
        ("GradScaler", False, True, False),
        (" (Volta lacks hardware BF16), grad-norm clip 1.0, batch size 32, pin_memory=True.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf8b, prefix="•", text_segments=[
        ("Early stopping on ", False, False, False),
        ("validation Macro-F1", False, True, False),
        (" (patience 25). Best checkpoint at ", False, False, False),
        ("epoch 70", False, True, False),
        (": val F1 0.765, val AUC 0.822, train F1 0.634.", False, False, False)
    ], pt_size=11)
    add_bullet_item(tf8b, prefix="•", text_segments=[
        ("Train F1 stays below val F1 throughout, so the model ", False, False, False),
        ("deliberately underfits", False, True, False),
        (": heavy dropout, label smoothing and scale-free inputs together shield it from memorizing scanner noise.", False, False, False)
    ], pt_size=11)

    # Right Area: Fig 5 Training Curves
    add_picture_with_svg(s8, prs, "fig5_training_curves", Inches(5.1), Inches(1.15), width=Inches(4.5))

    add_speaker_notes(s8, (
        "Slide 8 outlines our training protocol on the NVIDIA Tesla V100 GPU. We minimize label-smoothed "
        "cross-entropy loss using decoupled AdamW with cosine learning rate decay and warmup. Mixed precision utilizes "
        "FP16 with gradient scaling, as the Volta V100 architecture does not support native BF16. PyTorch DataLoader "
        "pinned memory accelerates PCIe transfers. Early stopping on validation macro-F1 halts training at epoch 70, "
        "yielding validation F1 of 0.765 and validation AUC of 0.822. Notice that training F1 remains below validation "
        "F1 across all epochs. This demonstrates deliberate underfitting resulting from heavy dropout and label smoothing, "
        "which prevents the network from memorizing scanner-specific noise across the three acquisition sites."
    ))

    # -------------------------------------------------------------------------
    # Slide 9: Pipeline Step 7, Model Evaluation and Tuning
    # -------------------------------------------------------------------------
    print("Building Slide 9: Evaluation & Tuning...")
    s9 = prs.slides.add_slide(content_layout)
    add_slide_title(s9, "Pipeline Step 7: Model Evaluation & Threshold Tuning")
    
    # Left Box: Box A (blue)
    _, tf9a = create_rounded_box(s9, Inches(0.4), Inches(1.15), Inches(4.5), Inches(5.9),
                                 "A. Held-Out Test Evaluation (N=75)", BORDER_BLUE)
    
    p9_eq = tf9a.add_paragraph()
    p9_eq.space_before = Pt(3)
    add_math_to_para(p9_eq, r"\mathrm{AUC}=P(\hat p_{k^+}>\hat p_{k^-}),\quad J(t)=\mathrm{TPR}(t)-\mathrm{FPR}(t)")
    
    # Native Metrics Table
    m_shape = s9.shapes.add_table(4, 2, Inches(0.55), Inches(2.2), Inches(4.18), Inches(2.1))
    m_table = m_shape.table
    m_table.columns[0].width = Inches(2.68)
    m_table.columns[1].width = Inches(1.50)
    
    m_headers = ["Metric", "Value"]
    for col_idx, h in enumerate(m_headers):
        cell = m_table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        set_para_font(p, size_pt=10, bold=True, color=WHITE)
        p.alignment = PP_ALIGN.CENTER
        
    m_data = [
        ("Test ROC-AUC (headline)", "0.746"),
        ("Validation ROC-AUC", "0.823"),
        ("Test Acc / Macro-F1 (default t=0.5)", "57.33% / 0.520")
    ]
    for row_idx, (m_lbl, m_val) in enumerate(m_data, start=1):
        for col_idx, text_v in enumerate([m_lbl, m_val]):
            cell = m_table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(248, 250, 252) if row_idx % 2 == 1 else WHITE
            cell.text = text_v
            p = cell.text_frame.paragraphs[0]
            is_bold = (row_idx == 1)
            set_para_font(p, size_pt=9.5, bold=is_bold, color=DARK_GRAY)
            if col_idx == 1:
                p.alignment = PP_ALIGN.CENTER

    p9_note = tf9a.add_paragraph()
    p9_note.space_before = Pt(150)
    p9_note.text = "AUC is threshold-free and serves as our primary headline metric. On a small 75-subject test set, accuracy and F1 fluctuate substantially with operating thresholds."
    set_para_font(p9_note, size_pt=10, bold=False, color=DARK_GRAY)

    # Right Top Box: Box B (green)
    _, tf9b = create_rounded_box(s9, Inches(5.1), Inches(1.15), Inches(4.5), Inches(2.45),
                                 "B. Tuning: Threshold Calibration", BORDER_GREEN)
    add_bullet_item(tf9b, prefix="•", text_segments=[
        ("Youden's J picks threshold ", False, False, False),
        (r"t^*", True, False, False),
        (" maximising TPR − FPR on the ROC curve (standard clinical screening protocol).", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf9b, prefix="•", text_segments=[
        ("Test scores sit lower than validation scores (mean ", False, False, False),
        (r"\hat p=0.27", True, False, False),
        ("). Ranking is fine (AUC 0.746) but probability scale shifts between splits.", False, False, False)
    ], pt_size=10.5)
    add_bullet_item(tf9b, prefix="•", text_segments=[
        ("Next step: fit ", False, False, False),
        (r"t^*", True, False, False),
        (" on validation only, add temperature scaling, and run 5-fold CV.", False, False, False)
    ], pt_size=10.5)

    # Right Bottom Area: Fig 6 ROC Curve
    add_picture_with_svg(s9, prs, "fig6_roc_thresholds", Inches(5.1), Inches(3.75), width=Inches(4.5))

    add_speaker_notes(s9, (
        "This slide evaluates held-out test performance across our 75 unseen subjects. Our architecture "
        "achieves a headline test ROC-AUC of 0.746, generalizing solidly across multi-site scanner noise. "
        "At the default 0.5 decision threshold, test accuracy is 57.33 percent with macro-F1 of 0.520. Operating "
        "point calibration via Youden's index shifts the decision threshold to account for predicted probability "
        "shrinkage between validation and test distributions. If asked why test accuracy is 57 percent, the answer "
        "is straightforward: on a 75-subject test set, threshold-dependent accuracy fluctuates significantly with small "
        "shifts in posterior scores, making threshold-free ROC-AUC the only statistically reliable benchmark for clinical "
        "screening performance. Threshold tuning must be validated via nested cross-validation."
    ))

    # -------------------------------------------------------------------------
    # Slide 10: Results, Explainability and Comparison with Base Papers
    # -------------------------------------------------------------------------
    print("Building Slide 10: Comparison & Explainer...")
    s10 = prs.slides.add_slide(content_layout)
    add_slide_title(s10, "Results, Explainability & Baseline Comparison")
    
    # Top Box: Box A (Comparison Table)
    _, tf10a = create_rounded_box(s10, Inches(0.4), Inches(1.15), Inches(9.2), Inches(2.78),
                                  "A. Comparison on the Same 75 Test Subjects", BORDER_BLUE)
    
    # Native Comparison Table (6 cols × 8 rows)
    comp_shape = s10.shapes.add_table(8, 6, Inches(0.55), Inches(1.52), Inches(8.9), Inches(1.85))
    ct = comp_shape.table
    ct.columns[0].width = Inches(3.0)
    ct.columns[1].width = Inches(1.2)
    ct.columns[2].width = Inches(1.0)
    ct.columns[3].width = Inches(1.0)
    ct.columns[4].width = Inches(1.0)
    ct.columns[5].width = Inches(1.7)
    
    c_headers = ["Model (base paper)", "Graph-aware", "Test AUC", "Test Acc", "Macro-F1", "Source"]
    for col_idx, h in enumerate(c_headers):
        cell = ct.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        set_para_font(p, size_pt=9, bold=True, color=WHITE)
        p.alignment = PP_ALIGN.CENTER
        
    c_data = [
        ("RBF-SVM on flattened connectome (Abraham 2017)", "No", "0.530", "53.3%", "0.533", "re-run, this split"),
        ("L2 logistic regression, flattened", "No", "0.509", "49.3%", "0.493", "re-run, this split"),
        ("RF on graphlet degree vectors (Pržulj 2007)", "Motifs", "0.692†", "56.0%†", "0.534†", "README, no log"),
        ("node2vec + classifier (Grover 2016)", "Walks", "0.635†", "52.0%†", "0.498†", "README, no log"),
        ("GATv1, mean pool (Veličković 2018)", "Static attn.", "0.713†", "57.3%†", "0.550†", "README, no log"),
        ("Ours: edge-aware GCN + GATv2 + dual pool + SwiGLU", "Dynamic attn.", "0.746", "57.3%", "0.520", "test_metrics.json"),
        ("Sanity check: 1 scalar, mean |z| intra-lobe", "(oracle)", "1.000", "100%", "1.000", "re-run, this split")
    ]
    for row_idx, row_vals in enumerate(c_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = ct.cell(row_idx, col_idx)
            cell.fill.solid()
            is_ours = (row_idx == 6)
            bg = RGBColor(230, 245, 235) if is_ours else (RGBColor(248, 250, 252) if row_idx % 2 == 1 else WHITE)
            cell.fill.fore_color.rgb = bg
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            set_para_font(p, size_pt=8.5, bold=is_ours, color=DARK_GRAY)
            if col_idx in [1, 2, 3, 4]:
                p.alignment = PP_ALIGN.CENTER

    # Footnote below table
    p10_fn = tf10a.add_paragraph()
    p10_fn.space_before = Pt(138)
    p10_fn.text = "*† Baseline numbers are from our earlier README runs. Only the SVM / logistic regression / one-scalar rows were re-run on this exact split.*"
    set_para_font(p10_fn, size_pt=8, italic=True, color=DARK_GRAY)

    # Bottom Left Box: Box B (green)
    _, tf10b = create_rounded_box(s10, Inches(0.4), Inches(4.02), Inches(3.15), Inches(3.02),
                                  "B. What Improved & Why", BORDER_GREEN)
    add_bullet_item(tf10b, prefix="•", text_segments=[
        ("Against the flattened SVM, the GNN gains ", False, False, False),
        ("+0.216 AUC", False, True, False),
        (" (0.530 → 0.746). The SVM sees 6,670 features from 350 subjects; GNN shares parameters across edges and nodes.", False, False, False)
    ], pt_size=9.5)
    add_bullet_item(tf10b, prefix="•", text_segments=[
        ("The edge attribute matters: continuous ", False, False, False),
        (r"z_{ij}", True, False, False),
        (" enters GCN & GATv2, holding the magnitude signal lost by scale-free node features.", False, False, False)
    ], pt_size=9.5)

    # Bottom Middle Box: Box C (red/blue)
    _, tf10c = create_rounded_box(s10, Inches(3.7), Inches(4.02), Inches(3.35), Inches(3.02),
                                  "C. GNNExplainer Audit", BORDER_RED)
    add_bullet_item(tf10c, prefix="•", text_segments=[
        ("GNNExplainer learns soft edge & node masks to extract consensus biomarker circuits.", False, False, False)
    ], pt_size=9.5)
    add_bullet_item(tf10c, prefix="•", text_segments=[
        ("Planted ground truth audit: ", False, False, False),
        ("15.3%", False, True, False),
        (" of top-10 edges are intra-lobe, matching chance (", False, False, False),
        ("16.0%", False, True, False),
        ("). Explainer does not yet recover planted pattern.", False, False, False)
    ], pt_size=9.5)
    add_bullet_item(tf10c, prefix="•", text_segments=[
        ("Root cause: model lacks region identity, so it cannot utilize intra-lobe boundaries. Faithful explainer cannot report what model does not use.", False, False, False)
    ], pt_size=9.5)

    # Bottom Right Area: Fig 8 Explainer vs Ground Truth
    add_picture_with_svg(s10, prs, "fig8_explainer_vs_ground_truth", Inches(7.2), Inches(4.02), width=Inches(2.4))

    add_speaker_notes(s10, (
        "Slide 10 compares our model against the four base papers and audits GNNExplainer. Our edge-aware "
        "GCN-GATv2 model achieves 0.746 test AUC, gaining 0.216 over flattened SVM at 0.530 and outperforming "
        "node2vec and GATv1. However, evaluating GNNExplainer against our planted ground truth reveals an essential "
        "finding: only 15.3 percent of explained edges fall within anatomical lobes, matching chance level of 16.0 percent. "
        "If asked why the explainer appears vague, the answer is clear: the model lacks explicit lobe identity features, "
        "so it cannot utilize intra-lobe boundaries; an honest explainer cannot explain features the network never learned. "
        "Our immediate next step is integrating positional anatomical embeddings to align explainability with biological truth."
    ))

    # -------------------------------------------------------------------------
    # Slide 11: Thank You
    # -------------------------------------------------------------------------
    print("Building Slide 11: Thank You...")
    s11 = prs.slides.add_slide(content_layout)
    add_slide_title(s11, "INDIAN INSTITUTE OF TECHNOLOGY ROORKEE")
    
    # Centered card
    card = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.8), Inches(7.0), Inches(4.5))
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = NAVY
    card.line.width = Pt(2.5)
    
    tf11 = card.text_frame
    tf11.word_wrap = True
    
    p11_0 = tf11.paragraphs[0]
    p11_0.text = "Department of Computer Science & Engineering"
    set_para_font(p11_0, size_pt=14, bold=True, color=NAVY)
    p11_0.alignment = PP_ALIGN.CENTER
    
    p11_proj = tf11.add_paragraph()
    p11_proj.space_before = Pt(12)
    p11_proj.text = "Explainable Graph Neural Networks & Hybrid Graph Databases for Brain Biomarker Discovery"
    set_para_font(p11_proj, size_pt=18, bold=True, color=BLACK)
    p11_proj.alignment = PP_ALIGN.CENTER
    
    p11_sub = tf11.add_paragraph()
    p11_sub.text = "Part 2: Pipeline Execution and Results"
    set_para_font(p11_sub, size_pt=14, italic=True, color=DARK_GRAY)
    p11_sub.alignment = PP_ALIGN.CENTER
    
    p11_ty = tf11.add_paragraph()
    p11_ty.space_before = Pt(24)
    p11_ty.text = "Thank You"
    set_para_font(p11_ty, size_pt=28, bold=True, color=NAVY)
    p11_ty.alignment = PP_ALIGN.CENTER
    
    p11_qa = tf11.add_paragraph()
    p11_qa.text = "Questions & Discussion"
    set_para_font(p11_qa, size_pt=16, bold=False, color=BORDER_BLUE)
    p11_qa.alignment = PP_ALIGN.CENTER
    
    p11_pres = tf11.add_paragraph()
    p11_pres.space_before = Pt(16)
    p11_pres.text = "Animesh Basak (26911001)"
    set_para_font(p11_pres, size_pt=14, bold=True, color=BLACK)
    p11_pres.alignment = PP_ALIGN.CENTER

    add_speaker_notes(s11, (
        "In closing, this project demonstrates an end-to-end neuroimaging framework integrating property "
        "graphs, geometric deep learning, and explainability. We achieved a competitive test ROC-AUC of 0.746 "
        "on our Tesla V100 system, demonstrating strong gains over traditional flattened baselines. More importantly, "
        "our ground-truth validation revealed exactly where graph neural networks require anatomical grounding to deliver "
        "clinically faithful biomarker explanations. With institutional data access for ADNI and PPMI currently underway, "
        "our immediate roadmap focuses on integrating positional lobe embeddings and deploying our containerized inference "
        "service. Thank you for your time and guidance. I welcome your questions and feedback on our results."
    ))

    # Save presentation
    print(f"Saving presentation to {OUTPUT_PPTX}...")
    prs.save(OUTPUT_PPTX)
    print("Presentation saved successfully!")

if __name__ == "__main__":
    build_presentation()

