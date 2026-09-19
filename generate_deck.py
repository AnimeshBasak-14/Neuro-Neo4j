"""
generate_deck.py
================
Builds a high-impact, professional, 16:9 widescreen PowerPoint presentation (.pptx)
from ppt.md, embedding the high-resolution charts from logs/plots/.

Author: basakanimesh14
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Color Palette: Deep clinical slate + navy + vibrant accents
DARK_BG     = RGBColor(15, 23, 42)      # Slate 900
LIGHT_BG    = RGBColor(248, 250, 252)   # Slate 50
CARD_BG     = RGBColor(255, 255, 255)   # White
TEXT_DARK   = RGBColor(30, 41, 59)      # Slate 800
TEXT_MUTED  = RGBColor(100, 116, 139)   # Slate 500
TEXT_LIGHT  = RGBColor(241, 245, 249)   # Slate 100
ACCENT_BLUE = RGBColor(14, 165, 233)   # Sky 500
ACCENT_NAVY = RGBColor(2, 132, 199)    # Light blue / Primary
ACCENT_GREEN= RGBColor(16, 185, 129)   # Emerald 500
CARD_BORDER = RGBColor(226, 232, 240)   # Slate 200

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    plots_dir = Path("logs/plots")

    def add_header(slide, title_text, category_text="NEURO-NEO4J CLINICAL AI FRAMEWORK"):
        # Top banner category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
        tf = cat_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = category_text.upper()
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = ACCENT_NAVY

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
        tf2 = title_box.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = title_text
        p2.font.size = Pt(22)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_DARK

    def set_slide_bg(slide, color):
        background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        background.fill.solid()
        background.fill.fore_color.rgb = color
        background.line.fill.background()
        return background

    # -------------------------------------------------------------------------
    # SLIDE 1: Title Slide (Dark Theme)
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1, DARK_BG)

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.0))
    tf = t_box.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "NEURO-NEO4J"
    p0.font.size = Pt(36)
    p0.font.bold = True
    p0.font.color.rgb = ACCENT_BLUE

    p1 = tf.add_paragraph()
    p1.text = "Hybrid Graph Database & Explainable GNN Pipeline\nfor Brain Biomarker Discovery in Neurodegenerative Disorders"
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_LIGHT

    p2 = tf.add_paragraph()
    p2.text = "\nProduction Deployment & Breakthrough Results on NVIDIA Tesla V100 GPU Cluster"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(148, 163, 184)

    # Metadata card
    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(5.2), Inches(11.3), Inches(1.2))
    mtf = meta_box.text_frame
    mp = mtf.paragraphs[0]
    mp.text = "Author: basakanimesh14  |  Hardware: NVIDIA Tesla V100-SXM2-32GB (Volta SM 7.0)  |  Atlas: AAL-116"
    mp.font.size = Pt(12)
    mp.font.color.rgb = RGBColor(148, 163, 184)

    # -------------------------------------------------------------------------
    # SLIDE 2: Executive Summary & Breakthrough Highlights
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2, LIGHT_BG)
    add_header(s2, "Executive Summary: Clinical Framework & Benchmark Breakthroughs")

    # 4 Metric Cards
    cards_data = [
        ("Peak Val ROC-AUC", "0.8329", "Peak Val Macro-F1: 0.7652", ACCENT_BLUE),
        ("Held-Out Test ROC-AUC", "0.7461", "+3.34% boost over standard GNN", ACCENT_NAVY),
        ("Calibrated Test Accuracy", "73.33%", "+5.33% gain via Youden's J", ACCENT_GREEN),
        ("Calibrated Test Macro-F1", "0.7310", "+5.15% gain on unseen multi-site data", RGBColor(139, 92, 246))
    ]
    for i, (title, val, sub, col) in enumerate(cards_data):
        left = Inches(0.8 + i * 2.95)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), Inches(2.8), Inches(1.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        tf = card.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title.upper()
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED

        p_val = tf.add_paragraph()
        p_val.text = val
        p_val.font.size = Pt(28)
        p_val.font.bold = True
        p_val.font.color.rgb = col

        p_sub = tf.add_paragraph()
        p_sub.text = sub
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = TEXT_DARK

    # Narrative Content block
    desc_box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.7), Inches(11.7), Inches(3.2))
    desc_box.fill.solid()
    desc_box.fill.fore_color.rgb = CARD_BG
    desc_box.line.color.rgb = CARD_BORDER
    dtf = desc_box.text_frame
    dtf.word_wrap = True

    dp0 = dtf.paragraphs[0]
    dp0.text = "Key Project Achievements"
    dp0.font.size = Pt(14)
    dp0.font.bold = True
    dp0.font.color.rgb = TEXT_DARK

    bullets = [
        "End-to-End Clinical Pipeline: Integrates enterprise graph database (Neo4j 5.x) with modern deep geometric learning (PyG 2.8).",
        "Overcoming Euclidean Limitations: Models human brain functional connectomes natively as relational graphs (116 regions, 2.54M edges) rather than flattened matrices.",
        "The 3 Model Pillars: (1) Edge-weight awareness with Pearson r, (2) Dual global pooling [mean+max], and (3) SOTA SwiGLU gated activation.",
        "Verifiable Explainability: GPU-accelerated GNNExplainer extracts patient-specific biomarker subgraphs, uncovering consensus Cerebellar & Cortico-striatal circuits.",
        "Production-Grade Hardware Tuning: 100% user-space headless execution on Tesla V100 with FP16 mixed precision, PCIe pinned memory, and disk-level binary caching."
    ]
    for b in bullets:
        bp = dtf.add_paragraph()
        bp.text = "• " + b
        bp.font.size = Pt(11)
        bp.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 3: Clinical Problem & Connectome Dataset
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3, LIGHT_BG)
    add_header(s3, "Clinical Problem Formulation & Multi-Site Connectome Data")

    # Left Column: Biological & Clinical Motivation
    c_left = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = CARD_BG
    c_left.line.color.rgb = CARD_BORDER
    ltf = c_left.text_frame
    ltf.word_wrap = True

    lp0 = ltf.paragraphs[0]
    lp0.text = "The Biological Challenge"
    lp0.font.size = Pt(14)
    lp0.font.bold = True
    lp0.font.color.rgb = ACCENT_NAVY

    l_items = [
        "Pre-Symptomatic Disconnection: Neurodegenerative disorders (Alzheimer's, Parkinson's) alter distributed functional communication networks years before structural atrophy or neuron death appears.",
        "Why Traditional ML Fails: Standard CNNs and MLPs flatten 116x116 correlation matrices into 1D vectors (6,670 features), destroying topological invariants and spatial neighborhood geometry.",
        "Graph-Level Formulation: We model each patient as a whole brain graph G = (V, E, X) to diagnose systemic global network breakdown rather than isolated regional damage."
    ]
    for it in l_items:
        p = ltf.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # Right Column: Multi-Site Cohort Specifications
    c_right = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = CARD_BG
    c_right.line.color.rgb = CARD_BORDER
    rtf = c_right.text_frame
    rtf.word_wrap = True

    rp0 = rtf.paragraphs[0]
    rp0.text = "Dataset & Safeguards (AAL-116 Atlas)"
    rp0.font.size = Pt(14)
    rp0.font.bold = True
    rp0.font.color.rgb = ACCENT_NAVY

    r_items = [
        "Cohort Scale: 500 total subjects (250 Healthy Control, 250 Diseased).",
        "Anatomical Parcellation: AAL-116 atlas dividing cerebrum, subcortex, and cerebellum into 116 standardized regions of interest.",
        "Edge Filtering: Pearson correlation thresholded at |r| >= 0.3, producing 2,545,839 functional connectivity edges (~7-15% network density).",
        "Multi-Site Generalization: Stratified across 3 clinical scanner sites (site_A, site_B, site_C) with varying noise standard deviations (0.02, 0.04, 0.06).",
        "Zero Identity Leakage: Strict subject-level stratified splitting: 70% Train (350), 15% Validation (75), 15% Held-Out Test (75). No scan slices cross split boundaries."
    ]
    for it in r_items:
        p = rtf.add_paragraph()
        p.text = "• " + it
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 4: Hardware & Engineering Stack (Tesla V100 Optimization)
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4, LIGHT_BG)
    add_header(s4, "Engineering Stack: NVIDIA Tesla V100 Cluster Optimizations")

    eng_cards = [
        ("Volta SM 7.0 Precision", "FP16 Mixed Precision", "Volta lacks hardware BF16 tensor cores. Utilizes torch.float16 with torch.cuda.amp.GradScaler() to double throughput while avoiding software emulation bottlenecks.", ACCENT_BLUE),
        ("PCIe Memory Pinning", "pin_memory=True", "Dedicated 32GB HBM2 VRAM across high-speed PCIe bus host-to-device transfers. Pinned memory buffers ensure zero memory stall during DataLoader batching.", ACCENT_NAVY),
        ("Dataset Caching", "16 min -> 0.05 sec", "Pre-computed 2.54M edge graph tensors cached as disk-level binary (cached_dataset.pt). Slashes epoch startup latency down to 50 milliseconds.", ACCENT_GREEN),
        ("Decoupled Architecture", "Safe Server Design", "Tesla V100 GPU is strictly reserved for training and explainer passes. Production REST serving runs on standard CPU microservices via FastAPI.", RGBColor(139, 92, 246))
    ]
    for i, (title, val, desc, col) in enumerate(eng_cards):
        row = i // 2
        col_idx = i % 2
        left = Inches(0.8 + col_idx * 5.95)
        top = Inches(1.6 + row * 2.75)

        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        ctf = card.text_frame
        ctf.word_wrap = True

        p = ctf.paragraphs[0]
        p.text = title.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED

        pv = ctf.add_paragraph()
        pv.text = val
        pv.font.size = Pt(18)
        pv.font.bold = True
        pv.font.color.rgb = col

        pd = ctf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 5: Deep Learning Architecture (The 3 Breakthrough Pillars)
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5, LIGHT_BG)
    add_header(s5, "Deep Learning Architecture: BrainGNNModel with 3 Breakthrough Pillars")

    pillars = [
        ("PILLAR 1: Edge-Weight Awareness", "Incorporates Pearson r directly into message passing", 
         "Standard GNNs discard edge weights after thresholding. We feed Pearson correlation values directly as edge attributes into GCNConv(edge_weight=|r|) and GATv2Conv(edge_dim=1, edge_attr=r), allowing attention heads to weigh communicative strength.", ACCENT_BLUE),
        ("PILLAR 2: Dual Global Pooling", "[global_mean_pool || global_max_pool]", 
         "Mean pooling alone averages over 116 regions, diluting focal circuit degradation. Dual pooling concatenates mean background connectivity with extreme regional signal spikes into a unified 2*hidden_channels embedding.", ACCENT_NAVY),
        ("PILLAR 3: SwiGLU Gated Activation", "SOTA LLaMA 3 Feed-Forward Readout Head", 
         "Replaced traditional ReLU with Swish Gated Linear Units: SwiGLU(x) = (Swish(xW_gate) * (xW_up))W_down. Acts as a dynamic noise gate to filter scanner SNR variations across clinical centers.", ACCENT_GREEN)
    ]
    for i, (p_title, p_sub, p_desc, col) in enumerate(pillars):
        left = Inches(0.8 + i * 3.95)
        card = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.6), Inches(3.8), Inches(5.3))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        ctf = card.text_frame
        ctf.word_wrap = True

        p = ctf.paragraphs[0]
        p.text = p_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col

        ps = ctf.add_paragraph()
        ps.text = p_sub
        ps.font.size = Pt(11)
        ps.font.bold = True
        ps.font.color.rgb = TEXT_DARK

        pd = ctf.add_paragraph()
        pd.text = "\n" + p_desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_MUTED

    # -------------------------------------------------------------------------
    # SLIDE 6: Training Dynamics & Convergence (Plot 1)
    # -------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6, LIGHT_BG)
    add_header(s6, "Training Dynamics: High-Throughput Convergence on Tesla V100")

    # Embed Plot 1
    plot1 = plots_dir / "1_training_convergence_curves.png"
    if plot1.exists():
        s6.shapes.add_picture(str(plot1), Inches(0.8), Inches(1.5), width=Inches(11.7))

    # Add descriptive callout below
    note_box = s6.shapes.add_textbox(Inches(0.8), Inches(5.6), Inches(11.7), Inches(1.5))
    ntf = note_box.text_frame
    ntf.word_wrap = True
    np0 = ntf.paragraphs[0]
    np0.text = "Key Optimization Insights:"
    np0.font.bold = True
    np0.font.size = Pt(12)
    np0.font.color.rgb = ACCENT_NAVY

    n_bullets = [
        "Throughput: 1.33 seconds per epoch on single Tesla V100 GPU (~235,000 trainable parameters).",
        "Loss Convergence: Cross-entropy loss drops smoothly from 0.944 down to 0.630 with Cosine Annealing and linear warmup.",
        "Peak Validation Milestone: Reached 0.8329 Peak Validation ROC-AUC (Val Macro-F1: 0.7652) at Epoch 70 before early stopping."
    ]
    for nb in n_bullets:
        p = ntf.add_paragraph()
        p.text = "• " + nb
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 7: Held-Out Test Evaluation & Threshold Calibration (Plot 2)
    # -------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7, LIGHT_BG)
    add_header(s7, "Held-Out Test Generalization & Youden's J-Statistic Calibration")

    # Embed Plot 2 on Left
    plot2 = plots_dir / "2_confusion_matrix_calibrated.png"
    if plot2.exists():
        s7.shapes.add_picture(str(plot2), Inches(0.8), Inches(1.6), width=Inches(6.8))

    # Right Column: Analytical Breakdown
    r_box = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.3))
    r_box.fill.solid()
    r_box.fill.fore_color.rgb = CARD_BG
    r_box.line.color.rgb = CARD_BORDER
    rtf = r_box.text_frame
    rtf.word_wrap = True

    rp0 = rtf.paragraphs[0]
    rp0.text = "Resolving Multi-Site Threshold Bias"
    rp0.font.size = Pt(13)
    rp0.font.bold = True
    rp0.font.color.rgb = ACCENT_NAVY

    r_text = [
        "Unseen Test Cohort: 75 held-out subjects (38 Diseased, 37 Healthy) across all 3 scanner sites.",
        "Breakthrough Test ROC-AUC: Achieved 0.7461 on unseen data, representing true ranking capability across multi-scanner protocols.",
        "The Default Threshold Trap: Uncalibrated default p=0.5 yields 57.33% accuracy due to scanner noise shifts.",
        "Youden's J Calibration: Calibrated operating threshold t* = 0.1762 (Youden's J = 0.4644) on validation predictions.",
        "Direct Accuracy Surge: Realigns operational test accuracy to 73.33% and Macro-F1 to 0.7310 without requiring any model retraining."
    ]
    for rt in r_text:
        p = rtf.add_paragraph()
        p.text = "• " + rt
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 8: Comprehensive Model Ablation Study (Plot 3)
    # -------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8, LIGHT_BG)
    add_header(s8, "Architectural Ablation Study across Baselines & Model Generations")

    plot3 = plots_dir / "3_model_ablation_comparison.png"
    if plot3.exists():
        s8.shapes.add_picture(str(plot3), Inches(0.8), Inches(1.6), width=Inches(7.2))

    ab_box = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.2), Inches(1.6), Inches(4.3), Inches(5.3))
    ab_box.fill.solid()
    ab_box.fill.fore_color.rgb = CARD_BG
    ab_box.line.color.rgb = CARD_BORDER
    atf = ab_box.text_frame
    atf.word_wrap = True

    ap0 = atf.paragraphs[0]
    ap0.text = "Empirical Architecture Evolution"
    ap0.font.size = Pt(13)
    ap0.font.bold = True
    ap0.font.color.rgb = ACCENT_NAVY

    a_items = [
        "RBF-SVM (0.671 AUC): Flattening correlation matrices loses relational topology.",
        "RF-GDV (0.692 AUC): Handcrafted graphlet orbit counts capture motifs but miss attention.",
        "node2vec (0.635 AUC): Random walks without disease supervision fail subtle connectivity shifts.",
        "GATv1 Baseline (0.7127 AUC): Good ranking, but static attention causes ranking collapse.",
        "Neuro-Neo4j Full (0.7461 AUC, 73.33% Acc): Combines edge weights, dual pooling, and SwiGLU to set state-of-the-art benchmark."
    ]
    for ai in a_items:
        p = atf.add_paragraph()
        p.text = "• " + ai
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 9: Explainable AI & Biomarker Discoveries (Plot 4)
    # -------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s9, LIGHT_BG)
    add_header(s9, "Explainable AI: GNNExplainer Consensus Biomarker Circuits")

    plot4 = plots_dir / "4_biomarker_circuits.png"
    if plot4.exists():
        s9.shapes.add_picture(str(plot4), Inches(0.8), Inches(1.6), width=Inches(6.8))

    bio_box = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.3))
    bio_box.fill.solid()
    bio_box.fill.fore_color.rgb = CARD_BG
    bio_box.line.color.rgb = CARD_BORDER
    btf = bio_box.text_frame
    btf.word_wrap = True

    bp0 = btf.paragraphs[0]
    bp0.text = "Discovered Biological Circuits"
    bp0.font.size = Pt(13)
    bp0.font.bold = True
    bp0.font.color.rgb = ACCENT_NAVY

    b_items = [
        "GPU-Accelerated Explainer: Runs 500 gradient steps per test subject natively on CUDA to extract minimal causal subgraphs.",
        "Hippocampus (R/L) & Amygdala: 100% consensus hubs in diseased subjects, matching early memory and limbic breakdown in Alzheimer's.",
        "Cerebellar Vermis Network: Replicated Vermis_6 to Vermis_10 breakdown, implicating Cerebellar Cognitive Affective Syndrome.",
        "Key Disrupted Pathways:\n - Lingual_L <-> Putamen_L (Visual-striatal integration)\n - Hippocampus_L <-> SupraMarginal_R (Memory-attention)\n - Frontal_Med_Orb_L <-> Cerebelum_Crus1_L (Executive control)"
    ]
    for bi in b_items:
        p = btf.add_paragraph()
        p.text = "• " + bi
        p.font.size = Pt(10.5)
        p.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 10: Clinical Serving & Real-Time Monitoring
    # -------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s10, LIGHT_BG)
    add_header(s10, "Clinical Serving Microservice & Drift Monitoring")

    serv_cards = [
        ("FastAPI Inference Service", "Sub-50ms Predictions", "Lightweight CPU REST microservice decoupled from GPU training hardware. Provides clean endpoints for real-time clinical workflows: /predict, /explain, /health, /metrics.", ACCENT_BLUE),
        ("Prometheus Monitoring", "Observability & SLA", "Exposes standardized /metrics/prometheus endpoint tracking prediction latencies, throughput, HTTP status codes, and server CPU/memory usage.", ACCENT_NAVY),
        ("Sliding-Window Drift Detection", "Continuous Governance", "Monitors Macro-F1 across a sliding window of recent predictions. Flags statistically significant distribution shifts when F1 drops >5% below 0.80 baseline.", ACCENT_GREEN),
        ("Automated Retrain Trigger", "Closed-Loop Retraining", "Automatically emits retrain alerts after 3 consecutive degraded windows, queuing accumulated clinical scans for GPU fine-tuning.", RGBColor(139, 92, 246))
    ]
    for i, (title, val, desc, col) in enumerate(serv_cards):
        row = i // 2
        col_idx = i % 2
        left = Inches(0.8 + col_idx * 5.95)
        top = Inches(1.6 + row * 2.75)

        card = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, Inches(5.75), Inches(2.55))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        ctf = card.text_frame
        ctf.word_wrap = True

        p = ctf.paragraphs[0]
        p.text = title.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED

        pv = ctf.add_paragraph()
        pv.text = val
        pv.font.size = Pt(18)
        pv.font.bold = True
        pv.font.color.rgb = col

        pd = ctf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_DARK

    # -------------------------------------------------------------------------
    # SLIDE 11: Summary, Roadmap & Conclusion
    # -------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s11, DARK_BG)

    st_box = s11.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(0.8))
    stf = st_box.text_frame
    sp0 = stf.paragraphs[0]
    sp0.text = "CONCLUSION & FUTURE ROADMAP"
    sp0.font.size = Pt(24)
    sp0.font.bold = True
    sp0.font.color.rgb = ACCENT_BLUE

    # Left: Core Deliverables
    c_del = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.8), Inches(5.4), Inches(5.0))
    c_del.fill.solid()
    c_del.fill.fore_color.rgb = RGBColor(30, 41, 59)
    c_del.line.color.rgb = RGBColor(51, 65, 85)
    dtf = c_del.text_frame
    dtf.word_wrap = True

    dp0 = dtf.paragraphs[0]
    dp0.text = "What We Delivered"
    dp0.font.size = Pt(14)
    dp0.font.bold = True
    dp0.font.color.rgb = TEXT_LIGHT

    del_items = [
        "Engineered complete hybrid graph AI pipeline combining Neo4j GDS and PyTorch Geometric.",
        "Achieved 0.833 Peak Val AUC, 0.746 Held-Out Test AUC, and 73.33% Calibrated Accuracy on Tesla V100.",
        "Uncovered verifiable consensus neural biomarkers matching clinical Alzheimer's/Parkinson's literature.",
        "Productionized entire system with headless V100 optimization, FastAPI REST serving, and drift detection.",
        "100% verified test suite (10/10 dry-run pass) and full publication-grade visualization suite."
    ]
    for di in del_items:
        p = dtf.add_paragraph()
        p.text = "✓ " + di
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(203, 213, 225)

    # Right: Future Scope
    c_fut = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(1.8), Inches(5.4), Inches(5.0))
    c_fut.fill.solid()
    c_fut.fill.fore_color.rgb = RGBColor(30, 41, 59)
    c_fut.line.color.rgb = RGBColor(51, 65, 85)
    ftf = c_fut.text_frame
    ftf.word_wrap = True

    fp0 = ftf.paragraphs[0]
    fp0.text = "Next Technical Steps"
    fp0.font.size = Pt(14)
    fp0.font.bold = True
    fp0.font.color.rgb = ACCENT_NAVY

    fut_items = [
        "Real Clinical Ingestion: Ingest real ADNI (Alzheimer's) and PPMI (Parkinson's) neuroimaging cohorts.",
        "Multi-Modal Fusion: Combine resting-state fMRI functional connectomes with DTI structural tractography.",
        "Multi-GPU DDP Scaling: Scale batch training across 4-8 Tesla V100 GPU nodes using torchrun.",
        "Interactive 3D WebGL Dashboard: Render 3D brain connectomes and GNNExplainer subgraphs directly in the browser."
    ]
    for fi in fut_items:
        p = ftf.add_paragraph()
        p.text = "→ " + fi
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(203, 213, 225)

    # Save presentation
    output_path = Path("Neuro-Neo4j-Presentation.pptx")
    prs.save(str(output_path))
    print(f"[SUCCESS] PowerPoint generated: {output_path.resolve()}")

if __name__ == "__main__":
    create_presentation()
