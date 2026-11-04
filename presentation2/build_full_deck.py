import sys
import os
import re
from pathlib import Path
from lxml import etree

import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml import parse_xml

import latex2mathml.converter
import mathml2omml

WORKSPACE_ROOT = Path("/mnt/hdd/animesh/Neuro-Neo4j")
TEMPLATE_PATH = Path("/home/user/Desktop/Rohit_Raj_BTP/AntiGravity/report/format/PPT_Group-No_RS-1.pptx")
FIGURES_DIR = WORKSPACE_ROOT / "presentation2" / "figures"
OUTPUT_PPTX = WORKSPACE_ROOT / "Neuro_Neo4j_Presentation2.pptx"

# Colors
NAVY = RGBColor(31, 56, 100)          # #1F3864
BLUE_BORDER = RGBColor(42, 120, 214)    # #2A78D6 (Neutral)
GREEN_BORDER = RGBColor(46, 125, 50)    # #2E7D32 (Results/fixes)
RED_BORDER = RGBColor(192, 0, 0)        # #C00000 (Problems/limitations)
TEXT_BLACK = RGBColor(20, 20, 20)
TEXT_MUTED = RGBColor(90, 90, 90)
BG_WHITE = RGBColor(255, 255, 255)
BG_LIGHT_BLUE = RGBColor(220, 235, 250) # #DCEBFA
BG_LIGHT_GREEN = RGBColor(221, 242, 228)# #DDF2E4

def latex_to_omml_element(latex_str):
    clean = latex_str.strip()
    clean = clean.replace(r"\textstyle", "").replace(r"\!", "")
    clean = clean.replace(r"\Vert", "||").replace(r"\lVert", "||").replace(r"\rVert", "||")
    clean = clean.replace(r"\odot", r"\cdot")
    clean = re.sub(r"\\operatorname\{([^}]+)\}", r"\\mathrm{\1}", clean)
    
    mml = latex2mathml.converter.convert(clean)
    omml_str = mathml2omml.convert(mml)
    if "xmlns:m=" not in omml_str:
        omml_str = omml_str.replace("<m:oMath>", '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">', 1)
    
    wrapper = etree.fromstring(
        '<a14:m xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"/>'
    )
    omml_elem = etree.fromstring(omml_str)
    wrapper.append(omml_elem)
    return wrapper

def add_equation_paragraph(tf, latex_str, pt_size=13, align=PP_ALIGN.CENTER, space_after=3):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    pPr = p._p.get_or_add_pPr()
    pPr.set("sz", str(int(pt_size * 100)))
    try:
        elem = latex_to_omml_element(latex_str)
        p._p.append(elem)
    except Exception as e:
        print(f"Eq conversion failed for '{latex_str}': {e}")
        r = p.add_run()
        r.text = latex_str
        r.font.name = "Cambria Math"
        r.font.size = Pt(pt_size)
    return p

def add_content_box(slide, left, top, width, height, title, border_color, fill_color=BG_WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.75)
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.name = "Calibri"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = NAVY
    p0.alignment = PP_ALIGN.CENTER
    p0.space_after = Pt(4)
    
    return shape, tf

def add_bullet(tf, text, pt_size=11, bold=False, level=0, space_after=3):
    p = tf.add_paragraph()
    p.level = level
    p.space_after = Pt(space_after)
    p.font.name = "Calibri"
    
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if not part:
            continue
        r = p.add_run()
        r.font.name = "Calibri"
        r.font.size = Pt(pt_size)
        if part.startswith("**") and part.endswith("**"):
            r.text = part[2:-2]
            r.font.bold = True
            r.font.color.rgb = TEXT_BLACK
        else:
            r.text = part
            r.font.bold = bold
            r.font.color.rgb = TEXT_BLACK
    return p

def add_picture_with_svg(slide, png_path, svg_path, left, top, width, height=None):
    pic = slide.shapes.add_picture(str(png_path), left, top, width=width, height=height)
    if svg_path and Path(svg_path).exists():
        try:
            svg_data = open(str(svg_path), "rb").read()
            pkg = slide.part.package
            svg_name = Path(svg_path).name
            partname = PackURI(f"/ppt/media/svg_{svg_name}")
            svg_part = Part(partname, "image/svg+xml", blob=svg_data, package=pkg)
            rId_svg = slide.part.relate_to(svg_part, pptx.opc.constants.RELATIONSHIP_TYPE.IMAGE)
            
            blip = pic._element.xpath(".//a:blip")[0]
            svg_xml = (
                f'<a:extLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'  <a:ext uri="{{96DAC542-7818-4354-9E77-4C4F5DA47E80}}">'
                f'    <asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" r:embed="{rId_svg}"/>'
                f'  </a:ext>'
                f'</a:extLst>'
            )
            blip.append(parse_xml(svg_xml))
        except Exception as e:
            print(f"SVG blip warning ({svg_path}): {e}")
    return pic

def set_slide_title(slide, title_text):
    title_shape = slide.shapes[0]
    title_shape.text_frame.clear()
    p = title_shape.text_frame.paragraphs[0]
    p.text = title_text
    p.font.name = "Calibri"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0, 0, 0)
    p.alignment = PP_ALIGN.LEFT

def set_notes(slide, notes_script):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes_script.strip()

def build_presentation():
    print(f"Loading template from {TEMPLATE_PATH}...")
    prs = pptx.Presentation(str(TEMPLATE_PATH))
    
    # In template: slide 0 is Title, slides 1-11 are Content slides
    # Remove extra slides if present so we have exactly 11 slides (1 to 10 + Thank you)
    while len(prs.slides) > 11:
        sldId = prs.slides._sldIdLst[11]
        prs.slides._sldIdLst.remove(sldId)
    
    # Clear content placeholder shape[1] from slides 1..10
    for idx in range(1, len(prs.slides)):
        slide = prs.slides[idx]
        if len(slide.shapes) > 1:
            sp = slide.shapes[1]
            slide.shapes._spTree.remove(sp._element)

    # ----------------------------------------------------
    # SLIDE 1: Title
    # ----------------------------------------------------
    print("Building Slide 1...")
    s1 = prs.slides[0]
    s1.shapes[0].text_frame.clear()
    p0 = s1.shapes[0].text_frame.paragraphs[0]
    p0.text = "Explainable Graph Neural Networks & Hybrid Graph Databases for Brain Biomarker Discovery"
    p0.font.name = "Calibri"
    p0.font.size = Pt(24)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(0, 0, 0)
    
    p1 = s1.shapes[0].text_frame.add_paragraph()
    p1.text = "Part 2: Pipeline Execution and Results"
    p1.font.name = "Calibri"
    p1.font.size = Pt(17)
    p1.font.italic = True
    p1.font.bold = False
    p1.font.color.rgb = NAVY
    
    s1.shapes[2].text_frame.clear()
    p2 = s1.shapes[2].text_frame.paragraphs[0]
    p2.text = "Animesh Basak (26911001)\nDepartment of Computer Science & Engineering\nIndian Institute of Technology Roorkee"
    p2.font.name = "Calibri"
    p2.font.size = Pt(15)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(0, 0, 0)
    
    s1.shapes[1].text_frame.clear()
    p3 = s1.shapes[1].text_frame.paragraphs[0]
    p3.text = "M.Tech. / Ph.D. Course Project Presentation — November 2026"
    p3.font.name = "Calibri"
    p3.font.size = Pt(13)
    p3.font.color.rgb = TEXT_MUTED

    s1_notes = (
        "Good morning, committee members. In this second presentation, I present the execution and empirical validation "
        "of our end-to-end neuroimaging pipeline, Neuro-Neo4j. In Part 1, I introduced our theoretical concept: "
        "combining Neo4j property graphs with geometric deep learning to diagnose neurological disease directly "
        "from resting-state functional connectomes. Today, I walk through each pipeline stage as implemented on our "
        "Tesla V100 GPU cluster. I cover our synthetic cohort generation, per-subject signal cleaning, topological feature "
        "audits, hybrid GCN-GATv2 neural architecture, and post-hoc biomarker extraction using GNNExplainer. "
        "I will share both what worked and what surprised us. Specifically, our graph model achieves a test ROC-AUC "
        "of 0.746, gaining substantial discriminatory power over flat baselines, while uncovering vital structural insights "
        "about brain representation."
    )
    set_notes(s1, s1_notes)

    # ----------------------------------------------------
    # SLIDE 2: Recap, Problem, Idea and Initial Features
    # ----------------------------------------------------
    print("Building Slide 2...")
    s2 = prs.slides[1]
    set_slide_title(s2, "Recap, Problem, Idea and Initial Features")
    
    # Box A: Blue border (Left)
    _, tf2_a = add_content_box(s2, Inches(0.25), Inches(1.22), Inches(4.55), Inches(5.45),
                               "A. Recap: Problem Statement & Core Concept", BLUE_BORDER)
    add_bullet(tf2_a, "Functional disconnection in resting-state fMRI appears years before structural atrophy; goal is early detection from connectome alone.", pt_size=11)
    add_bullet(tf2_a, "Task: Graph-level binary classification into Healthy (0) vs Neurodegenerative (1):", pt_size=11)
    add_equation_paragraph(tf2_a, r"P(y_k=1\mid G_k)=\mathrm{softmax}(f_\theta(A_k,X_k))_1", pt_size=12)
    add_bullet(tf2_a, "Plus extraction of a post-hoc biological explanation subgraph $G_s \\subseteq G_k$.", pt_size=11)
    add_bullet(tf2_a, "Each subject brain is a graph $G_k=(V,E_k,X_k)$ on the AAL-116 atlas with $|V|=116$ anatomical regions and thresholded correlation edges.", pt_size=11)
    add_bullet(tf2_a, "Core architectural pipeline from Deck 1: Edge-weight-aware GCN $\\to$ GATv2 dynamic attention with continuous correlation attributes $\\to$ Dual pooling [mean, max] $\\to$ SwiGLU classification head $\\to$ GNNExplainer.", pt_size=11)

    # Box B: Green border (Right)
    _, tf2_b = add_content_box(s2, Inches(5.0), Inches(1.22), Inches(4.55), Inches(5.45),
                               "B. Initial Features Selected & Engineering", GREEN_BORDER)
    add_bullet(tf2_b, "**Topological Descriptors Computed** (5 node + 1 edge):", pt_size=11, bold=True)
    
    t_shape2 = s2.shapes.add_table(7, 3, Inches(5.1), Inches(1.75), Inches(4.35), Inches(2.75))
    tbl2 = t_shape2.table
    tbl2.columns[0].width = Inches(1.0)
    tbl2.columns[1].width = Inches(1.3)
    tbl2.columns[2].width = Inches(2.05)
    
    headers2 = ["Level", "Feature", "How it is computed"]
    for c_idx, h in enumerate(headers2):
        cell = tbl2.cell(0, c_idx)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"; p.font.size = Pt(9.5); p.font.bold = True
        p.font.color.rgb = BG_WHITE; p.alignment = PP_ALIGN.CENTER
        
    rows2 = [
        ("Node (5-dim)", "Degree", "k_i / max_j k_j (binary neighbor ratio)"),
        ("Node", "PageRank", "Power iteration on |z_ij|, damping 0.85"),
        ("Node", "Clustering", "Weighted geometric-mean triangle intensity"),
        ("Node", "Betweenness", "Normalised shortest-path betweenness"),
        ("Node", "Eigenvector", "Leading eigenvector of |W|, max-scaled"),
        ("Edge (1-dim)", "z_ij", "Signed z-score (GATv2); |z_ij| for GCN"),
    ]
    for r_idx, row in enumerate(rows2):
        for c_idx, val in enumerate(row):
            cell = tbl2.cell(r_idx + 1, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"; p.font.size = Pt(8.5)
            if c_idx == 0: p.font.bold = True
            
    tb_sub2 = s2.shapes.add_textbox(Inches(5.05), Inches(4.6), Inches(4.45), Inches(1.95))
    tf2_sub = tb_sub2.text_frame
    tf2_sub.word_wrap = True
    add_bullet(tf2_sub, "**Initial Engineering Flow**:", pt_size=10.5, bold=True)
    add_bullet(tf2_sub, "Global-signal removal $\\to$ z-scoring off-diagonal entries $\\to$ pruning $|z_{ij}| < 0.3$ $\\to$ symmetric COO edge list $\\to$ NetworkX / Neo4j-GDS centralities.", pt_size=10)
    add_bullet(tf2_sub, "**Critical Threshold Clarification**: In Deck 1, threshold was stated as $|r_{ij}| \\ge 0.3$. In our actual code, it is applied **after** z-scoring: $|z_{ij}| \\ge 0.3$. This keeps ~76% density (5,092 edges per brain).", pt_size=10)

    s2_notes = (
        "This slide recaps our core clinical formulation and initial feature representations. Our objective is early "
        "graph-level binary classification between healthy controls and neurodegenerative disease subjects on the "
        "116-region AAL atlas. Functional disconnection emerges years before visible structural atrophy, so we target "
        "connectomes directly. Our initial setup computes five node-level topological metrics—normalized degree, PageRank, "
        "clustering coefficient, betweenness, and eigenvector centrality—alongside continuous edge weights derived from "
        "z-scored Pearson correlations. Deck 1 specified an edge threshold of correlation magnitude at least 0.3. "
        "In our actual implementation, this threshold is applied after z-scoring off-diagonal entries. This critical "
        "distinction preserves roughly 76% of all possible edges, maintaining dense cross-region paths for graph "
        "convolution rather than producing an overly sparse graph."
    )
    set_notes(s2, s2_notes)

    # ----------------------------------------------------
    # SLIDE 3: Pipeline Step 1, Data Collection
    # ----------------------------------------------------
    print("Building Slide 3...")
    s3 = prs.slides[2]
    set_slide_title(s3, "Pipeline Step 1: Data Collection & Cohort Generation")
    
    # Box A: Blue border (Left)
    _, tf3_a = add_content_box(s3, Inches(0.25), Inches(1.22), Inches(4.55), Inches(5.45),
                               "A. Cohort Selection Rationale", BLUE_BORDER)
    add_bullet(tf3_a, "Clinical cohorts (ADNI, PPMI) require signed institutional data-use agreements that remain pending approval.", pt_size=11)
    add_bullet(tf3_a, "To build, debug, and validate the full deep geometric pipeline end-to-end, we generated a **synthetic AAL-116 cohort with a planted, mathematically verifiable disease signature**.", pt_size=11)
    add_bullet(tf3_a, "**Cohort Scale**: $N=500$ subjects (250 Healthy, 250 Diseased, strictly alternating by index), ages 50–84.", pt_size=11)
    add_bullet(tf3_a, "**Multi-Scanner Variation**: Assigned round-robin across 3 imaging sites (Site A, Site B, Site C).", pt_size=11)
    add_bullet(tf3_a, "**The Planted Ground-Truth Advantage**: Because the disease signature is explicitly engineered into the connectome, we can mathematically evaluate **explanation faithfulness** against ground truth. Real patient data cannot provide this objective baseline.", pt_size=11)

    # Box B: Green border (Right Upper)
    _, tf3_b = add_content_box(s3, Inches(5.0), Inches(1.22), Inches(4.55), Inches(3.6),
                               "B. Generative Model (`generate_synthetic.py`)", GREEN_BORDER)
    add_bullet(tf3_b, "**Base Functional Covariance & Correlation**:", pt_size=10.5, bold=True)
    add_equation_paragraph(tf3_b, r"\Sigma = BB^\top + nI,\quad B_{ij}\sim\mathcal N(0,1),\quad R_{ij}=\frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\Sigma_{jj}}}", pt_size=11.5)
    
    add_bullet(tf3_b, "**Planted Disease Perturbation** (Diseased subjects only):", pt_size=10.5, bold=True)
    add_equation_paragraph(tf3_b, r"R_{i,i+1}\leftarrow 0.4\,R_{i,i+1}\quad (\text{45 homotopic L–R pairs})", pt_size=11)
    add_equation_paragraph(tf3_b, r"R_{ij}\leftarrow \min(1,\,1.3\,R_{ij})\quad (i,j\text{ in same anatomical lobe})", pt_size=11)
    
    add_bullet(tf3_b, "**Scanner Site Noise Injection**:", pt_size=10.5, bold=True)
    add_equation_paragraph(tf3_b, r"R\leftarrow \mathrm{clip}\left(R+\tfrac12(\varepsilon+\varepsilon^\top),-1,1\right),\ \varepsilon_{ij}\sim\mathcal N(0,\sigma_s^2)", pt_size=11)

    # Table: Site noise distribution (Right Lower)
    t_shape3 = s3.shapes.add_table(4, 3, Inches(5.0), Inches(4.95), Inches(4.55), Inches(1.65))
    tbl3 = t_shape3.table
    tbl3.columns[0].width = Inches(1.3)
    tbl3.columns[1].width = Inches(1.75)
    tbl3.columns[2].width = Inches(1.5)
    
    headers3 = ["Scanner Site", "Noise Level (σ_s)", "Subject Count"]
    for c_idx, h in enumerate(headers3):
        cell = tbl3.cell(0, c_idx)
        cell.text = h
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"; p.font.size = Pt(10); p.font.bold = True
        p.font.color.rgb = BG_WHITE; p.alignment = PP_ALIGN.CENTER
        
    rows3 = [
        ("Site A", "0.02 (Low noise)", "167 subjects"),
        ("Site B", "0.04 (Moderate)", "167 subjects"),
        ("Site C", "0.06 (High noise)", "166 subjects"),
    ]
    for r_idx, row in enumerate(rows3):
        for c_idx, val in enumerate(row):
            cell = tbl3.cell(r_idx + 1, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"; p.font.size = Pt(9.5)
            p.alignment = PP_ALIGN.CENTER

    s3_notes = (
        "Here we detail our data collection step. Real clinical cohorts from ADNI and PPMI require institutional "
        "agreements that remain pending approval. To develop, test, and benchmark the entire pipeline end-to-end, "
        "we generated a synthetic cohort of 500 subjects with a planted, mathematically known disease pattern. "
        "Each brain has 116 regions, with 250 healthy and 250 diseased subjects across three scanner sites. "
        "Diseased brains receive two perturbations: a 60% attenuation across 45 homotopic left-right connections, "
        "and a 30% boost in within-lobe correlations. Scanner noise simulates multi-site acquisition variance. "
        "Because this disease pattern is planted, we can directly evaluate explanation fidelity against ground truth. "
        "Likely question: Why use synthetic data instead of clinical scans? Answer: Clinical agreements are pending, "
        "and planted signatures allow exact mathematical auditing of model explanations."
    )
    set_notes(s3, s3_notes)

    # ----------------------------------------------------
    # SLIDE 4: Pipeline Step 2, Data Cleaning and Preprocessing
    # ----------------------------------------------------
    print("Building Slide 4...")
    s4 = prs.slides[3]
    set_slide_title(s4, "Pipeline Step 2: Cleaning, Preprocessing & Leakage Control")
    
    # Box A: Blue border (Upper Left)
    _, tf4_a = add_content_box(s4, Inches(0.25), Inches(1.22), Inches(4.55), Inches(2.55),
                               "A. Four Transforms per Subject (`transform.py`)", BLUE_BORDER)
    add_bullet(tf4_a, "1. **Global Signal Removal**: Subtract mean off-diagonal correlation, diagonal reset to 1:", pt_size=10.5)
    add_equation_paragraph(tf4_a, r"\tilde R = R-\mu_g\mathbf 1\mathbf 1^\top,\quad \mu_g=\frac{1}{n(n-1)}\sum_{i\ne j}R_{ij}", pt_size=10.5, space_after=1)
    add_bullet(tf4_a, "2. **Per-Subject Z-Scoring**: Zero-mean, unit-variance on off-diagonals; diagonal set to 0:", pt_size=10.5)
    add_equation_paragraph(tf4_a, r"z_{ij}=\frac{\tilde R_{ij}-\mu}{\sigma},\quad W_{ij}=z_{ij}\cdot\mathbb 1[|z_{ij}|\ge 0.3]", pt_size=10.5, space_after=1)
    add_bullet(tf4_a, "3. **Sparsification & Edge Export**: Upper triangle $i<j, W_{ij}\\ne 0 \\to$ `edge_list.csv` (2,545,839 rows) loaded into Neo4j via `LOAD CSV`.", pt_size=10)

    # Box B: Green border (Upper Right)
    _, tf4_b = add_content_box(s4, Inches(5.0), Inches(1.22), Inches(4.55), Inches(2.55),
                               "B. Leakage Control Protocols", GREEN_BORDER)
    add_bullet(tf4_b, "**Stratified Partitioning** (Seed 42): 500 subjects partitioned into **350 Train / 75 Val / 75 Test** (70% / 15% / 15%).", pt_size=10.5)
    add_bullet(tf4_b, "**Exact Class Preservation**: 50/50 Healthy-to-Diseased balance maintained in every individual split.", pt_size=10.5)
    add_bullet(tf4_b, "**Patient-ID Partitioning**: Splits are grouped strictly by `PatientID`. Zero subject overlap across train, val, and test partitions (verified via programmatic assertion).", pt_size=10.5)
    add_bullet(tf4_b, "**Zero Splicing Leakage**: Preprocessing transforms (steps 1–3) are computed **strictly per subject**. No test distribution metrics ever leak into training.", pt_size=10.5)

    # FLOWCHART 1: Lower half
    dash_box = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.6), Inches(4.0), Inches(4.3), Inches(1.8))
    dash_box.fill.background()
    dash_box.line.color.rgb = NAVY
    dash_box.line.width = Pt(1.5)
    dash_box.line.dash_style = pptx.enum.dml.MSO_LINE.DASH
    
    dash_lbl = s4.shapes.add_textbox(Inches(1.65), Inches(3.82), Inches(4.2), Inches(0.3))
    p_dl = dash_lbl.text_frame.paragraphs[0]
    p_dl.text = "Per-Subject Transforms (Zero Leakage Across Cohort)"
    p_dl.font.name = "Calibri"; p_dl.font.size = Pt(9.5); p_dl.font.bold = True
    p_dl.font.color.rgb = NAVY; p_dl.alignment = PP_ALIGN.CENTER
    
    fc1_data = [
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.25), Inches(4.25), Inches(1.2), Inches(1.15),
         "Synthetic\nCohort", "500 × (116×116) R\n3 Scanner Sites", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.7), Inches(4.25), Inches(1.25), Inches(1.15),
         "Global Signal\nRemoval", "R - μ_g·11ᵀ\nDiag reset to 1", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.15), Inches(4.25), Inches(1.25), Inches(1.15),
         "Per-Subject\nZ-Scoring", "Off-diagonals\nμ=0, σ=1", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.6), Inches(4.25), Inches(1.2), Inches(1.15),
         "Sparsify\n|z| ≥ 0.3", "5,092 edges\nkept / subject", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.CAN, Inches(6.05), Inches(4.15), Inches(1.1), Inches(1.3),
         "Neo4j\nGraph DB", "(:BrainRegion)\n-[:CONNECTED]->", BG_LIGHT_GREEN, GREEN_BORDER),
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.35), Inches(4.25), Inches(1.15), Inches(1.15),
         "GDS / NX\nMetrics", "5 Node Metrics\n1 Edge Attribute", BG_LIGHT_GREEN, GREEN_BORDER),
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.65), Inches(4.25), Inches(1.1), Inches(1.15),
         "PyG Data\nObjects", "x ∈ R^{116×5}\ne ∈ R^{E×1}", BG_LIGHT_GREEN, GREEN_BORDER),
    ]
    
    for sh_type, left, top, w, h, t1, t2, fill, stroke in fc1_data:
        sh = s4.shapes.add_shape(sh_type, left, top, w, h)
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
        sh.line.color.rgb = stroke; sh.line.width = Pt(1.5)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.06); tf.margin_bottom = Inches(0.06)
        tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
        p1 = tf.paragraphs[0]
        p1.text = t1; p1.font.name = "Calibri"; p1.font.size = Pt(9.5); p1.font.bold = True
        p1.font.color.rgb = NAVY; p1.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = t2; p2.font.name = "Calibri"; p2.font.size = Pt(8); p2.font.color.rgb = TEXT_BLACK
        p2.alignment = PP_ALIGN.CENTER

    arrows_fc1 = [
        (Inches(1.45), Inches(4.75), Inches(0.25), Inches(0.16)),
        (Inches(2.95), Inches(4.75), Inches(0.2), Inches(0.16)),
        (Inches(4.4), Inches(4.75), Inches(0.2), Inches(0.16)),
        (Inches(5.8), Inches(4.75), Inches(0.25), Inches(0.16)),
        (Inches(7.15), Inches(4.75), Inches(0.2), Inches(0.16)),
        (Inches(8.5), Inches(4.75), Inches(0.15), Inches(0.16)),
    ]
    for a_l, a_t, a_w, a_h in arrows_fc1:
        arr = s4.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, a_l, a_t, a_w, a_h)
        arr.fill.solid(); arr.fill.fore_color.rgb = NAVY
        arr.line.fill.background()

    s_banner = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(5.9), Inches(2.95), Inches(0.75))
    s_banner.fill.solid(); s_banner.fill.fore_color.rgb = NAVY
    s_banner.line.fill.background()
    tf_sb = s_banner.text_frame
    p_sb = tf_sb.paragraphs[0]
    p_sb.text = "Stratified Cohort Partitioning (by PatientID)"
    p_sb.font.name = "Calibri"; p_sb.font.size = Pt(9.5); p_sb.font.bold = True
    p_sb.font.color.rgb = BG_WHITE; p_sb.alignment = PP_ALIGN.CENTER
    p_sb2 = tf_sb.add_paragraph()
    p_sb2.text = "350 Train (70%)  |  75 Val (15%)  |  75 Test (15%)"
    p_sb2.font.name = "Calibri"; p_sb2.font.size = Pt(8.5); p_sb2.font.color.rgb = BG_LIGHT_BLUE
    p_sb2.alignment = PP_ALIGN.CENTER

    s4_notes = (
        "Slide 4 illustrates our preprocessing pipeline and strict leakage prevention protocols. For every subject, "
        "we apply four sequential operations: global signal removal by subtracting mean off-diagonal correlation, "
        "subject-specific z-scoring, thresholding at absolute z greater than or equal to 0.3, and upper-triangular edge "
        "export into Neo4j. Over 2.5 million relationships were loaded into Neo4j to compute graph metrics. Critically, "
        "to prevent data snooping, all cleaning statistics are computed independently per subject rather than across the "
        "whole cohort. Our split partitions 500 subjects into 350 training, 75 validation, and 75 test brains using "
        "stratified sampling on patient ID. We verified programmatically that zero subject overlap exists across "
        "partitions. No test information ever leaks into training distributions."
    )
    set_notes(s4, s4_notes)

    # ----------------------------------------------------
    # SLIDE 5: Pipeline Step 3, Exploratory Data Analysis
    # ----------------------------------------------------
    print("Building Slide 5...")
    s5 = prs.slides[4]
    set_slide_title(s5, "Pipeline Step 3: Exploratory Data Analysis")
    
    # Box A: Blue border (Upper Left)
    _, tf5_a = add_content_box(s5, Inches(0.25), Inches(1.22), Inches(4.55), Inches(2.2),
                               "A. Connectome Graph Statistics", BLUE_BORDER)
    add_bullet(tf5_a, "**Edge Density Retention**: Retained edges per brain: **5,092 ± 34** out of 6,670 possible pairs (**76.3% density**).", pt_size=11)
    add_bullet(tf5_a, "**Class Inseparability**: Edge count distributions between Healthy and Diseased overlap almost completely (Fig 1, right).", pt_size=11)
    add_bullet(tf5_a, "**Weight Symmetry**: Mean weight ≈ 0, SD 1.14, exactly 50.0% negative. First-order distributions are completely indistinguishable by eye (Fig 1, left).", pt_size=11)

    # Box B: Red border (Lower Left)
    _, tf5_b = add_content_box(s5, Inches(0.25), Inches(3.55), Inches(4.55), Inches(3.1),
                               "B. Where the Class Signal Actually Lives", RED_BORDER)
    add_bullet(tf5_b, "**Group-Difference Matrix** (Diseased vs Healthy, Fig 3):", pt_size=10.5, bold=True)
    add_equation_paragraph(tf5_b, r"\Delta_{ij}=\overline{|W_{ij}|}_{D}-\overline{|W_{ij}|}_{H}", pt_size=11, space_after=1)
    add_bullet(tf5_b, "**Planted Signature Exposed**: Red positive blocks appear across all intra-lobe diagonals; a sharp blue negative line marks 45 homotopic L–R pairs.", pt_size=10.5)
    add_bullet(tf5_b, "**Single-Scalar Oracle Metric**:", pt_size=10.5, bold=True)
    add_equation_paragraph(tf5_b, r"s_k=\mathrm{mean}_{(i,j)\in\mathrm{intra-lobe}}|W_{ij}|\implies \mathbf{AUC = 1.000}", pt_size=11, space_after=1)
    add_bullet(tf5_b, "**Key Modeller Insight**: The signal is a **block-level magnitude pattern that depends entirely on region identity**. A model must know anatomical lobe groupings to directly exploit it.", pt_size=10.5)

    # Visuals: Fig 3 (Right)
    fig3_png = FIGURES_DIR / "fig3_group_difference_matrix.png"
    fig3_svg = FIGURES_DIR / "fig3_group_difference_matrix.svg"
    add_picture_with_svg(s5, fig3_png, fig3_svg, Inches(5.0), Inches(1.22), width=Inches(4.55))

    s5_notes = (
        "Slide 5 presents our exploratory data analysis, revealing where the classification signal truly resides. "
        "The two classes appear virtually indistinguishable from first-order summary statistics: each subject retains "
        "approximately 5,092 edges, and weight distributions mirror each other with zero mean. However, examining "
        "the group-difference matrix in Figure 3 exposes the planted pathology: strong positive blocks along within-lobe "
        "diagonals and negative homotopic pairs. In fact, computing a single summary scalar—the mean absolute "
        "intra-lobe weight—achieves a perfect test AUC of 1.000. This proves that our disease signature is a structured, "
        "block-level magnitude pattern tied to anatomical region identity. To exploit this signal, a machine learning "
        "model must either know region assignments or learn them through message passing."
    )
    set_notes(s5, s5_notes)

    # ----------------------------------------------------
    # SLIDE 6: Pipeline Step 4, Feature Engineering and Selection
    # ----------------------------------------------------
    print("Building Slide 6...")
    s6 = prs.slides[5]
    set_slide_title(s6, "Pipeline Step 4: Feature Engineering & Selection Audit")
    
    # Box A: Blue border (Left)
    _, tf6_a = add_content_box(s6, Inches(0.25), Inches(1.22), Inches(4.55), Inches(5.45),
                               "A. Node Descriptors (w_ij = |W_ij|, k_i = |N(i)|)", BLUE_BORDER)
    add_bullet(tf6_a, "**1. Degree Centrality & PageRank Centrality**:", pt_size=11, bold=True)
    add_equation_paragraph(tf6_a, r"x_i^{\deg}=\frac{k_i}{\max_j k_j},\quad \pi = \frac{1-d}{n}\mathbf 1 + d\,P^\top\pi,\quad P_{ij}=\frac{w_{ij}}{\sum_l w_{il}},\ d=0.85", pt_size=11)
    
    add_bullet(tf6_a, "**2. Weighted Clustering Coefficient (Onnela)**:", pt_size=11, bold=True)
    add_equation_paragraph(tf6_a, r"C_i=\frac{1}{k_i(k_i-1)}\sum_{j,h}\big(\hat w_{ij}\hat w_{ih}\hat w_{jh}\big)^{1/3}", pt_size=11)
    
    add_bullet(tf6_a, "**3. Shortest-Path Betweenness Centrality**:", pt_size=11, bold=True)
    add_equation_paragraph(tf6_a, r"b_i=\frac{2}{(n-1)(n-2)}\sum_{s\ne i\ne t}\frac{\sigma_{st}(i)}{\sigma_{st}}", pt_size=11)
    
    add_bullet(tf6_a, "**4. Eigenvector Centrality (Dominant Eigenmode)**:", pt_size=11, bold=True)
    add_equation_paragraph(tf6_a, r"\lambda_1 e = We,\quad x_i^{\text{eig}} = e_i / \max_j e_j", pt_size=11)
    
    add_bullet(tf6_a, "Descriptors measure topological density, hub prominence, shortest-path communication, and spectral influence across the functional network.", pt_size=10.5)

    # Box B: Red border (Right)
    _, tf6_b = add_content_box(s6, Inches(5.0), Inches(1.22), Inches(4.55), Inches(5.45),
                               "B. Selection Audit: Scale-Invariance & Distance Inversion", RED_BORDER)
    add_bullet(tf6_b, "**The Scale-Free Feature Dilemma**:", pt_size=11.5, bold=True)
    add_bullet(tf6_b, "All 5 node metrics are mathematically **scale-free**: each is divided by a maximum, sums to unity, or computes ratios of path counts.", pt_size=11)
    add_bullet(tf6_b, "Multiplying within-lobe connection weights by 1.3 barely alters these normalized ratios. Consequently, the planted disease magnitude is mostly **stripped away before the GNN ever sees it**.", pt_size=11)
    
    add_bullet(tf6_b, "**NetworkX Betweenness Distance Inversion**:", pt_size=11.5, bold=True)
    add_bullet(tf6_b, "NetworkX interprets edge `weight` as a **distance metric** in Dijkstra shortest-path calculations. High functional correlation is treated as a long path, inverting its biological significance.", pt_size=11)
    
    add_bullet(tf6_b, "**What Actually Carries the Classification Signal**:", pt_size=11.5, bold=True)
    add_bullet(tf6_b, "1. The continuous edge attribute $z_{ij}$ retained in GCN and GATv2 layers.", pt_size=11)
    add_bullet(tf6_b, "2. Anatomical region identity (missing in node features; explored on Slide 10).", pt_size=11)

    s6_notes = (
        "Here we audit our five selected node features: degree, PageRank, clustering coefficient, betweenness, and "
        "eigenvector centrality. While standard in network neuroscience, our mathematical audit reveals a significant "
        "limitation: all five descriptors are scale-free. They rely on ratios, max-normalizations, or probability "
        "vectors. When disease scales intra-lobe weights by 1.3, normalized node statistics barely budge. "
        "Consequently, our feature engineering accidentally discards the planted disease magnitude before the graph "
        "neural network ever receives it. Furthermore, NetworkX betweenness treats weights as travel distances, "
        "inverting strong functional couplings into distant paths. Only continuous edge weights, preserved as GATv2 edge "
        "attributes, retain the disease signal. This audit explains our architectural reliance on edge-aware message passing."
    )
    set_notes(s6, s6_notes)

    # ----------------------------------------------------
    # SLIDE 7: Pipeline Step 5, Model Selection
    # ----------------------------------------------------
    print("Building Slide 7...")
    s7 = prs.slides[6]
    set_slide_title(s7, "Pipeline Step 5: Model Selection & Architecture")
    
    # Flowchart 2: Left column stack
    fc2_boxes = [
        (MSO_SHAPE.PARALLELOGRAM, Inches(0.25), Inches(1.22), Inches(3.2), Inches(0.55),
         "Input Graph Representation", "x ∈ R^{116×5}, z_ij ∈ R^{E×1}", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.RECTANGLE, Inches(0.25), Inches(1.95), Inches(3.2), Inches(0.55),
         "Linear Input Projection", "5 → 192, SiLU, Dropout 0.25", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.RECTANGLE, Inches(0.25), Inches(2.68), Inches(3.2), Inches(0.65),
         "Edge-Aware GCNConv × 3", "BN + SiLU + Drop 0.25 + Residual (×3)", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.RECTANGLE, Inches(0.25), Inches(3.51), Inches(3.2), Inches(0.65),
         "GATv2Conv (4 heads × 48)", "Edge attr z_ij + BN + SiLU + Residual", BG_LIGHT_BLUE, BLUE_BORDER),
        (MSO_SHAPE.RECTANGLE, Inches(0.25), Inches(4.34), Inches(3.2), Inches(0.55),
         "Dual Global Pooling", "Mean Pool ‖ Max Pool → R^{384}", BG_LIGHT_GREEN, GREEN_BORDER),
        (MSO_SHAPE.RECTANGLE, Inches(0.25), Inches(5.07), Inches(3.2), Inches(0.55),
         "SwiGLU Gating Head", "384 → 384 → 96 Dimensional Gate", BG_LIGHT_GREEN, GREEN_BORDER),
        (MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.25), Inches(5.8), Inches(3.2), Inches(0.55),
         "Classifier Output", "Dropout 0.25 → Linear 96→2 → Softmax", NAVY, NAVY),
    ]
    
    for sh_type, left, top, w, h, t1, t2, fill, stroke in fc2_boxes:
        sh = s7.shapes.add_shape(sh_type, left, top, w, h)
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
        sh.line.color.rgb = stroke; sh.line.width = Pt(1.5)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.margin_top = Inches(0.04); tf.margin_bottom = Inches(0.04)
        p1 = tf.paragraphs[0]
        p1.text = t1; p1.font.name = "Calibri"; p1.font.size = Pt(9.5); p1.font.bold = True
        p1.font.color.rgb = BG_WHITE if fill == NAVY else NAVY
        p1.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.text = t2; p2.font.name = "Calibri"; p2.font.size = Pt(8)
        p2.font.color.rgb = BG_LIGHT_BLUE if fill == NAVY else TEXT_BLACK
        p2.alignment = PP_ALIGN.CENTER

    for arr_y in [Inches(1.77), Inches(2.50), Inches(3.33), Inches(4.16), Inches(4.89), Inches(5.62)]:
        arr = s7.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(1.75), arr_y, Inches(0.2), Inches(0.18))
        arr.fill.solid(); arr.fill.fore_color.rgb = NAVY
        arr.line.fill.background()

    badge = s7.shapes.add_shape(MSO_SHAPE.OVAL, Inches(3.1), Inches(2.72), Inches(0.45), Inches(0.45))
    badge.fill.solid(); badge.fill.fore_color.rgb = NAVY
    badge.line.fill.background()
    p_b = badge.text_frame.paragraphs[0]
    p_b.text = "×3"; p_b.font.name = "Calibri"; p_b.font.size = Pt(9); p_b.font.bold = True
    p_b.font.color.rgb = BG_WHITE; p_b.alignment = PP_ALIGN.CENTER

    # Box A: Blue border (Right Upper)
    _, tf7_a = add_content_box(s7, Inches(3.7), Inches(1.22), Inches(5.85), Inches(3.3),
                               "A. Mathematical Architecture Formulation (521,378 Params, d=192)", BLUE_BORDER)
    add_bullet(tf7_a, "**Input Projection & Edge-Weighted GCN Layers (×3 with Residual)**:", pt_size=10, bold=True)
    add_equation_paragraph(tf7_a, r"h_i^{(0)}=\mathrm{SiLU}(W_{\text{in}}x_i+b_{\text{in}})\in\mathbb R^{192}", pt_size=10.5, space_after=1)
    add_equation_paragraph(tf7_a, r"h_i^{(l+1)}=h_i^{(l)}+\mathrm{Drop}\Big(\mathrm{SiLU}\big(\mathrm{BN}\big(\sum_{j\in\mathcal N(i)\cup\{i\}}\frac{\tilde w_{ij}}{\sqrt{\tilde d_i\tilde d_j}}\,\Theta^{(l)}h_j^{(l)}\big)\big)\Big),\ \tilde d_i=1+\sum_j w_{ij}", pt_size=10, space_after=2)
    
    add_bullet(tf7_a, "**GATv2 Dynamic Attention Layer (4 heads × 48, Edge Dim = 1)**:", pt_size=10, bold=True)
    add_equation_paragraph(tf7_a, r"e_{ij}^{k}=a_k^\top\mathrm{LeakyReLU}_{0.2}(W_s^k h_i+W_t^k h_j+W_e^k z_{ij}),\ \alpha_{ij}^{k}=\frac{\exp e_{ij}^{k}}{\sum_{l\in\mathcal N(i)}\exp e_{il}^{k}}", pt_size=10, space_after=1)
    add_equation_paragraph(tf7_a, r"h_i'=h_i+\mathrm{Drop}\Big(\mathrm{SiLU}\big(\mathrm{BN}\big(\Vert_{k=1}^{4}\sum_{j}\alpha_{ij}^{k}W_t^k h_j\big)\big)\Big)", pt_size=10, space_after=2)
    
    add_bullet(tf7_a, "**Dual Pooling Readout & SwiGLU Gating Head**:", pt_size=10, bold=True)
    add_equation_paragraph(tf7_a, r"h_G=\big[\tfrac{1}{116}\sum_i h_i'\ \Vert\ \max_i h_i'\big]\in\mathbb R^{384},\ u=(\mathrm{SiLU}(h_GW_g)\odot h_GW_u)W_d\in\mathbb R^{96}", pt_size=10, space_after=1)

    # Box B: Green border (Right Lower)
    _, tf7_b = add_content_box(s7, Inches(3.7), Inches(4.65), Inches(5.85), Inches(2.0),
                               "B. Design Rationale for Each Block", GREEN_BORDER)
    add_bullet(tf7_b, "**GCN Layers**: Inexpensive isotropic smoothing that leverages continuous coupling weights $w_{ij}$.", pt_size=10.5)
    add_bullet(tf7_b, "**GATv2 Dynamic Attention**: Non-linearity precedes projection vector $a$, enabling query-dependent dynamic attention ranking (Brody et al. 2022; GATv1 fails this).", pt_size=10.5)
    add_bullet(tf7_b, "**Dual Pooling [Mean ‖ Max]**: Mean pooling captures broad global baseline; Max pooling isolates acute focal hub lesions.", pt_size=10.5)
    add_bullet(tf7_b, "**SwiGLU Head**: Multiplicative gating provides expressive feature compression prior to softmax.", pt_size=10.5)

    s7_notes = (
        "Slide 7 details our neural architecture, comprising 521,378 trainable parameters at hidden dimension 192. "
        "Raw 5-dimensional node features are projected through an initial linear layer with SiLU activation. We then "
        "stack three edge-weighted GCN layers with residual connections, followed by a 4-head GATv2 layer. GATv2 is "
        "essential here: unlike standard GATv1, its scoring function applies non-linear projection before computing "
        "attention logits, enabling true dynamic query-dependent ranking of brain regions. For graph readout, we "
        "concatenate global mean pooling with global max pooling. Mean pooling captures diffuse background connectivity, "
        "while max pooling detects localized hub disruptions. A SwiGLU gating head compresses 384 dimensions to 96 "
        "before the final two-class softmax projection."
    )
    set_notes(s7, s7_notes)

    # ----------------------------------------------------
    # SLIDE 8: Pipeline Step 6, Model Training
    # ----------------------------------------------------
    print("Building Slide 8...")
    s8 = prs.slides[7]
    set_slide_title(s8, "Pipeline Step 6: Model Training & Convergence Dynamics")
    
    # Box A: Blue border (Upper Left)
    _, tf8_a = add_content_box(s8, Inches(0.25), Inches(1.22), Inches(4.55), Inches(2.6),
                               "A. Objective & Optimiser", BLUE_BORDER)
    add_bullet(tf8_a, "**Label-Smoothed Cross-Entropy Loss** ($\\epsilon=0.1$ regularization):", pt_size=10.5, bold=True)
    add_equation_paragraph(tf8_a, r"\mathcal L(\theta)=-\frac1{|\mathcal B|}\sum_{k\in\mathcal B}\sum_{c\in\{0,1\}}q_{kc}\log\mathrm{softmax}(z_k)_c,\ q_{kc}=(1-\epsilon)\mathbb 1[c=y_k]+\frac{\epsilon}{2}", pt_size=10.5, space_after=2)
    
    add_bullet(tf8_a, "**Decoupled AdamW Optimization** ($\\eta_0=10^{-3}, \\lambda=10^{-4}$):", pt_size=10.5, bold=True)
    add_equation_paragraph(tf8_a, r"m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\ v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\ \theta_{t+1}=\theta_t-\eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}+\lambda\theta_t\Big)", pt_size=10, space_after=2)
    
    add_bullet(tf8_a, "**Cosine Learning Rate Schedule**: Warmup $\\eta_t = \\eta_0\\frac{t+1}{5}$ for $t < 5$; then cosine decay to zero over remaining 195 epochs.", pt_size=10)

    # Box B: Green border (Lower Left)
    _, tf8_b = add_content_box(s8, Inches(0.25), Inches(3.95), Inches(4.55), Inches(2.7),
                               "B. Tesla V100 Run Settings & Outcome", GREEN_BORDER)
    add_bullet(tf8_b, "**Volta Hardware Optimization**: FP16 mixed precision with `torch.cuda.amp.GradScaler()` (Volta lacks hardware BF16), batch size 32, DataLoader `pin_memory=True`.", pt_size=10.5)
    add_bullet(tf8_b, "**Early Stopping Checkpoint**: Monitored on **Validation Macro-F1**, patience 25. Best checkpoint reached at **Epoch 70**:", pt_size=10.5)
    add_bullet(tf8_b, "• **Val Macro-F1**: **0.765**  |  **Val ROC-AUC**: **0.822**  |  **Train F1**: **0.634**", pt_size=10.5, bold=True)
    add_bullet(tf8_b, "**Underfitting Regime Confirmed**: Train F1 stays below Val F1 throughout the entire run. Heavy dropout (0.25), label smoothing, and scale-free node inputs prevent overfitting. The model underfits, never overfits.", pt_size=10.5)

    # Visual: Fig 5 (Right)
    fig5_png = FIGURES_DIR / "fig5_training_curves.png"
    fig5_svg = FIGURES_DIR / "fig5_training_curves.svg"
    add_picture_with_svg(s8, fig5_png, fig5_svg, Inches(5.0), Inches(1.22), width=Inches(4.55))

    s8_notes = (
        "This slide tracks model optimization on our Tesla V100 GPU server. We train using AdamW with cosine learning "
        "rate scheduling and cross-entropy loss augmented with 10% label smoothing. Training runs in FP16 mixed precision "
        "using PyTorch GradScaler, with DataLoader memory pinning over the PCIe bus. We monitored validation Macro-F1 "
        "with an early stopping patience of 25 epochs. The optimal checkpoint was reached at epoch 70, recording a "
        "validation Macro-F1 of 0.765 and validation ROC-AUC of 0.822. Interestingly, training F1 remained below "
        "validation F1 throughout the run. This demonstrates that heavy dropout, label smoothing, and scale-free node "
        "inputs prevent overfitting, leaving the model in a mildly underfit regime."
    )
    set_notes(s8, s8_notes)

    # ----------------------------------------------------
    # SLIDE 9: Pipeline Step 7, Model Evaluation and Tuning
    # ----------------------------------------------------
    print("Building Slide 9...")
    s9 = prs.slides[8]
    set_slide_title(s9, "Pipeline Step 7: Model Evaluation & Threshold Tuning")
    
    # Box A: Blue border (Upper Left)
    _, tf9_a = add_content_box(s9, Inches(0.25), Inches(1.22), Inches(4.55), Inches(2.7),
                               "A. Held-Out Test Cohort Evaluation (N=75)", BLUE_BORDER)
    add_bullet(tf9_a, "**Evaluation Metrics** (AUC, Macro-F1, Youden's J):", pt_size=10.5, bold=True)
    add_equation_paragraph(tf9_a, r"\mathrm{AUC}=P(\hat p_{k^+}>\hat p_{k^-}),\ \mathrm{F1}_{\text{macro}}=\frac12\sum_c\frac{2\mathrm{TP}_c}{2\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c},\ J(t)=\mathrm{TPR}(t)-\mathrm{FPR}(t)", pt_size=9.5, space_after=2)
    
    t_shape9 = s9.shapes.add_table(4, 2, Inches(0.4), Inches(2.25), Inches(4.25), Inches(1.15))
    tbl9 = t_shape9.table
    tbl9.columns[0].width = Inches(2.8)
    tbl9.columns[1].width = Inches(1.45)
    
    headers9 = ["Evaluation Metric", "Measured Value"]
    for c_idx, h in enumerate(headers9):
        cell = tbl9.cell(0, c_idx)
        cell.text = h; cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"; p.font.size = Pt(9.5); p.font.bold = True
        p.font.color.rgb = BG_WHITE; p.alignment = PP_ALIGN.CENTER
        
    rows9 = [
        ("Test ROC-AUC (Headline Result)", "0.746"),
        ("Validation ROC-AUC", "0.823"),
        ("Test Accuracy / Macro-F1 (default t = 0.5)", "57.33% / 0.520"),
    ]
    for r_idx, row in enumerate(rows9):
        for c_idx, val in enumerate(row):
            cell = tbl9.cell(r_idx + 1, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"; p.font.size = Pt(9)
            if r_idx == 0: p.font.bold = True
            p.alignment = PP_ALIGN.CENTER if c_idx == 1 else PP_ALIGN.LEFT
            
    add_bullet(tf9_a, "AUC is threshold-free and rank-based, making it the primary reliable metric on small cohorts ($N=75$). Raw accuracy moves heavily with operating threshold.", pt_size=9.5)

    # Box B: Green border (Lower Left)
    _, tf9_b = add_content_box(s9, Inches(0.25), Inches(4.05), Inches(4.55), Inches(2.6),
                               "B. Tuning: Threshold Calibration Analysis", GREEN_BORDER)
    add_bullet(tf9_b, "**Operating Threshold Selection**:", pt_size=10.5, bold=True)
    add_bullet(tf9_b, "Youden's $J(t) = \\mathrm{TPR}(t) - \\mathrm{FPR}(t)$ selects optimal threshold $t^*$ on the ROC curve to balance clinical sensitivity and specificity.", pt_size=10)
    add_bullet(tf9_b, "**Probability Scale Shift**:", pt_size=10.5, bold=True)
    add_bullet(tf9_b, "Test scores sit lower than validation scores (mean predicted probability $\\hat{p} = 0.27$). Subject ranking remains strong (AUC 0.746), but fixed default thresholding ($t=0.5$) distorts accuracy.", pt_size=10)
    add_bullet(tf9_b, "**Methodological Next Steps**: Derive operating thresholds exclusively on validation splits, incorporate temperature scaling, and evaluate across 5-fold cross-validation.", pt_size=10)

    # Visual: Fig 6 (Right)
    fig6_png = FIGURES_DIR / "fig6_roc_thresholds.png"
    fig6_svg = FIGURES_DIR / "fig6_roc_thresholds.svg"
    add_picture_with_svg(s9, fig6_png, fig6_svg, Inches(5.0), Inches(1.22), width=Inches(4.55))

    s9_notes = (
        "Slide 9 evaluates our best checkpoint on the held-out test set of 75 unseen subjects. Our headline result is "
        "a test ROC-AUC of 0.746, down from 0.823 on validation. At the standard 0.5 decision threshold, raw accuracy is "
        "57.33% with Macro-F1 of 0.520. Because test probabilities shifted downward toward a mean of 0.27, fixed "
        "thresholds penalize raw accuracy. However, ROC-AUC evaluates ranking discrimination independent of threshold "
        "shifting. Applying Youden's J statistic identifies an optimal operating point, balancing sensitivity and "
        "specificity. Likely question: Why is raw test accuracy around 57% when ROC-AUC reaches 0.746? Answer: In small "
        "75-subject test sets, raw accuracy is highly sensitive to probability calibration shifts, whereas ROC-AUC "
        "reliably reflects true underlying discriminative separation."
    )
    set_notes(s9, s9_notes)

    # ----------------------------------------------------
    # SLIDE 10: Results, Explainability and Comparison with the 4 Base Papers
    # ----------------------------------------------------
    print("Building Slide 10...")
    s10 = prs.slides[9]
    set_slide_title(s10, "Results, Explainability & Baseline Comparison")
    
    t_shape10 = s10.shapes.add_table(8, 6, Inches(0.25), Inches(1.22), Inches(9.3), Inches(2.35))
    tbl10 = t_shape10.table
    tbl10.columns[0].width = Inches(3.6)
    tbl10.columns[1].width = Inches(1.2)
    tbl10.columns[2].width = Inches(1.0)
    tbl10.columns[3].width = Inches(1.0)
    tbl10.columns[4].width = Inches(1.0)
    tbl10.columns[5].width = Inches(1.5)
    
    headers10 = ["Model (Base Paper)", "Graph-Aware", "Test AUC", "Test Acc", "Macro-F1", "Source"]
    for c_idx, h in enumerate(headers10):
        cell = tbl10.cell(0, c_idx)
        cell.text = h; cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Calibri"; p.font.size = Pt(9.5); p.font.bold = True
        p.font.color.rgb = BG_WHITE; p.alignment = PP_ALIGN.CENTER
        
    rows10 = [
        ("RBF-SVM on flattened connectome (Abraham 2017)", "No", "0.530", "53.3%", "0.533", "re-run, this split"),
        ("L2 logistic regression, flattened", "No", "0.509", "49.3%", "0.493", "re-run, this split"),
        ("RF on graphlet degree vectors (Pržulj 2007)", "Motifs", "0.692†", "56.0%†", "0.534†", "README, no log"),
        ("node2vec + classifier (Grover 2016)", "Walks", "0.635†", "52.0%†", "0.498†", "README, no log"),
        ("GATv1, mean pool (Veličković 2018)", "Static attn.", "0.713†", "57.3%†", "0.550†", "README, no log"),
        ("Ours: edge-aware GCN + GATv2 + dual pool + SwiGLU", "Dynamic attn.", "0.746", "57.3%", "0.520", "test_metrics.json"),
        ("Sanity check: 1 scalar, mean |z| intra-lobe", "(oracle)", "1.000", "100%", "1.000", "re-run, this split"),
    ]
    for r_idx, row in enumerate(rows10):
        is_ours = (r_idx == 5)
        for c_idx, val in enumerate(row):
            cell = tbl10.cell(r_idx + 1, c_idx)
            cell.text = val
            if is_ours:
                cell.fill.solid(); cell.fill.fore_color.rgb = BG_LIGHT_GREEN
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Calibri"; p.font.size = Pt(8.5)
            if is_ours: p.font.bold = True
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER

    fn_box = s10.shapes.add_textbox(Inches(0.25), Inches(3.58), Inches(9.3), Inches(0.25))
    p_fn = fn_box.text_frame.paragraphs[0]
    p_fn.text = "† Baseline numbers are from our earlier README runs. Only the SVM / logistic regression / one-scalar rows were re-run on this exact split."
    p_fn.font.name = "Calibri"; p_fn.font.size = Pt(8); p_fn.font.italic = True
    p_fn.font.color.rgb = TEXT_MUTED

    _, tf10_b = add_content_box(s10, Inches(0.25), Inches(3.85), Inches(3.1), Inches(2.8),
                                "B. What Improved and Why", GREEN_BORDER)
    add_bullet(tf10_b, "**+0.216 AUC Gain over Flattened SVM** (0.530 → 0.746):", pt_size=9.5, bold=True)
    add_bullet(tf10_b, "Flattened SVM operates on $p=6670$ coordinates from only 350 training subjects. The GNN shares parameters across nodes, preventing $O(n^2)$ sample complexity explosion.", pt_size=9)
    add_bullet(tf10_b, "**Continuous Edge Attributes Matter**:", pt_size=9.5, bold=True)
    add_bullet(tf10_b, "Continuous $z_{ij}$ enters both GCN normalisation and GATv2 scoring, preserving the only magnitude signal surviving scale-free node features.", pt_size=9)

    _, tf10_c = add_content_box(s10, Inches(3.45), Inches(3.85), Inches(3.2), Inches(2.8),
                                "C. Explainability & Ground Truth", BLUE_BORDER)
    add_bullet(tf10_c, "**GNNExplainer Consensus Pipeline**:", pt_size=9.5, bold=True)
    add_bullet(tf10_c, "Optimizes soft edge mask $M_e$ (500 steps, lr 0.01) with frozen $\\theta$. Aggregates top-10 edges into consensus circuits.", pt_size=9)
    add_bullet(tf10_c, "**Precision Evaluation Audit**:", pt_size=9.5, bold=True)
    add_bullet(tf10_c, "**15.3%** of top-10 edges are intra-lobe, matching chance level (**16.0%**). The explainer does not yet recover the planted pattern.", pt_size=9)
    add_bullet(tf10_c, "**Why**: Model lacks anatomical region identity. A faithful explainer cannot report patterns the model cannot structurally represent.", pt_size=9)

    fig8_png = FIGURES_DIR / "fig8_explainer_vs_ground_truth.png"
    fig8_svg = FIGURES_DIR / "fig8_explainer_vs_ground_truth.svg"
    add_picture_with_svg(s10, fig8_png, fig8_svg, Inches(6.75), Inches(3.85), width=Inches(2.8))

    s10_notes = (
        "Slide 10 summarizes our benchmark comparisons and explainer findings. Compared to the flattened RBF-SVM baseline "
        "from Abraham 2017 at 0.530 AUC, our architecture achieves 0.746 AUC—a gain of 0.216. Weight-sharing across "
        "nodes prevents parameter explosion on small cohorts. For explainability, we ran GNNExplainer for 500 iterations "
        "per subject to extract top-10 salient circuits. Surprisingly, only 15.3% of explained edges lie within anatomical "
        "lobes, indistinguishable from the 16.0% chance level. Why? The network was never given anatomical lobe identities, "
        "so its node convolutions cannot learn lobe-specific groupings. A faithful explainer cannot highlight features "
        "the model does not use. Likely question: Why is the explainer precision near chance? Answer: The network "
        "lacks anatomical lobe features, so faithful explainability correctly reflects that the model relies on edge weights "
        "rather than lobe structures."
    )
    set_notes(s10, s10_notes)

    # ----------------------------------------------------
    # SLIDE 11: Thank You
    # ----------------------------------------------------
    print("Building Slide 11...")
    s11 = prs.slides[10]
    set_slide_title(s11, "Summary & Future Roadmap")
    
    _, tf11 = add_content_box(s11, Inches(1.0), Inches(1.5), Inches(7.5), Inches(4.8),
                              "Project Synthesis & Acknowledgments", BLUE_BORDER)
    add_bullet(tf11, "**Key Achievements Demonstrated in Part 2**:", pt_size=12, bold=True)
    add_bullet(tf11, "• Successfully built and validated an end-to-end neuroimaging pipeline coupling Neo4j property graphs with geometric deep learning.", pt_size=11)
    add_bullet(tf11, "• Achieved **0.746 Test ROC-AUC** across multi-scanner synthetic cohorts, gaining **+0.216 AUC** over flattened clinical SVM baselines.", pt_size=11)
    add_bullet(tf11, "• Demonstrated that continuous edge attributes prevent feature attenuation, outperforming scale-free topological descriptors.", pt_size=11)
    add_bullet(tf11, "• Conducted rigorous mathematical audits exposing betweenness distance inversion and explainer faithfulness boundaries.", pt_size=11)
    
    add_bullet(tf11, "**Immediate Roadmap Prior to Next Review**:", pt_size=12, bold=True)
    add_bullet(tf11, "• Incorporate anatomical lobe assignments as learnable positional node embeddings to unlock lobe-level disease patterns.", pt_size=11)
    add_bullet(tf11, "• Apply temperature scaling and validation-derived operating thresholds to eliminate probability scale shifts.", pt_size=11)
    add_bullet(tf11, "• Validate the complete framework on real Alzheimer's (ADNI) and Parkinson's (PPMI) cohorts once data agreements are executed.", pt_size=11)
    
    add_bullet(tf11, "**Acknowledgments & Resources**:", pt_size=12, bold=True)
    add_bullet(tf11, "Department of Computer Science & Engineering, Indian Institute of Technology Roorkee.", pt_size=11)
    add_bullet(tf11, "Project Repository: `basakanimesh14 / Neuro-Neo4j` (Optimized for NVIDIA Tesla V100 GPU cluster).", pt_size=11)

    s11_notes = (
        "In this final slide, I summarize how our hybrid graph architecture establishes a functional bridge between property databases and "
        "geometric deep learning for neuroimaging. We demonstrated a +0.216 AUC improvement over flattened vector "
        "baselines on multi-scanner connectomes, verified the protective role of continuous edge attributes, and "
        "identified critical representational bottlenecks through our explainer audit. Our immediate roadmap focuses "
        "on incorporating anatomical lobe coordinates as positional embeddings, calibrating prediction probabilities "
        "via temperature scaling, and validating the framework on clinical ADNI cohorts once data agreements are executed. "
        "Thank you for your time and guidance. I welcome your questions and discussion."
    )
    set_notes(s11, s11_notes)

    print(f"Saving final presentation to {OUTPUT_PPTX}...")
    prs.save(str(OUTPUT_PPTX))
    print("Presentation successfully created!")

if __name__ == "__main__":
    build_presentation()
