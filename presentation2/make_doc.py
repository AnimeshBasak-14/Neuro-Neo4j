"""
make_doc.py — Complete builder for Neuro_Neo4j_Companion_Document.docx
"""

import re
import sys
from pathlib import Path

import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from build_companion_document import (
    create_document, set_para_format, add_display_omml,
    render_rich_text, add_heading_1, add_heading_2, add_heading_3,
    add_figure_with_caption, add_styled_table,
    REPO_ROOT, SOURCE_MD, FIG_DIR, OUTPUT_DOCX,
    NAVY, DARK_GRAY, GRAY, ACCENT_BLUE
)

def add_title_page(doc):
    p_top = doc.add_paragraph()
    set_para_format(p_top, space_before=72, space_after=12)
    
    # Institution
    p_inst = doc.add_paragraph()
    set_para_format(p_inst, space_before=0, space_after=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_inst = p_inst.add_run("INDIAN INSTITUTE OF TECHNOLOGY ROORKEE")
    r_inst.font.name = "Calibri"
    r_inst.font.size = Pt(14)
    r_inst.font.bold = True
    r_inst.font.color.rgb = GRAY
    
    # Title
    p_title = doc.add_paragraph()
    set_para_format(p_title, space_before=12, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_title = p_title.add_run("Explainable Graph Neural Networks for Brain Biomarker Discovery:\nMathematics and Logic of the Pipeline")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(24)
    r_title.font.bold = True
    r_title.font.color.rgb = NAVY
    
    # Subtitle
    p_sub = doc.add_paragraph()
    set_para_format(p_sub, space_before=0, space_after=36, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_sub = p_sub.add_run("Companion Reference Document to Presentation 2 (Execution & Results)")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(15)
    r_sub.font.italic = True
    r_sub.font.color.rgb = ACCENT_BLUE
    
    # Decorative line
    p_line = doc.add_paragraph()
    set_para_format(p_line, space_before=0, space_after=72, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_line = p_line.add_run("—" * 32)
    r_line.font.name = "Calibri"
    r_line.font.size = Pt(12)
    r_line.font.color.rgb = ACCENT_BLUE
    
    # Author & Institution Metadata
    p_meta = doc.add_paragraph()
    set_para_format(p_meta, space_before=36, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_auth = p_meta.add_run("Animesh Basak (Roll No. 26911001)\n")
    r_auth.font.name = "Calibri"
    r_auth.font.size = Pt(13)
    r_auth.font.bold = True
    r_auth.font.color.rgb = DARK_GRAY
    
    r_dept = p_meta.add_run("Department of Computer Science & Engineering\nIndian Institute of Technology Roorkee\n")
    r_dept.font.name = "Calibri"
    r_dept.font.size = Pt(12)
    r_dept.font.color.rgb = DARK_GRAY
    
    r_date = p_meta.add_run("November 2026")
    r_date.font.name = "Calibri"
    r_date.font.size = Pt(12)
    r_date.font.italic = True
    r_date.font.color.rgb = GRAY
    
    doc.add_page_break()

def add_table_of_contents(doc):
    p_toc_title = doc.add_paragraph()
    set_para_format(p_toc_title, space_before=12, space_after=12)
    r_toc = p_toc_title.add_run("Table of Contents")
    r_toc.font.name = "Calibri"
    r_toc.font.size = Pt(18)
    r_toc.font.bold = True
    r_toc.font.color.rgb = NAVY
    
    # Dotted leader entries for visual rendering in PDF and Word
    toc_items = [
        ("How to Read this Document", "3", 0),
        ("Global Notation", "3", 1),
        ("Part I: The Four Base Papers", "4", 0),
        ("1. Kernel SVM on Vectorised Connectomes (Abraham et al., NeuroImage 2017)", "4", 1),
        ("2. Random Forest on Graphlet Degree Vectors (Pržulj, Bioinformatics 2007)", "6", 1),
        ("3. node2vec Embeddings (Grover & Leskovec, KDD 2016)", "7", 1),
        ("4. Graph Attention Networks, GATv1 (Veličković et al., ICLR 2018)", "8", 1),
        ("Part II: Problem Statement", "10", 0),
        ("Part III: The Pipeline, Step by Step", "11", 0),
        ("5. Step 1: Data Collection", "11", 1),
        ("6. Step 2: Cleaning and Preprocessing", "13", 1),
        ("7. Step 3: Exploratory Data Analysis", "15", 1),
        ("8. Step 4: Feature Engineering and Selection", "17", 1),
        ("9. Step 5: Model Selection", "19", 1),
        ("10. Step 6: Training", "23", 1),
        ("11. Step 7: Evaluation and Tuning", "25", 1),
        ("12. Step 8: Explainability with GNNExplainer", "27", 1),
        ("Part IV: Interpretation", "30", 0),
        ("13. Why the GNN Reached 0.746 and Not 1.000", "30", 1),
        ("14. Comparison with the Four Base Papers", "32", 1),
        ("15. What Changes Before the Next Review", "34", 1),
        ("16. References", "35", 1),
    ]
    
    for title, page, level in toc_items:
        p = doc.add_paragraph()
        indent = Inches(0.25 * level)
        set_para_format(p, space_before=1, space_after=2)
        p.paragraph_format.left_indent = indent
        
        r1 = p.add_run(title)
        r1.font.name = "Calibri"
        r1.font.size = Pt(10.5 if level == 0 else 10)
        r1.font.bold = (level == 0)
        r1.font.color.rgb = NAVY if level == 0 else DARK_GRAY
        
        # Dots leader
        dots_count = max(4, 75 - len(title) - level * 4)
        r_dots = p.add_run(" " + ". " * dots_count)
        r_dots.font.name = "Calibri"
        r_dots.font.size = Pt(8.5)
        r_dots.font.color.rgb = RGBColor(180, 180, 180)
        
        r2 = p.add_run(f" {page}")
        r2.font.name = "Calibri"
        r2.font.size = Pt(10)
        r2.font.bold = (level == 0)
        r2.font.color.rgb = DARK_GRAY

    # Auto TOC field code for native Word updating
    p_fld = doc.add_paragraph()
    set_para_format(p_fld, space_before=0, space_after=12)
    fld_xml = r'''
    <w:p %s>
      <w:r><w:fldChar w:fldCharType="begin"/></w:r>
      <w:r><w:instrText xml:space="preserve"> TOC \o "1-3" \h \z \u </w:instrText></w:r>
      <w:r><w:fldChar w:fldCharType="separate"/></w:r>
      <w:r><w:fldChar w:fldCharType="end"/></w:r>
    </w:p>
    ''' % nsdecls("w")
    p_fld._p.append(parse_xml(fld_xml))
    
    doc.add_page_break()

def parse_and_build():
    print(f"Reading markdown from {SOURCE_MD}...")
    with open(SOURCE_MD, "r", encoding="utf-8") as f:
        content = f.read()

    doc = create_document()
    add_title_page(doc)
    add_table_of_contents(doc)
    
    lines = content.splitlines()
    i = 0
    in_yaml = False
    
    while i < len(lines):
        line = lines[i]
        
        # Skip frontmatter
        if i == 0 and line.strip() == "---":
            in_yaml = True
            i += 1
            while i < len(lines) and lines[i].strip() != "---":
                i += 1
            i += 1
            continue
            
        strip_l = line.strip()
        
        # Skip empty lines and \newpage
        if not strip_l or strip_l == r"\newpage":
            i += 1
            continue
            
        # 1. Headings
        if strip_l.startswith("# "):
            heading_text = strip_l[2:].strip()
            # If Part heading, add page break before except for How to read
            if heading_text.startswith("Part"):
                doc.add_page_break()
            add_heading_1(doc, heading_text)
            i += 1
            continue
        elif strip_l.startswith("## "):
            heading_text = strip_l[3:].strip()
            add_heading_2(doc, heading_text)
            i += 1
            continue
        elif strip_l.startswith("### "):
            heading_text = strip_l[4:].strip()
            add_heading_3(doc, heading_text)
            i += 1
            continue
            
        # 2. Display Equations: $$...$$
        if strip_l.startswith("$$"):
            # Could be single line or multiline
            if strip_l.endswith("$$") and len(strip_l) > 4:
                eq_text = strip_l[2:-2].strip()
                add_display_omml(doc, eq_text)
                i += 1
                continue
            else:
                # Multiline equation
                eq_lines = []
                first_line = strip_l[2:].strip()
                if first_line:
                    eq_lines.append(first_line)
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    eq_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    last_line = lines[i].strip()[:-2].strip()
                    if last_line:
                        eq_lines.append(last_line)
                    i += 1
                add_display_omml(doc, "\n".join(eq_lines))
                continue
                
        # 3. Markdown Tables: lines starting with |
        if strip_l.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
                
            if len(table_lines) >= 3:
                # Header row
                raw_headers = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                # Data rows (skipping separator row 1)
                data_rows = []
                for t_row in table_lines[2:]:
                    cells = [c.strip() for c in t_row.split("|")[1:-1]]
                    data_rows.append(cells)
                
                # Special column width settings for specific tables
                if "Symbol" in raw_headers[0]:
                    col_widths = [1.5, 5.0]
                elif "Threshold rule" in raw_headers[0]:
                    col_widths = [2.2, 1.2, 1.5, 1.6]
                elif "Quantity" in raw_headers[0]:
                    col_widths = [2.6, 2.5, 1.4]
                elif "Model" in raw_headers[0]:
                    col_widths = [2.4, 1.1, 0.8, 0.7, 0.8, 0.7]
                else:
                    col_widths = None
                    
                add_styled_table(doc, raw_headers, data_rows, col_widths=col_widths)
            continue
            
        # 4. Code Blocks
        if strip_l.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1 # skip closing ```
            
            p_code = doc.add_paragraph()
            set_para_format(p_code, space_before=4, space_after=6)
            p_code.paragraph_format.left_indent = Inches(0.25)
            r_c = p_code.add_run("\n".join(code_lines))
            r_c.font.name = "Consolas"
            r_c.font.size = Pt(9.5)
            r_c.font.color.rgb = RGBColor(60, 60, 60)
            continue
            
        # 5. Bullet Lists / Numbered Lists
        if strip_l.startswith("- ") or strip_l.startswith("* "):
            p = doc.add_paragraph()
            set_para_format(p, space_before=1, space_after=3)
            p.paragraph_format.left_indent = Inches(0.25)
            r_bullet = p.add_run("•  ")
            r_bullet.font.name = "Calibri"
            r_bullet.font.size = Pt(11)
            r_bullet.font.bold = True
            r_bullet.font.color.rgb = ACCENT_BLUE
            render_rich_text(p, strip_l[2:].strip())
            i += 1
            continue
        elif re.match(r'^\d+\.\s+', strip_l):
            m = re.match(r'^(\d+\.)\s+(.+)$', strip_l)
            p = doc.add_paragraph()
            set_para_format(p, space_before=1, space_after=3)
            p.paragraph_format.left_indent = Inches(0.25)
            r_num = p.add_run(f"{m.group(1)}  ")
            r_num.font.name = "Calibri"
            r_num.font.size = Pt(11)
            r_num.font.bold = True
            r_num.font.color.rgb = DARK_GRAY
            render_rich_text(p, m.group(2))
            i += 1
            continue
            
        # 6. Standard Paragraph
        p = doc.add_paragraph()
        set_para_format(p, space_before=0, space_after=5, line_spacing=1.15)
        render_rich_text(p, strip_l)
        
        # Check if we should insert figures at specific text anchors!
        # Figure 1: after Section 7.1 Density and weights
        if "Nothing in the first-order statistics separates the classes." in strip_l:
            add_figure_with_caption(doc, "fig1_eda_edges", 1,
                                    "First-order graph density and edge weight distributions across Healthy and Diseased cohorts.")
            
        # Figures 2, 3, 4: after Section 7.2 Locating the signal
        if "Healthy medians sit near 0.765 and Diseased near 0.895 with no overlap (Fig 2)." in strip_l:
            add_figure_with_caption(doc, "fig2_eda_signal_location", 2,
                                    "Distribution of intra-lobe and inter-lobe connection strengths showing separation by planted signal.")
            add_figure_with_caption(doc, "fig3_group_difference_matrix", 3,
                                    "Group-difference matrix $\\Delta_{ij}$ by anatomical lobe showing planted hyperconnectivity blocks and homotopic attenuation.")
            add_figure_with_caption(doc, "fig4_brain_graph_subject", 4,
                                    "Anatomical graph visualization of single-subject functional connectome on the AAL-116 atlas.")
            
        # Figure 5: after Section 10.4 Early stopping
        if "The model is not memorising 350 subjects, it is failing to fit them. Section 13 explains why." in strip_l:
            add_figure_with_caption(doc, "fig5_training_curves", 5,
                                    "Training and validation loss, Macro-F1, and ROC-AUC convergence trajectories over 70 epochs on NVIDIA Tesla V100.")
            
        # Figure 6: in Section 11.2 after results paragraph
        if "Test ROC-AUC = **0.746**; validation ROC-AUC = 0.823." in strip_l:
            add_figure_with_caption(doc, "fig6_roc_thresholds", 6,
                                    "Held-out test ROC curve showing default (0.5), validation-optimal (0.503), and leaky test-calibrated (0.176) operating thresholds.")
            
        # Figure 8: in Section 12.4 after audit table
        if "The region ranking in `biomarker_explanations.json` (Vermis_10, Vermis_9, … in that order) is what you get when every importance ties at 0" in strip_l:
            add_figure_with_caption(doc, "fig8_explainer_vs_ground_truth", 8,
                                    "GNNExplainer top-10 consensus biomarker circuits compared against the planted biological ground-truth connectivity matrix.")
            
        # Figure 7: in Section 14 after comparison table
        if "That gain is real and it is the main positive result." in strip_l or "- Against Paper 1 we have a verified, reproduced improvement in ranking quality (AUC +0.216)." in strip_l:
            add_figure_with_caption(doc, "fig7_model_comparison", 7,
                                    "Empirical comparison of test ROC-AUC across flattened machine learning baselines, topological walks, and our edge-aware GCN-GATv2 architecture.")
            
        i += 1

    print(f"Saving Word document to {OUTPUT_DOCX}...")
    doc.save(OUTPUT_DOCX)
    print("Word document saved successfully!")

if __name__ == "__main__":
    parse_and_build()
