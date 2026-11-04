"""
build_presentation.py — Generates Presentation 2 (.pptx) for Neuro-Neo4j IIT Roorkee presentation.
Follows all specifications in PPT2_SLIDES.md and user guidelines:
- Reuses IITR_template_pptx(19062016) via PPT_Group-No_RS-1.pptx
- Black bold titles, blue rule, IITR logo top right, IIT ROORKEE footer bar & page number
- Rounded content boxes with navy bold headings, colored borders (blue=neutral, green=result/fix, red=problem/limitation)
- Native OMML PowerPoint equations fitting on 14 pt lines
- Native PowerPoint tables (Slide 3 site table, Slide 9 metrics table, Slide 10 comparison table)
- Native PowerPoint Flowchart 1 (Slide 4) and Flowchart 2 (Slide 7)
- Figures placed with SVG + PNG fallback
- Exact dagger footnote on slide 10
- Speaker notes on every slide: 90-130 words in first person, answering required questions on slides 3, 9, 10
- Exact verified numbers (Test ROC-AUC 0.746; never mentions 73.33% accuracy on slides)
"""

import os
import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.opc.package import Part

import latex2mathml.converter
import mathml2omml

TEMPLATE_PATH = "/home/user/Desktop/Rohit_Raj_BTP/AntiGravity/report/format/PPT_Group-No_RS-1.pptx"
REPO_ROOT = Path("/mnt/hdd/animesh/Neuro-Neo4j")
FIG_DIR = REPO_ROOT / "presentation2" / "figures"
OUTPUT_PPTX = REPO_ROOT / "presentation2" / "Neuro_Neo4j_Presentation2.pptx"

# Color definitions
NAVY = RGBColor(31, 56, 100)      # #1F3864 - Headings
BLACK = RGBColor(0, 0, 0)         # Titles
DARK_GRAY = RGBColor(30, 30, 30)  # Body text
WHITE = RGBColor(255, 255, 255)   # Background fill

BORDER_BLUE = RGBColor(42, 120, 214)   # Neutral content (#2A78D6)
BORDER_GREEN = RGBColor(30, 126, 52)   # Results / fixes (#1E7E34)
BORDER_RED = RGBColor(192, 0, 0)       # Problems / limitations (#C00000)

FILL_LIGHT_BLUE = RGBColor(220, 235, 250) # #DCEBFA (Flowchart 1)
FILL_LIGHT_GREEN = RGBColor(221, 242, 228) # #DDF2E4 (Flowchart 1)

def add_math_to_para(p, latex_str):
    """Converts a LaTeX string to OMML and appends to paragraph."""
    try:
        mml = latex2mathml.converter.convert(latex_str)
        omml = mathml2omml.convert(mml)
        m_decl = nsdecls("m")
        if "xmlns:m" not in omml:
            omml = omml.replace("<m:oMath>", f"<m:oMath {m_decl}>")
        elem = parse_xml(omml)
        p._p.append(elem)
    except Exception as e:
        print(f"Warning: could not render math '{latex_str}': {e}")
        r = p.add_run()
        r.text = latex_str

def set_para_font(p, name="Calibri", size_pt=12, bold=False, italic=False, color=DARK_GRAY):
    p.font.name = name
    p.font.size = Pt(size_pt)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color

def create_rounded_box(slide, left, top, width, height, title, border_color):
    """Creates a rounded content box with centred bold navy heading."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = border_color
    box.line.width = Pt(2.0)
    
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.12)
    tf.margin_bottom = Inches(0.12)
    
    # Heading paragraph
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    p0.text = title
    set_para_font(p0, name="Calibri", size_pt=13, bold=True, color=NAVY)
    p0.space_after = Pt(6)
    
    return box, tf

def add_bullet_item(tf, prefix="", text_segments=None, pt_size=11, bold_prefix=True, space_after=4):
    """
    Appends a bullet paragraph to a text frame.
    text_segments: list of tuples: (text, is_math, bold, italic)
    """
    p = tf.add_paragraph()
    p.space_after = Pt(space_after)
    p.space_before = Pt(2)
    p.level = 0
    
    if prefix:
        r0 = p.add_run()
        r0.text = prefix + " "
        set_para_font(p, size_pt=pt_size)
        r0.font.bold = bold_prefix
        r0.font.name = "Calibri"
        r0.font.size = Pt(pt_size)
        r0.font.color.rgb = DARK_GRAY
        
    if text_segments:
        for item in text_segments:
            seg_text = item[0]
            is_math = item[1] if len(item) > 1 else False
            is_bold = item[2] if len(item) > 2 else False
            is_italic = item[3] if len(item) > 3 else False
            color = item[4] if len(item) > 4 else DARK_GRAY
            
            if is_math:
                add_math_to_para(p, seg_text)
            else:
                r = p.add_run()
                r.text = seg_text
                r.font.name = "Calibri"
                r.font.size = Pt(pt_size)
                r.font.bold = is_bold
                r.font.italic = is_italic
                r.font.color.rgb = color
    return p

def add_slide_title(slide, title_text):
    """Adds standard black bold title at top left."""
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.18), Inches(8.5), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0)
    tf.margin_top = Inches(0)
    p = tf.paragraphs[0]
    p.text = title_text
    set_para_font(p, name="Calibri", size_pt=20, bold=True, color=BLACK)
    return tb

def add_picture_with_svg(slide, prs, base_name, left, top, width=None, height=None):
    """Embeds PNG with SVG vector extension in PPTX."""
    png_path = FIG_DIR / f"{base_name}.png"
    svg_path = FIG_DIR / f"{base_name}.svg"
    
    if not png_path.exists():
        print(f"Error: {png_path} does not exist!")
        return None
        
    pic = slide.shapes.add_picture(str(png_path), left, top, width=width, height=height)
    
    if svg_path.exists():
        try:
            with open(svg_path, "rb") as f:
                svg_blob = f.read()
            partname = prs.part.package.next_image_partname("svg")
            svg_part = Part(partname, "image/svg+xml", prs.part.package, svg_blob)
            rId = slide.part.relate_to(svg_part, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
            
            blip = pic._element.xpath(".//a:blip")[0]
            ext_xml = f"""
            <a:extLst {nsdecls("a")}>
              <a:ext uri="{{96DAC542-7B16-4481-9AB0-3857D30F1308}}">
                <asvg:svgBlip xmlns:asvg="http://schemas.microsoft.com/office/drawing/2016/SVG/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="{rId}"/>
              </a:ext>
            </a:extLst>
            """
            blip.append(parse_xml(ext_xml))
        except Exception as e:
            print(f"Warning: SVG embedding for {base_name} fell back to PNG: {e}")
            
    return pic

def add_speaker_notes(slide, notes_text):
    """Attaches required speaker script to notes pane."""
    slide.notes_slide.notes_text_frame.text = notes_text.strip()

