import sys
import os
import re
from pathlib import Path
from lxml import etree

import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml import parse_xml

import latex2mathml.converter
import mathml2omml

WORKSPACE_ROOT = Path("/mnt/hdd/animesh/Neuro-Neo4j")
TEMPLATE_PATH = Path("/home/user/Desktop/Rohit_Raj_BTP/AntiGravity/report/format/PPT_Group-No_RS-1.pptx")
FIGURES_DIR = WORKSPACE_ROOT / "presentation2" / "figures"
OUTPUT_PPTX = WORKSPACE_ROOT / "Neuro_Neo4j_Presentation2.pptx"

# Color constants
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

print("Base deck setup ready.")
