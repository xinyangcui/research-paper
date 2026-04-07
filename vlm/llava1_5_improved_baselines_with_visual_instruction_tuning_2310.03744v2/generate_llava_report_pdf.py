from __future__ import annotations

import html
import io
import re
import sys
from pathlib import Path

import markdown as md
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(__file__).resolve().parent
MARKDOWN_PATH = ROOT / "llava1_5_improved_baselines_with_visual_instruction_tuning_reading_report.md"
PDF_OUTPUT_PATH = ROOT / "llava1_5_improved_baselines_with_visual_instruction_tuning_reading_report.pdf"
HTML_OUTPUT_PATH = ROOT / "llava1_5_improved_baselines_with_visual_instruction_tuning_2310.03744v2.html"


def register_fonts() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))


def build_styles():
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyZH",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=11,
        leading=17,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    return {
        "title": ParagraphStyle(
            "TitleZH",
            parent=styles["Title"],
            fontName="STSong-Light",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            spaceAfter=10,
            textColor=colors.HexColor("#17324d"),
        ),
        "h1": ParagraphStyle(
            "H1ZH",
            parent=styles["Heading1"],
            fontName="STSong-Light",
            fontSize=17,
            leading=22,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor("#12355b"),
        ),
        "h2": ParagraphStyle(
            "H2ZH",
            parent=styles["Heading2"],
            fontName="STSong-Light",
            fontSize=14,
            leading=19,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor("#1c4f7a"),
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "BulletZH",
            parent=body,
            leftIndent=14,
            firstLineIndent=-10,
            bulletIndent=0,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "CaptionZH",
            parent=body,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceBefore=3,
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "SmallZH",
            parent=body,
            fontSize=9.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#666666"),
        ),
        "code": ParagraphStyle(
            "CodeZH",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.HexColor("#f4f6f8"),
            borderPadding=6,
            borderWidth=0.5,
            borderColor=colors.HexColor("#c7d1db"),
            borderRadius=3,
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def format_inline(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"`([^`]+)`",
        lambda m: f'<font name="Courier">{html.escape(m.group(1))}</font>',
        text,
    )
    return text.replace("\n", "<br/>")


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        width, height = img.size
        scale = min(max_width / width, max_height / height, 1.0)
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        if target_size != img.size:
            img = img.resize(target_size, PILImage.Resampling.LANCZOS)

        buf = io.BytesIO()
        save_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        if save_format == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format=save_format, quality=92)
        buf.seek(0)

    flowable = Image(buf, width=target_size[0], height=target_size[1])
    flowable._img_buffer = buf
    return flowable


def make_llava15_pipeline_diagram() -> Drawing:
    drawing = Drawing(510, 240)

    def box(x, y, w, h, label, fill="#e8f1fa", size=10):
        drawing.add(
            Rect(
                x,
                y,
                w,
                h,
                rx=8,
                ry=8,
                strokeColor=colors.HexColor("#355070"),
                fillColor=colors.HexColor(fill),
                strokeWidth=1.2,
            )
        )
        for i, line in enumerate(label.split("\n")):
            drawing.add(
                String(
                    x + w / 2,
                    y + h / 2 + (len(label.split("\n")) - 1 - 2 * i) * 6,
                    line,
                    fontName="Helvetica",
                    fontSize=size,
                    fillColor=colors.HexColor("#1f2d3d"),
                    textAnchor="middle",
                )
            )

    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.4))
        drawing.add(Line(x2 - 6, y2 + 3, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.4))
        drawing.add(Line(x2 - 6, y2 - 3, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.4))

    drawing.add(
        String(
            255,
            224,
            "LLaVA-1.5 Training / Data Flow",
            fontName="Helvetica-Bold",
            fontSize=14,
            fillColor=colors.HexColor("#17324d"),
            textAnchor="middle",
        )
    )

    box(10, 160, 95, 42, "558K\nImage-Text Pairs", "#dbe7c9")
    box(130, 160, 92, 42, "CLIP ViT-L/14\n336px", "#d9eaf7")
    box(247, 160, 100, 42, "MLP Projector\nmlp2x_gelu", "#f8e6b8")
    box(372, 160, 92, 42, "Vicuna v1.5\n7B / 13B", "#f7d9d9")

    box(10, 85, 110, 44, "665K Instruct Mix\nLLaVA / VQA / OCR /\nRegion / ShareGPT", "#ece8f6", size=9)
    box(145, 85, 96, 44, "Response Format\nPrompts", "#eaf4e2")
    box(266, 85, 96, 44, "Conversation JSON\nimage + conversations", "#e6f5f1", size=9)
    box(387, 85, 110, 44, "Visual Instruction\nTuning", "#fde7dd")

    box(130, 18, 100, 40, "HD Variant:\nsplit grids", "#eef2ff")
    box(255, 18, 100, 40, "remove pad +\nrow-end token", "#f3ecff")
    box(380, 18, 100, 40, "concat global\ncontext", "#e8f5e9")

    arrow(105, 181, 130, 181)
    arrow(222, 181, 247, 181)
    arrow(347, 181, 372, 181)
    arrow(120, 107, 145, 107)
    arrow(241, 107, 266, 107)
    arrow(362, 107, 387, 107)
    drawing.add(Line(297, 129, 297, 156, strokeColor=colors.HexColor("#355070"), strokeWidth=1.2))
    drawing.add(Line(297, 156, 297, 160, strokeColor=colors.HexColor("#355070"), strokeWidth=1.2))
    arrow(230, 38, 255, 38)
    arrow(355, 38, 380, 38)
    drawing.add(Line(430, 58, 430, 82, strokeColor=colors.HexColor("#355070"), strokeWidth=1.2))
    drawing.add(Line(430, 82, 430, 85, strokeColor=colors.HexColor("#355070"), strokeWidth=1.2))

    drawing.add(
        String(
            255,
            147,
            "Stage 1: alignment pretraining",
            fontName="Helvetica",
            fontSize=10,
            fillColor=colors.HexColor("#40566b"),
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            255,
            71,
            "Stage 2: instruction tuning with explicit output-format control",
            fontName="Helvetica",
            fontSize=10,
            fillColor=colors.HexColor("#40566b"),
            textAnchor="middle",
        )
    )
    return drawing


def make_llava15_architecture_diagram() -> Drawing:
    drawing = Drawing(510, 285)

    def box(x, y, w, h, label, fill="#e8f1fa", size=9, stroke="#355070"):
        drawing.add(
            Rect(
                x,
                y,
                w,
                h,
                rx=8,
                ry=8,
                strokeColor=colors.HexColor(stroke),
                fillColor=colors.HexColor(fill),
                strokeWidth=1.1,
            )
        )
        lines = label.split("\n")
        for i, line in enumerate(lines):
            drawing.add(
                String(
                    x + w / 2,
                    y + h / 2 + (len(lines) - 1 - 2 * i) * 5,
                    line,
                    fontName="Helvetica",
                    fontSize=size,
                    fillColor=colors.HexColor("#1f2d3d"),
                    textAnchor="middle",
                )
            )

    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.3))
        drawing.add(Line(x2 - 6, y2 + 3, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.3))
        drawing.add(Line(x2 - 6, y2 - 3, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.3))

    drawing.add(
        String(
            255,
            266,
            "LLaVA-1.5 Model Architecture / Single-Sample Data Flow",
            fontName="Helvetica-Bold",
            fontSize=13,
            fillColor=colors.HexColor("#17324d"),
            textAnchor="middle",
        )
    )

    box(10, 142, 75, 54, "Raw Sample\nimage path +\nconversations", "#eef2ff")

    box(100, 192, 100, 54, "Text Branch\npreprocess_multimodal\nmove <image> to front", "#e7f5ef", size=8.3)
    box(215, 192, 90, 54, "Template +\ntokenizer_image_token\ninput_ids [1, T]", "#ddf4ea", size=8.1)

    box(100, 86, 100, 54, "Image Branch\nload PIL + pad/resize\nnormalize", "#e9f1fb", size=8.3)
    box(215, 86, 90, 54, "CLIP ViT-L/14-336\npatch feats\n[1, 576, 1024]", "#d9eaf7", size=8.0)
    box(320, 86, 90, 54, "MLP Projector\nLinear -> GELU -> Linear\n[1, 576, 5120]", "#f8e6b8", size=7.8)

    box(320, 172, 90, 74, "Fusion in\nprepare_inputs_labels_for_multimodal\nreplace IMAGE_TOKEN_INDEX\nnew_input_embeds [1, S, 5120]", "#fde7dd", size=7.2)
    box(420, 148, 80, 98, "Vicuna / LLaMA\nDecoder Blocks\n-> lm_head\nlogits [1, S, V]\nassistant-only loss", "#f7d9d9", size=7.7)

    box(10, 20, 295, 44, "HD / anyres note: select_best_resolution -> tile -> optional unpad/newline -> concat local + global tokens", "#f4f0ff", size=7.6)
    box(320, 20, 180, 44, "Mask Rule: system / user / image tokens = -100", "#fff1db", size=8.2)

    arrow(85, 169, 100, 219)
    arrow(85, 169, 100, 113)
    arrow(200, 219, 215, 219)
    arrow(200, 113, 215, 113)
    arrow(305, 219, 320, 209)
    arrow(305, 113, 320, 113)
    arrow(365, 140, 365, 172)
    arrow(410, 209, 420, 209)

    drawing.add(Line(200, 42, 150, 42, strokeColor=colors.HexColor("#7b61aa"), strokeWidth=1.1))
    drawing.add(Line(150, 42, 150, 86, strokeColor=colors.HexColor("#7b61aa"), strokeWidth=1.1))
    drawing.add(
        String(
            190,
            55,
            "optional high-resolution branch",
            fontName="Helvetica",
            fontSize=8.5,
            fillColor=colors.HexColor("#6b4fa3"),
            textAnchor="middle",
        )
    )
    drawing.add(Line(410, 42, 365, 42, strokeColor=colors.HexColor("#a46d1f"), strokeWidth=1.0))
    drawing.add(Line(365, 42, 365, 172, strokeColor=colors.HexColor("#a46d1f"), strokeWidth=1.0))
    drawing.add(
        String(
            430,
            55,
            "mask is attached here",
            fontName="Helvetica",
            fontSize=8.2,
            fillColor=colors.HexColor("#8a5a12"),
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            365,
            156,
            "S = T - 1 + N_v",
            fontName="Helvetica-Bold",
            fontSize=9,
            fillColor=colors.HexColor("#7a2e2e"),
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            365,
            145,
            "base 336: N_v = 576",
            fontName="Helvetica",
            fontSize=8.2,
            fillColor=colors.HexColor("#7a2e2e"),
            textAnchor="middle",
        )
    )
    return drawing


def parse_markdown(text: str, styles: dict) -> list:
    story = []
    paragraph_lines = []
    in_code = False
    code_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines).strip()
            if paragraph:
                story.append(Paragraph(format_inline(paragraph), styles["body"]))
            paragraph_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        if line.strip() == "{{LLAVA15_PIPELINE_DIAGRAM}}":
            flush_paragraph()
            story.append(make_llava15_pipeline_diagram())
            story.append(
                Paragraph(
                    "图 3  根据论文、附录与官方训练脚本整理的 LLaVA-1.5 数据流与训练流程示意图。",
                    styles["caption"],
                )
            )
            continue

        if line.strip() == "{{LLAVA15_ARCHITECTURE_DIAGRAM}}":
            flush_paragraph()
            story.append(make_llava15_architecture_diagram())
            story.append(
                Paragraph(
                    "图 4  根据官方实现整理的 LLaVA-1.5 单样本模型架构与张量流转示意图。",
                    styles["caption"],
                )
            )
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            flush_paragraph()
            caption, rel_path = image_match.groups()
            path = (ROOT / rel_path).resolve()
            if path.exists():
                img = scaled_image(path, max_width=160 * mm, max_height=96 * mm)
                story.append(img)
                story.append(Paragraph(format_inline(caption), styles["caption"]))
            else:
                story.append(Paragraph(format_inline(f"[缺失图片] {caption} ({rel_path})"), styles["caption"]))
            continue

        if line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(format_inline(line[2:]), styles["title"]))
            continue

        if line.startswith("## "):
            flush_paragraph()
            story.append(Spacer(1, 4))
            story.append(Paragraph(format_inline(line[3:]), styles["h1"]))
            continue

        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(format_inline(line[4:]), styles["h2"]))
            continue

        if line.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(f"• {format_inline(line[2:])}", styles["bullet"]))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawRightString(doc.pagesize[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(markdown_path: Path, output_path: Path) -> None:
    register_fonts()
    styles = build_styles()
    text = markdown_path.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="LLaVA-1.5 论文精读报告",
        author="OpenAI Codex",
    )

    story = parse_markdown(text, styles)
    story.insert(
        1,
        Paragraph(
            "阅读报告生成时间：2026-04-08 | 内容来源：论文正文、附录、官方 README / 脚本与报告整理",
            styles["small"],
        ),
    )
    story.insert(2, Spacer(1, 4))
    story.insert(3, HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#d0d5dd")))
    story.insert(4, Spacer(1, 8))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def make_html_diagram() -> str:
    return """
<div class="diagram-wrap">
  <svg viewBox="0 0 980 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="LLaVA-1.5 pipeline diagram">
    <style>
      .box { fill: #e8f1fa; stroke: #355070; stroke-width: 2; rx: 14; ry: 14; }
      .txt { font: 18px sans-serif; fill: #1f2d3d; text-anchor: middle; }
      .sub { font: 16px sans-serif; fill: #40566b; text-anchor: middle; }
      .title { font: 700 24px sans-serif; fill: #17324d; text-anchor: middle; }
      .arrow { stroke: #355070; stroke-width: 2.5; fill: none; marker-end: url(#m); }
    </style>
    <defs>
      <marker id="m" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
        <path d="M0,0 L9,3 L0,6 Z" fill="#355070"/>
      </marker>
    </defs>
    <text x="490" y="32" class="title">LLaVA-1.5 Training / Data Flow</text>
    <rect class="box" x="20" y="70" width="180" height="64" fill="#dbe7c9"/>
    <rect class="box" x="250" y="70" width="150" height="64" fill="#d9eaf7"/>
    <rect class="box" x="450" y="70" width="160" height="64" fill="#f8e6b8"/>
    <rect class="box" x="660" y="70" width="160" height="64" fill="#f7d9d9"/>
    <text x="110" y="98" class="txt">558K</text><text x="110" y="120" class="txt">Image-Text Pairs</text>
    <text x="325" y="98" class="txt">CLIP ViT-L/14</text><text x="325" y="120" class="txt">336px</text>
    <text x="530" y="98" class="txt">MLP Projector</text><text x="530" y="120" class="txt">mlp2x_gelu</text>
    <text x="740" y="98" class="txt">Vicuna v1.5</text><text x="740" y="120" class="txt">7B / 13B</text>
    <line class="arrow" x1="200" y1="102" x2="250" y2="102"/>
    <line class="arrow" x1="400" y1="102" x2="450" y2="102"/>
    <line class="arrow" x1="610" y1="102" x2="660" y2="102"/>

    <rect class="box" x="20" y="180" width="220" height="78" fill="#ece8f6"/>
    <rect class="box" x="290" y="180" width="180" height="78" fill="#eaf4e2"/>
    <rect class="box" x="520" y="180" width="180" height="78" fill="#e6f5f1"/>
    <rect class="box" x="750" y="180" width="180" height="78" fill="#fde7dd"/>
    <text x="130" y="208" class="txt">665K Instruct Mix</text><text x="130" y="230" class="txt">LLaVA / VQA / OCR /</text><text x="130" y="252" class="txt">Region / ShareGPT</text>
    <text x="380" y="218" class="txt">Response Format</text><text x="380" y="240" class="txt">Prompts</text>
    <text x="610" y="218" class="txt">Conversation JSON</text><text x="610" y="240" class="txt">image + conversations</text>
    <text x="840" y="218" class="txt">Visual Instruction</text><text x="840" y="240" class="txt">Tuning</text>
    <line class="arrow" x1="240" y1="219" x2="290" y2="219"/>
    <line class="arrow" x1="470" y1="219" x2="520" y2="219"/>
    <line class="arrow" x1="700" y1="219" x2="750" y2="219"/>
    <line x1="530" y1="134" x2="530" y2="178" stroke="#355070" stroke-width="2"/>
    <text x="490" y="162" class="sub">Stage 1: alignment pretraining</text>
    <text x="490" y="292" class="sub">Stage 2: instruction tuning with explicit output-format control</text>
  </svg>
  <p class="diagram-caption">图 3 根据论文、附录与官方训练脚本整理的 LLaVA-1.5 数据流与训练流程示意图。</p>
</div>
"""


def make_html_architecture_diagram() -> str:
    return """
<div class="diagram-wrap">
  <svg viewBox="0 0 1180 560" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="LLaVA-1.5 architecture and tensor flow">
    <style>
      .box { stroke: #355070; stroke-width: 2; rx: 14; ry: 14; }
      .txt { font: 17px sans-serif; fill: #1f2d3d; text-anchor: middle; }
      .small { font: 14px sans-serif; fill: #1f2d3d; text-anchor: middle; }
      .hint { font: 14px sans-serif; fill: #5a6472; text-anchor: middle; }
      .accent { font: 700 16px sans-serif; fill: #7a2e2e; text-anchor: middle; }
      .title { font: 700 24px sans-serif; fill: #17324d; text-anchor: middle; }
      .arrow { stroke: #355070; stroke-width: 2.5; fill: none; marker-end: url(#m2); }
    </style>
    <defs>
      <marker id="m2" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
        <path d="M0,0 L9,3 L0,6 Z" fill="#355070"/>
      </marker>
    </defs>
    <text x="590" y="34" class="title">LLaVA-1.5 Model Architecture / Single-Sample Data Flow</text>

    <rect class="box" x="20" y="180" width="170" height="92" fill="#eef2ff"/>
    <text x="105" y="215" class="txt">Raw Sample</text>
    <text x="105" y="243" class="txt">image path +</text>
    <text x="105" y="271" class="txt">conversations</text>

    <rect class="box" x="245" y="80" width="210" height="92" fill="#e7f5ef"/>
    <text x="350" y="112" class="txt">Text Branch</text>
    <text x="350" y="140" class="small">preprocess_multimodal</text>
    <text x="350" y="164" class="small">move &lt;image&gt; to front</text>

    <rect class="box" x="500" y="80" width="220" height="92" fill="#ddf4ea"/>
    <text x="610" y="108" class="small">Conversation Template +</text>
    <text x="610" y="136" class="small">tokenizer_image_token</text>
    <text x="610" y="164" class="small">input_ids [1, T]</text>

    <rect class="box" x="245" y="300" width="210" height="92" fill="#e9f1fb"/>
    <text x="350" y="330" class="txt">Image Branch</text>
    <text x="350" y="358" class="small">load PIL + pad/resize</text>
    <text x="350" y="382" class="small">normalize</text>

    <rect class="box" x="500" y="300" width="220" height="92" fill="#d9eaf7"/>
    <text x="610" y="328" class="small">CLIP ViT-L/14-336</text>
    <text x="610" y="356" class="small">select_layer=-2</text>
    <text x="610" y="380" class="small">patch feats [1, 576, 1024]</text>

    <rect class="box" x="765" y="300" width="190" height="92" fill="#f8e6b8"/>
    <text x="860" y="328" class="small">MLP Projector</text>
    <text x="860" y="356" class="small">Linear -&gt; GELU -&gt; Linear</text>
    <text x="860" y="380" class="small">[1, 576, 5120] (13B)</text>

    <rect class="box" x="765" y="98" width="215" height="136" fill="#fde7dd"/>
    <text x="872" y="128" class="small">Fusion in</text>
    <text x="872" y="152" class="small">prepare_inputs_labels_for_multimodal</text>
    <text x="872" y="180" class="small">embed text + replace</text>
    <text x="872" y="204" class="small">IMAGE_TOKEN_INDEX</text>
    <text x="872" y="228" class="small">new_input_embeds [1, S, 5120]</text>

    <rect class="box" x="20" y="432" width="435" height="64" fill="#f4f0ff"/>
    <text x="238" y="458" class="small">HD / anyres note: select_best_resolution -&gt; tile -&gt; optional unpad/newline -&gt; concat local + global tokens</text>

    <rect class="box" x="765" y="432" width="215" height="64" fill="#fff1db"/>
    <text x="872" y="458" class="small">Mask Rule</text>
    <text x="872" y="482" class="small">system / user / image tokens = -100</text>

    <rect class="box" x="1015" y="98" width="145" height="136" fill="#f7d9d9"/>
    <text x="1088" y="132" class="small">Vicuna / LLaMA</text>
    <text x="1088" y="160" class="small">Decoder Blocks</text>
    <text x="1088" y="188" class="small">x40 for 13B</text>
    <text x="1088" y="212" class="small">x32 for 7B</text>

    <rect class="box" x="1015" y="268" width="145" height="92" fill="#ece8f6"/>
    <text x="1088" y="304" class="small">LM Head</text>
    <text x="1088" y="332" class="small">logits</text>
    <text x="1088" y="356" class="small">[1, S, V]</text>

    <rect class="box" x="1015" y="392" width="145" height="104" fill="#eaf4e2"/>
    <text x="1088" y="428" class="small">Causal LM Loss</text>
    <text x="1088" y="456" class="small">assistant</text>
    <text x="1088" y="480" class="small">response only</text>

    <line class="arrow" x1="190" y1="226" x2="245" y2="126"/>
    <line class="arrow" x1="190" y1="226" x2="245" y2="346"/>
    <line class="arrow" x1="455" y1="126" x2="500" y2="126"/>
    <line class="arrow" x1="455" y1="346" x2="500" y2="346"/>
    <line class="arrow" x1="720" y1="346" x2="765" y2="346"/>
    <line class="arrow" x1="720" y1="126" x2="765" y2="166"/>
    <line class="arrow" x1="860" y1="300" x2="860" y2="234"/>
    <line class="arrow" x1="980" y1="166" x2="1015" y2="166"/>
    <line class="arrow" x1="1088" y1="234" x2="1088" y2="268"/>
    <line class="arrow" x1="1088" y1="360" x2="1088" y2="392"/>

    <line x1="455" y1="464" x2="860" y2="464" stroke="#7b61aa" stroke-width="2"/>
    <line x1="860" y1="464" x2="860" y2="392" stroke="#7b61aa" stroke-width="2"/>
    <text x="650" y="452" class="hint">optional high-resolution branch</text>

    <line x1="980" y1="464" x2="930" y2="464" stroke="#a46d1f" stroke-width="2"/>
    <line x1="930" y1="464" x2="930" y2="234" stroke="#a46d1f" stroke-width="2"/>
    <text x="950" y="452" class="hint">mask is attached here</text>

    <text x="872" y="80" class="accent">S = T - 1 + N_v</text>
    <text x="872" y="58" class="hint">base 336: N_v = 576</text>
  </svg>
  <p class="diagram-caption">图 4 根据官方实现整理的 LLaVA-1.5 单样本模型架构与张量流转示意图。</p>
</div>
"""


def build_html(markdown_path: Path, output_path: Path) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    text = text.replace("{{LLAVA15_PIPELINE_DIAGRAM}}", make_html_diagram())
    text = text.replace("{{LLAVA15_ARCHITECTURE_DIAGRAM}}", make_html_architecture_diagram())
    body = md.markdown(
        text,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html5",
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLaVA-1.5 阅读报告</title>
  <style>
    :root {{
      --fg: #1f2937;
      --muted: #5b6472;
      --accent: #17324d;
      --border: #d8dee6;
      --bg: #fbfcfd;
      --codebg: #f4f6f8;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: "Noto Serif CJK SC", "Source Han Serif SC", "PingFang SC", "Microsoft YaHei", serif;
      line-height: 1.8;
    }}
    main {{
      max-width: 900px;
      margin: 0 auto;
      padding: 32px 24px 64px;
    }}
    h1, h2, h3 {{
      color: var(--accent);
      line-height: 1.4;
    }}
    h1 {{ font-size: 2.1rem; margin-top: 0; }}
    h2 {{ margin-top: 2.2rem; border-top: 1px solid var(--border); padding-top: 1.4rem; }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 1rem auto 0.4rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: white;
    }}
    p, li {{ font-size: 1rem; }}
    code {{
      background: var(--codebg);
      padding: 0.12rem 0.32rem;
      border-radius: 4px;
      font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 0.92em;
    }}
    pre {{
      background: var(--codebg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      overflow-x: auto;
    }}
    pre code {{ background: transparent; padding: 0; }}
    .diagram-wrap {{
      margin: 1rem 0 1.25rem;
      padding: 12px;
      background: white;
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    .diagram-caption {{
      margin: 0.4rem 0 0;
      text-align: center;
      color: var(--muted);
      font-size: 0.92rem;
    }}
  </style>
</head>
<body>
  <main>
    {body}
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> int:
    markdown_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else MARKDOWN_PATH
    pdf_output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else PDF_OUTPUT_PATH
    html_output_path = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else HTML_OUTPUT_PATH
    build_pdf(markdown_path, pdf_output_path)
    build_html(markdown_path, html_output_path)
    print(pdf_output_path)
    print(html_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
