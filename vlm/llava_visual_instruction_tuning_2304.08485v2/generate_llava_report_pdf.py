from __future__ import annotations

import html
import io
import re
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
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
MARKDOWN_PATH = ROOT / "llava_visual_instruction_tuning_reading_report.md"
OUTPUT_PATH = ROOT / "llava_visual_instruction_tuning_reading_report.pdf"


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
            textColor=colors.HexColor("#1c355e"),
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
            textColor=colors.HexColor("#19466f"),
        ),
        "h3": ParagraphStyle(
            "H3ZH",
            parent=styles["Heading3"],
            fontName="STSong-Light",
            fontSize=12.5,
            leading=17,
            spaceBefore=6,
            spaceAfter=5,
            textColor=colors.HexColor("#265c87"),
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
        # Re-encode through Pillow to avoid viewer-specific issues with source PNG/JPG variants.
        save_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
        if save_format == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format=save_format, quality=92)
        buf.seek(0)

    flowable = Image(buf, width=target_size[0], height=target_size[1])
    flowable._img_buffer = buf
    return flowable


def make_architecture_diagram() -> Drawing:
    drawing = Drawing(480, 180)

    def box(x, y, w, h, label, fill):
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
        drawing.add(
            String(
                x + w / 2,
                y + h / 2,
                label,
                fontName="Helvetica",
                fontSize=11,
                fillColor=colors.HexColor("#1f2d3d"),
                textAnchor="middle",
            )
        )

    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.5))
        drawing.add(Line(x2 - 6, y2 + 3, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.5))
        drawing.add(Line(x2 - 6, y2 - 3, x2, y2, strokeColor=colors.HexColor("#355070"), strokeWidth=1.5))

    box(20, 95, 90, 40, "Image", "#d9eaf7")
    box(135, 95, 100, 40, "CLIP ViT-L/14", "#dbe7c9")
    box(260, 95, 105, 40, "Linear Projector", "#f8e6b8")
    box(390, 95, 70, 40, "Vicuna", "#f7d9d9")

    box(20, 25, 130, 40, "Caption + Boxes", "#ece8f6")
    box(175, 25, 95, 40, "GPT-4 Teacher", "#f6e7ee")
    box(295, 25, 165, 40, "Instruct Data: conv / detail / reason", "#e1f0ea")

    arrow(110, 115, 135, 115)
    arrow(235, 115, 260, 115)
    arrow(365, 115, 390, 115)

    arrow(150, 45, 175, 45)
    arrow(270, 45, 295, 45)
    drawing.add(Line(377, 65, 377, 92, strokeColor=colors.HexColor("#355070"), strokeWidth=1.2))
    drawing.add(Line(377, 92, 390, 92, strokeColor=colors.HexColor("#355070"), strokeWidth=1.2))

    drawing.add(
        String(
            240,
            160,
            "LLaVA Pipeline (redrawn from the paper)",
            fontName="Helvetica-Bold",
            fontSize=13,
            fillColor=colors.HexColor("#1c355e"),
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            240,
            145,
            "Stage 1: feature alignment   |   Stage 2: visual instruction tuning",
            fontName="Helvetica",
            fontSize=10,
            fillColor=colors.HexColor("#40566b"),
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

        if line.strip() == "{{LLAVA_ARCH_DIAGRAM}}":
            flush_paragraph()
            story.append(make_architecture_diagram())
            story.append(
                Paragraph("图 3a  根据论文逻辑重绘的 LLaVA 数据与训练流程示意图。", styles["caption"])
            )
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            flush_paragraph()
            caption, rel_path = image_match.groups()
            path = (ROOT / rel_path).resolve()
            if path.exists():
                img = scaled_image(path, max_width=160 * mm, max_height=95 * mm)
                story.append(img)
                story.append(Paragraph(format_inline(caption), styles["caption"]))
            else:
                story.append(
                    Paragraph(
                        format_inline(f"[缺失图片] {caption} ({rel_path})"),
                        styles["caption"],
                    )
                )
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
        title="LLaVA 论文精读报告",
        author="OpenAI Codex",
    )

    story = parse_markdown(text, styles)
    story.insert(
        1,
        Paragraph(
            "阅读报告生成时间：2026-04-07 | 内容来源：论文正文、附录、公开页面与报告整理",
            styles["small"],
        ),
    )
    story.insert(2, Spacer(1, 4))
    story.insert(3, HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#d0d5dd")))
    story.insert(4, Spacer(1, 8))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> int:
    markdown_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else MARKDOWN_PATH
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else OUTPUT_PATH
    build_pdf(markdown_path, output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
