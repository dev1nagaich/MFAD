"""
MFAD — Multi-Factor AI Deepfake Detection System
report/template.py

ReportLab PDF layout engine for the forensic report.
Maps to §8 Legal Certification + §9–§10 Analyst Declaration of reference report DFA-2025-TC-00471.

Design: Dark forensic aesthetic — deep navy + electric cyan accents.
Every section maps directly to a measurement or finding from the reference report.
"""

import io
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, HRFlowable, Image,
    NextPageTemplate, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.pdfgen import canvas as pdfgen_canvas
from reportlab.platypus.flowables import BalancedColumns


# ─────────────────────────────────────────────
#  DESIGN TOKENS — change here, changes everywhere
# ─────────────────────────────────────────────

class T:
    """Design token namespace."""
    # Colours
    NAVY        = colors.HexColor("#0A0E1A")   # page background feel (used in header/footer)
    NAVY_MID    = colors.HexColor("#0F1628")   # section header bg
    NAVY_LIGHT  = colors.HexColor("#1A2340")   # table row alt bg
    CYAN        = colors.HexColor("#00D4FF")   # accent / highlight
    CYAN_DIM    = colors.HexColor("#0099BB")   # secondary accent
    WHITE       = colors.HexColor("#FFFFFF")
    OFF_WHITE   = colors.HexColor("#E8EDF5")
    GREY_LIGHT  = colors.HexColor("#C5CDD8")
    GREY_MID    = colors.HexColor("#8899AA")
    RED_ALERT   = colors.HexColor("#FF3B5C")   # DEEPFAKE verdict
    ORANGE_WARN = colors.HexColor("#FF8C00")   # UNCERTAIN verdict
    GREEN_OK    = colors.HexColor("#00C47A")   # AUTHENTIC verdict
    BORDER      = colors.HexColor("#1E2D50")   # table border

    # Typography — Helvetica family (always available in ReportLab)
    FONT_BODY   = "Helvetica"
    FONT_BOLD   = "Helvetica-Bold"
    FONT_OBLIQ  = "Helvetica-Oblique"

    # Sizes
    PAGE_W, PAGE_H = A4
    MARGIN_L    = 18 * mm
    MARGIN_R    = 18 * mm
    MARGIN_T    = 22 * mm
    MARGIN_B    = 22 * mm
    CONTENT_W   = PAGE_W - MARGIN_L - MARGIN_R


# ─────────────────────────────────────────────
#  PARAGRAPH STYLES
# ─────────────────────────────────────────────

def build_styles():
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName=T.FONT_BOLD,
        fontSize=28,
        textColor=T.WHITE,
        leading=34,
        alignment=TA_LEFT,
        spaceAfter=4 * mm,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontName=T.FONT_BODY,
        fontSize=11,
        textColor=T.CYAN,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=2 * mm,
        tracking=2,
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta",
        fontName=T.FONT_BODY,
        fontSize=9,
        textColor=T.GREY_LIGHT,
        leading=14,
        alignment=TA_LEFT,
    )
    styles["section_heading"] = ParagraphStyle(
        "section_heading",
        fontName=T.FONT_BOLD,
        fontSize=10,
        textColor=T.CYAN,
        leading=14,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
        tracking=3,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName=T.FONT_BODY,
        fontSize=8.5,
        textColor=colors.HexColor("#1A1A2E"),
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=2 * mm,
    )
    styles["body_white"] = ParagraphStyle(
        "body_white",
        fontName=T.FONT_BODY,
        fontSize=8.5,
        textColor=T.OFF_WHITE,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=2 * mm,
    )
    styles["label"] = ParagraphStyle(
        "label",
        fontName=T.FONT_BOLD,
        fontSize=7.5,
        textColor=T.GREY_MID,
        leading=11,
        tracking=1,
    )
    styles["value"] = ParagraphStyle(
        "value",
        fontName=T.FONT_BODY,
        fontSize=8.5,
        textColor=colors.HexColor("#1A1A2E"),
        leading=13,
    )
    styles["mono"] = ParagraphStyle(
        "mono",
        fontName="Courier",
        fontSize=7.5,
        textColor=T.CYAN_DIM,
        leading=11,
    )
    styles["verdict_fake"] = ParagraphStyle(
        "verdict_fake",
        fontName=T.FONT_BOLD,
        fontSize=22,
        textColor=T.RED_ALERT,
        leading=26,
        alignment=TA_CENTER,
    )
    styles["verdict_authentic"] = ParagraphStyle(
        "verdict_authentic",
        fontName=T.FONT_BOLD,
        fontSize=22,
        textColor=T.GREEN_OK,
        leading=26,
        alignment=TA_CENTER,
    )
    styles["verdict_uncertain"] = ParagraphStyle(
        "verdict_uncertain",
        fontName=T.FONT_BOLD,
        fontSize=22,
        textColor=T.ORANGE_WARN,
        leading=26,
        alignment=TA_CENTER,
    )
    styles["table_header"] = ParagraphStyle(
        "table_header",
        fontName=T.FONT_BOLD,
        fontSize=7.5,
        textColor=T.CYAN,
        leading=11,
        tracking=1,
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell",
        fontName=T.FONT_BODY,
        fontSize=8,
        textColor=colors.HexColor("#1A1A2E"),
        leading=12,
    )
    styles["table_cell_mono"] = ParagraphStyle(
        "table_cell_mono",
        fontName="Courier",
        fontSize=7.5,
        textColor=colors.HexColor("#1A1A2E"),
        leading=11,
    )
    styles["footer_text"] = ParagraphStyle(
        "footer_text",
        fontName=T.FONT_BODY,
        fontSize=7,
        textColor=T.GREY_MID,
        leading=10,
    )
    styles["compliance_item"] = ParagraphStyle(
        "compliance_item",
        fontName=T.FONT_BODY,
        fontSize=8,
        textColor=colors.HexColor("#1A1A2E"),
        leading=13,
        leftIndent=10,
        spaceAfter=1 * mm,
    )

    return styles


# ─────────────────────────────────────────────
#  CUSTOM FLOWABLES
# ─────────────────────────────────────────────

class CyanRule(Flowable):
    """A full-width horizontal rule in cyan — used as section divider."""
    def __init__(self, width=None, thickness=1.5, color=None):
        super().__init__()
        self.width = width or T.CONTENT_W
        self.thickness = thickness
        self.color = color or T.CYAN
        self.height = self.thickness + 2

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness / 2, self.width, self.thickness / 2)


class SectionHeader(Flowable):
    """
    Dark navy pill-shaped section header bar with cyan left accent stripe.
    Looks like a dashboard panel header.
    """
    def __init__(self, text, width=None, ref=""):
        super().__init__()
        self.text = text
        self.ref = ref
        self.width = width or T.CONTENT_W
        self.height = 10 * mm

    def draw(self):
        c = self.canv
        w, h = self.width, self.height

        # Background bar
        c.setFillColor(T.NAVY_MID)
        c.roundRect(0, 0, w, h, 2, fill=1, stroke=0)

        # Left cyan accent stripe
        c.setFillColor(T.CYAN)
        c.rect(0, 0, 3, h, fill=1, stroke=0)

        # Section title text
        c.setFillColor(T.WHITE)
        c.setFont(T.FONT_BOLD, 9)
        c.drawString(8, h / 2 - 3.5, self.text.upper())

        # Right-aligned reference tag
        if self.ref:
            c.setFillColor(T.CYAN)
            c.setFont(T.FONT_BODY, 7)
            c.drawRightString(w - 6, h / 2 - 3, self.ref)


class ScoreBar(Flowable):
    """
    Visual score bar for a single module.
    Shows a dark track with a coloured fill bar and numeric label.
    Used in §6 per-module confidence display.
    """
    def __init__(self, label, score, width=None):
        super().__init__()
        self.label = label
        self.score = float(score)
        self.width = width or T.CONTENT_W
        self.height = 8 * mm

    def _score_color(self, s):
        if s >= 0.70:
            return T.RED_ALERT
        elif s >= 0.35:
            return T.ORANGE_WARN
        else:
            return T.GREEN_OK

    def draw(self):
        c = self.canv
        label_w = 52 * mm
        bar_w = self.width - label_w - 20
        bar_h = 4.5
        y_center = self.height / 2

        # Label
        c.setFillColor(colors.HexColor("#333355"))
        c.setFont(T.FONT_BODY, 8)
        c.drawString(0, y_center - 3, self.label)

        # Track (empty bar)
        c.setFillColor(colors.HexColor("#D8DCE8"))
        c.roundRect(label_w, y_center - bar_h / 2, bar_w, bar_h, 2, fill=1, stroke=0)

        # Fill bar
        fill_w = bar_w * self.score
        c.setFillColor(self._score_color(self.score))
        c.roundRect(label_w, y_center - bar_h / 2, max(fill_w, 2), bar_h, 2, fill=1, stroke=0)

        # Numeric score label
        c.setFillColor(colors.HexColor("#1A1A2E"))
        c.setFont(T.FONT_BOLD, 8)
        pct = f"{self.score * 100:.1f}%"
        c.drawRightString(self.width, y_center - 3, pct)


class VerdictBadge(Flowable):
    """
    Large centred verdict badge — DEEPFAKE / AUTHENTIC / UNCERTAIN.
    Maps to the primary decision output of BayesianFusion (§6).
    """
    def __init__(self, verdict, final_score, confidence_interval, width=None):
        super().__init__()
        self.verdict = verdict.upper()
        self.score = float(final_score)
        self.ci = confidence_interval  # [lower, upper]
        self.width = width or T.CONTENT_W
        self.height = 38 * mm

    def _colours(self):
        if self.verdict == "DEEPFAKE":
            return T.RED_ALERT, colors.HexColor("#2A0A10")
        elif self.verdict == "AUTHENTIC":
            return T.GREEN_OK, colors.HexColor("#0A2A1A")
        else:
            return T.ORANGE_WARN, colors.HexColor("#2A1A00")

    def draw(self):
        c = self.canv
        accent, bg = self._colours()
        w, h = self.width, self.height

        # Outer rounded rect (bg)
        c.setFillColor(bg)
        c.setStrokeColor(accent)
        c.setLineWidth(1.5)
        c.roundRect(0, 0, w, h, 6, fill=1, stroke=1)

        # Left thick accent bar
        c.setFillColor(accent)
        c.rect(0, 0, 5, h, fill=1, stroke=0)

        # Verdict text
        c.setFillColor(accent)
        c.setFont(T.FONT_BOLD, 24)
        c.drawCentredString(w / 2, h / 2 + 4, self.verdict)

        # Score
        c.setFillColor(T.WHITE)
        c.setFont(T.FONT_BOLD, 13)
        score_txt = f"DeepFake Prediction Score:  {self.score * 100:.1f}%"
        c.drawCentredString(w / 2, h / 2 - 10, score_txt)

        # Confidence interval
        if self.ci and len(self.ci) == 2:
            c.setFillColor(T.GREY_LIGHT)
            c.setFont(T.FONT_BODY, 8)
            ci_txt = f"95% Confidence Interval:  [{self.ci[0]*100:.1f}% — {self.ci[1]*100:.1f}%]"
            c.drawCentredString(w / 2, h / 2 - 22, ci_txt)


class HeatmapEmbed(Flowable):
    """
    Embeds the Grad-CAM heatmap image with a caption and border.
    Maps to §5.4 VLM Explainability Heat-Map.
    If path doesn't exist, renders a placeholder.
    """
    def __init__(self, path, caption="", width=None):
        super().__init__()
        self.path = path
        self.caption = caption
        self.img_w = width or (T.CONTENT_W * 0.55)
        self.img_h = self.img_w * 0.75
        self.height = self.img_h + 12 * mm

    def draw(self):
        c = self.canv
        x = (T.CONTENT_W - self.img_w) / 2
        y = 10 * mm

        # Border frame
        c.setStrokeColor(T.CYAN_DIM)
        c.setLineWidth(0.8)
        c.roundRect(x - 2, y - 2, self.img_w + 4, self.img_h + 4, 3, fill=0, stroke=1)

        # Image or placeholder
        if self.path and os.path.exists(self.path):
            c.drawImage(self.path, x, y, self.img_w, self.img_h, preserveAspectRatio=True)
        else:
            # Placeholder
            c.setFillColor(T.NAVY_LIGHT)
            c.rect(x, y, self.img_w, self.img_h, fill=1, stroke=0)
            c.setFillColor(T.GREY_MID)
            c.setFont(T.FONT_BODY, 8)
            c.drawCentredString(x + self.img_w / 2, y + self.img_h / 2 - 4, "[Grad-CAM Heatmap]")

        # Caption below
        if self.caption:
            c.setFillColor(T.GREY_MID)
            c.setFont(T.FONT_OBLIQ, 7.5)
            c.drawCentredString(T.CONTENT_W / 2, y - 6, self.caption)


class KeyValueBlock(Flowable):
    """
    Compact two-column key-value block for metadata fields.
    Used for chain-of-custody fields, EXIF fields, etc.
    """
    def __init__(self, pairs, width=None):
        super().__init__()
        self.pairs = pairs  # list of (label, value, flag?) tuples
        self.width = width or T.CONTENT_W
        self.row_h = 7 * mm
        self.height = len(pairs) * self.row_h

    def draw(self):
        c = self.canv
        col1 = self.width * 0.38
        col2 = self.width * 0.62

        for i, item in enumerate(self.pairs):
            label = item[0]
            value = str(item[1])
            flagged = item[2] if len(item) > 2 else False

            y = self.height - (i + 1) * self.row_h

            # Alternating row background
            if i % 2 == 0:
                c.setFillColor(colors.HexColor("#F4F6FB"))
                c.rect(0, y, self.width, self.row_h, fill=1, stroke=0)

            # Bottom border
            c.setStrokeColor(colors.HexColor("#E0E5EF"))
            c.setLineWidth(0.4)
            c.line(0, y, self.width, y)

            # Label
            c.setFillColor(T.GREY_MID)
            c.setFont(T.FONT_BOLD, 7.5)
            c.drawString(4, y + self.row_h / 2 - 3.5, label)

            # Value
            val_color = T.RED_ALERT if flagged else colors.HexColor("#1A1A2E")
            c.setFillColor(val_color)
            c.setFont("Courier" if flagged else T.FONT_BODY, 8)
            c.drawString(col1 + 4, y + self.row_h / 2 - 3.5, value)

            # Flag indicator dot
            if flagged:
                c.setFillColor(T.RED_ALERT)
                c.circle(self.width - 6, y + self.row_h / 2, 2.5, fill=1, stroke=0)


# ─────────────────────────────────────────────
#  PAGE TEMPLATES  (header / footer)
# ─────────────────────────────────────────────

def _draw_cover_page(c, doc):
    """Full-bleed navy cover page background."""
    w, h = A4
    c.saveState()

    # Full-page dark background
    c.setFillColor(T.NAVY)
    c.rect(0, 0, w, h, fill=1, stroke=0)

    # Diagonal cyan accent stripe (top-right corner design element)
    c.setFillColor(colors.HexColor("#001822"))
    p = c.beginPath()
    p.moveTo(w * 0.5, h)
    p.lineTo(w, h)
    p.lineTo(w, h * 0.55)
    p.close()
    c.drawPath(p, fill=1, stroke=0)

    # Top cyan bar
    c.setFillColor(T.CYAN)
    c.rect(0, h - 3, w, 3, fill=1, stroke=0)

    # Bottom thin cyan line
    c.setFillColor(T.CYAN_DIM)
    c.rect(0, 18 * mm, w, 0.8, fill=1, stroke=0)

    # Bottom footer text
    c.setFillColor(T.GREY_MID)
    c.setFont(T.FONT_BODY, 7)
    c.drawString(T.MARGIN_L, 12 * mm, "CONFIDENTIAL — FOR AUTHORISED FORENSIC USE ONLY")
    c.drawRightString(w - T.MARGIN_R, 12 * mm,
                      "Reference Standard: ISO/IEC 27037 · SWGDE · NIST SP 800-101r1")
    c.restoreState()


def _draw_inner_page(c, doc):
    """Header + footer for all inner pages."""
    w, h = A4
    c.saveState()

    # ── HEADER ──────────────────────────────
    # Dark navy top strip
    c.setFillColor(T.NAVY_MID)
    c.rect(0, h - T.MARGIN_T, w, T.MARGIN_T, fill=1, stroke=0)

    # Cyan top rule
    c.setFillColor(T.CYAN)
    c.rect(0, h - T.MARGIN_T, w, 2.5, fill=1, stroke=0)

    # Lab name left
    c.setFillColor(T.CYAN)
    c.setFont(T.FONT_BOLD, 8)
    c.drawString(T.MARGIN_L, h - T.MARGIN_T + 7, "MFAD FORENSIC AI LABORATORY")

    # Report ID right
    report_id = getattr(doc, "_report_id", "DFA-XXXX-TC-XXXXXX")
    c.setFillColor(T.GREY_LIGHT)
    c.setFont(T.FONT_BODY, 7.5)
    c.drawRightString(w - T.MARGIN_R, h - T.MARGIN_T + 7, f"Report ID: {report_id}")

    # ── FOOTER ──────────────────────────────
    # Thin cyan line
    c.setFillColor(T.CYAN_DIM)
    c.rect(T.MARGIN_L, T.MARGIN_B - 4, T.CONTENT_W, 0.8, fill=1, stroke=0)

    # Confidential left
    c.setFillColor(T.GREY_MID)
    c.setFont(T.FONT_BODY, 7)
    c.drawString(T.MARGIN_L, T.MARGIN_B - 12, "CONFIDENTIAL — FORENSIC EVIDENCE")

    # Page number centre
    c.setFillColor(T.GREY_MID)
    c.setFont(T.FONT_BODY, 7)
    c.drawCentredString(w / 2, T.MARGIN_B - 12, f"Page {doc.page}")

    # Date right
    c.setFillColor(T.GREY_MID)
    c.setFont(T.FONT_BODY, 7)
    c.drawRightString(w - T.MARGIN_R, T.MARGIN_B - 12,
                      datetime.utcnow().strftime("%Y-%m-%d UTC"))
    c.restoreState()


# ─────────────────────────────────────────────
#  TABLE HELPERS
# ─────────────────────────────────────────────

def _std_table_style(header_rows=1):
    """Standard table style used throughout the report."""
    style = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), T.NAVY_MID),
        ("TEXTCOLOR",  (0, 0), (-1, header_rows - 1), T.CYAN),
        ("FONTNAME",   (0, 0), (-1, header_rows - 1), T.FONT_BOLD),
        ("FONTSIZE",   (0, 0), (-1, header_rows - 1), 7.5),
        ("TOPPADDING", (0, 0), (-1, header_rows - 1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, header_rows - 1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),

        # Body rows
        ("FONTNAME",   (0, header_rows), (-1, -1), T.FONT_BODY),
        ("FONTSIZE",   (0, header_rows), (-1, -1), 8),
        ("TEXTCOLOR",  (0, header_rows), (-1, -1), colors.HexColor("#1A1A2E")),
        ("TOPPADDING", (0, header_rows), (-1, -1), 4),
        ("BOTTOMPADDING", (0, header_rows), (-1, -1), 4),

        # Alternating row shading
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
         [colors.HexColor("#F9FAFB"), colors.HexColor("#EEF1F8")]),

        # Grid lines
        ("GRID",       (0, 0), (-1, -1), 0.4, T.BORDER),
        ("LINEBELOW",  (0, 0), (-1, 0),  1.0, T.CYAN),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(style)


# ─────────────────────────────────────────────
#  SECTION BUILDERS
# ─────────────────────────────────────────────

def _build_cover(ctx, styles):
    """
    Cover page content — rendered on the full-bleed dark background.
    """
    story = []

    # Top spacer (push content down from the top bar)
    story.append(Spacer(1, 55 * mm))

    # Main title
    story.append(Paragraph("FORENSIC IMAGE<br/>AUTHENTICATION REPORT", styles["cover_title"]))
    story.append(Spacer(1, 3 * mm))
    story.append(CyanRule(thickness=2))
    story.append(Spacer(1, 5 * mm))

    # Subtitle / system name
    story.append(Paragraph(
        "MFAD — MULTI-FACTOR AI DEEPFAKE DETECTION SYSTEM",
        styles["cover_sub"]
    ))
    story.append(Spacer(1, 8 * mm))

    # Meta info block
    report_id  = ctx.get("report_id", "DFA-XXXX-TC-XXXXXX")
    generated  = ctx.get("generated_at", datetime.utcnow().isoformat())
    analyst    = ctx.get("analyst_name", "N/A")
    lab        = ctx.get("lab_accreditation", "N/A")
    image_path = ctx.get("image_path", "N/A")
    sha256     = ctx.get("hash_sha256", "N/A")

    meta_rows = [
        ["REPORT ID",       report_id],
        ["DATE GENERATED",  generated],
        ["ANALYST",         analyst],
        ["LABORATORY",      lab],
        ["SUBJECT IMAGE",   os.path.basename(str(image_path))],
        ["SHA-256 HASH",    sha256[:32] + "..." if len(str(sha256)) > 35 else sha256],
    ]

    cover_label_style = ParagraphStyle("cl", fontName=T.FONT_BOLD, fontSize=7,
                                        textColor=T.GREY_MID, leading=11, tracking=2)
    cover_value_style = ParagraphStyle("cv", fontName=T.FONT_BODY, fontSize=8.5,
                                        textColor=T.WHITE, leading=13)

    for label, value in meta_rows:
        row_data = [[
            Paragraph(label, cover_label_style),
            Paragraph(str(value), cover_value_style),
        ]]
        t = Table(row_data, colWidths=[T.CONTENT_W * 0.35, T.CONTENT_W * 0.65])
        t.setStyle(TableStyle([
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW",     (0, 0), (-1, -1), 0.4, colors.HexColor("#1E2D50")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(t)

    story.append(Spacer(1, 10 * mm))

    # Verdict preview badge (large, centred)
    decision = ctx.get("decision", "UNCERTAIN")
    final_score = ctx.get("final_score", 0.5)
    ci = ctx.get("confidence_interval", [0.0, 1.0])
    story.append(VerdictBadge(decision, final_score, ci))

    story.append(Spacer(1, 8 * mm))

    # Compliance standards line
    standards = ctx.get("compliance_standards",
                        ["ISO/IEC 27037:2012", "SWGDE 2023", "NIST SP 800-101r1"])
    std_text = "  ·  ".join(standards)
    std_style = ParagraphStyle("std", fontName=T.FONT_BODY, fontSize=7,
                                textColor=T.GREY_MID, alignment=TA_CENTER)
    story.append(Paragraph(std_text, std_style))

    story.append(PageBreak())
    return story


def _build_chain_of_custody(ctx, styles):
    """
    §1 — Chain of Custody & Image Provenance.
    Maps to PreprocessingAgent outputs.
    """
    story = []
    story.append(SectionHeader("§1  CHAIN OF CUSTODY & IMAGE PROVENANCE", ref="§1 + §5.6"))
    story.append(Spacer(1, 3 * mm))

    sha256 = ctx.get("hash_sha256", "N/A")
    pairs = [
        ("Image Path",          ctx.get("image_path", "N/A"),           False),
        ("SHA-256 Hash",        sha256,                                   False),
        ("Hash Verified",       str(ctx.get("hash_sha256_verified", False)), False),
        ("EXIF Camera Present", str(ctx.get("exif_camera_present", "N/A")),
                                not ctx.get("exif_camera_present", True)),
        ("ELA Chi² Score",      f"{ctx.get('ela_chi2', 0.0):.2f}",
                                float(ctx.get("ela_chi2", 0)) > 200),
        ("Thumbnail Mismatch",  str(ctx.get("thumbnail_mismatch", False)),
                                ctx.get("thumbnail_mismatch", False)),
        ("PRNU Absent",         str(ctx.get("prnu_absent", False)),
                                ctx.get("prnu_absent", False)),
        ("Software Tag",        str(ctx.get("software_tag", "")) or "—",  False),
        ("ICC Profile",         str(ctx.get("icc_profile", "")) or "—",   False),
        ("Face Bounding Box",   str(ctx.get("face_bbox", "N/A")),          False),
        ("Face Crop Path",      str(ctx.get("face_crop_path", "N/A")),     False),
        ("Normalized Path",     str(ctx.get("normalized_path", "N/A")),    False),
    ]
    story.append(KeyValueBlock(pairs))
    story.append(Spacer(1, 4 * mm))
    return story


def _build_per_module_findings(ctx, styles):
    """
    §5.1 – §5.6  Individual agent findings tables.
    Each sub-section maps to one agent's output keys.
    """
    story = []

    # ── §5.1 Geometry ─────────────────────────────────────────
    story.append(SectionHeader("§5.1  FACIAL GEOMETRY & LANDMARK ANALYSIS", ref="GeometryAgent"))
    story.append(Spacer(1, 2 * mm))

    geo_data = [
        ["Metric", "Measured Value", "Authentic Baseline", "Status"],
        ["Symmetry Index",
         f"{ctx.get('symmetry_index', 'N/A')}",
         "0.92 – 1.00",
         "⚠ ANOMALOUS" if float(ctx.get('symmetry_index', 1) or 1) < 0.92 else "✓ NORMAL"],
        ["Jaw Curvature (°)",
         f"{ctx.get('jaw_curvature_deg', 'N/A')}",
         "< 5°",
         "⚠ ANOMALOUS" if float(ctx.get('jaw_curvature_deg', 0) or 0) > 5 else "✓ NORMAL"],
        ["Ear Alignment (px)",
         f"{ctx.get('ear_alignment_px', 'N/A')}",
         "< 3 px",
         "⚠ ANOMALOUS" if float(ctx.get('ear_alignment_px', 0) or 0) > 3 else "✓ NORMAL"],
        ["Philtrum Length (mm)",
         f"{ctx.get('philtrum_length_mm', 'N/A')}",
         "11 – 18 mm",
         "—"],
        ["Inter-ocular Dist (px)",
         f"{ctx.get('interocular_dist_px', 'N/A')}",
         "58 – 72 px",
         "—"],
        ["Eye Aspect Ratio L/R",
         f"{ctx.get('eye_aspect_ratio_l','N/A')} / {ctx.get('eye_aspect_ratio_r','N/A')}",
         "0.28 – 0.34",
         "—"],
        ["Lip Thickness Ratio",
         f"{ctx.get('lip_thickness_ratio', 'N/A')}",
         "0.18 – 0.26",
         "—"],
        ["Neck-Face Boundary",
         str(ctx.get('neck_face_boundary', 'N/A')),
         "smooth",
         "⚠ ANOMALOUS" if str(ctx.get('neck_face_boundary', '')) == "sharp_edge" else "✓ NORMAL"],
        ["Anomaly Score",
         f"{ctx.get('geometry_anomaly_score', ctx.get('anomaly_score', 'N/A'))}",
         "< 0.35 = authentic",
         ""],
    ]
    col_w = [T.CONTENT_W * 0.36, T.CONTENT_W * 0.22, T.CONTENT_W * 0.25, T.CONTENT_W * 0.17]
    geo_table = Table(geo_data, colWidths=col_w)
    geo_table.setStyle(_std_table_style())
    # Colour status column
    for i, row in enumerate(geo_data[1:], 1):
        status = row[3]
        if "ANOMALOUS" in status:
            geo_table.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), T.RED_ALERT),
                                            ("FONTNAME",  (3, i), (3, i), T.FONT_BOLD)]))
        elif "NORMAL" in status:
            geo_table.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), T.GREEN_OK)]))
    story.append(geo_table)
    story.append(Spacer(1, 4 * mm))

    # ── §5.2 Frequency + GAN ──────────────────────────────────
    story.append(SectionHeader("§5.2  GAN ARTEFACT & FREQUENCY-DOMAIN ANALYSIS", ref="FrequencyAgent"))
    story.append(Spacer(1, 2 * mm))

    freq_data = [
        ["Metric", "Value", "Interpretation"],
        ["FFT Mid-Band Anomaly (16–50 cyc/px)",
         f"{ctx.get('fft_mid_anomaly_db', 'N/A')} dB",
         "GAN upsampling leaves +ve excess here"],
        ["FFT High-Band Anomaly (51–100 cyc/px)",
         f"{ctx.get('fft_high_anomaly_db', 'N/A')} dB",
         "p < 0.001 flag if positive"],
        ["FFT Ultra-High (>100 cyc/px)",
         f"{ctx.get('fft_ultrahigh_anomaly_db', 'N/A')} dB",
         "StyleGAN2 signature (+15.6 dB in reference)"],
        ["GAN Probability (EfficientNet-B4)",
         f"{float(ctx.get('gan_probability', 0))*100:.1f}%",
         "FaceForensics++ v3 fine-tuned classifier"],
        ["Upsampling Grid Detected",
         str(ctx.get('upsampling_grid_detected', 'N/A')),
         "4×4 px DCGAN transposed-conv artefact"],
        ["Frequency Anomaly Score",
         str(ctx.get('frequency_anomaly_score', ctx.get('freq_anomaly_score', 'N/A'))),
         "Combined FFT + GAN signal"],
    ]
    freq_table = Table(freq_data,
                       colWidths=[T.CONTENT_W * 0.42, T.CONTENT_W * 0.2, T.CONTENT_W * 0.38])
    freq_table.setStyle(_std_table_style())
    story.append(freq_table)
    story.append(Spacer(1, 4 * mm))

    # ── §5.3 Texture ──────────────────────────────────────────
    story.append(SectionHeader("§5.3  TEXTURE CONSISTENCY & SKIN-TONE MAPPING", ref="TextureAgent"))
    story.append(Spacer(1, 2 * mm))

    tex_data = [
        ["Zone Pair", "EMD Score", "Baseline", "Flag"],
        ["Forehead ↔ Cheek",
         f"{ctx.get('forehead_cheek_emd', 'N/A')}",
         "< 0.08", "⚠" if float(ctx.get('forehead_cheek_emd', 0) or 0) > 0.08 else "✓"],
        ["Cheek ↔ Jaw (Left)",
         f"{ctx.get('cheek_jaw_emd_l', 'N/A')}",
         "< 0.08", "⚠" if float(ctx.get('cheek_jaw_emd_l', 0) or 0) > 0.08 else "✓"],
        ["Cheek ↔ Jaw (Right)",
         f"{ctx.get('cheek_jaw_emd_r', 'N/A')}",
         "< 0.08", "⚠" if float(ctx.get('cheek_jaw_emd_r', 0) or 0) > 0.08 else "✓"],
        ["Periorbital ↔ Nasal Bridge",
         f"{ctx.get('periorbital_nasal_emd', 'N/A')}",
         "< 0.08", "⚠" if float(ctx.get('periorbital_nasal_emd', 0) or 0) > 0.08 else "✓"],
        ["Upper Lip ↔ Chin",
         f"{ctx.get('lip_chin_emd', 'N/A')}",
         "< 0.08", "⚠" if float(ctx.get('lip_chin_emd', 0) or 0) > 0.08 else "✓"],
        ["Neck ↔ Face",
         f"{ctx.get('neck_face_emd', 'N/A')}",
         "< 0.15 (primary seam indicator)",
         "⚠ SEAM" if float(ctx.get('neck_face_emd', 0) or 0) > 0.15 else "✓"],
        ["LBP Uniformity", f"{ctx.get('lbp_uniformity', 'N/A')}",
         "> 0.85", "⚠" if float(ctx.get('lbp_uniformity', 1) or 1) < 0.85 else "✓"],
        ["Seam Detected", str(ctx.get('seam_detected', 'N/A')), "False", ""],
    ]
    tex_table = Table(tex_data,
                      colWidths=[T.CONTENT_W*0.36, T.CONTENT_W*0.2, T.CONTENT_W*0.31, T.CONTENT_W*0.13])
    tex_table.setStyle(_std_table_style())
    story.append(tex_table)
    story.append(Spacer(1, 4 * mm))

    # ── §5.4 VLM / Heatmap ────────────────────────────────────
    story.append(SectionHeader("§5.4  EXPLAINABILITY HEAT-MAP ANALYSIS (VLM ATTENTION)", ref="VLMAgent"))
    story.append(Spacer(1, 2 * mm))

    vlm_data = [
        ["Field", "Value"],
        ["VLM Verdict",       str(ctx.get("vlm_verdict", "N/A"))],
        ["VLM Confidence",    f"{float(ctx.get('vlm_confidence', 0))*100:.1f}%"],
        ["Saliency Score",    str(ctx.get("saliency_score", "N/A"))],
        ["Zone GAN Prob.",    f"{float(ctx.get('zone_gan_probability', 0))*100:.1f}%"],
        ["High Activation",   ", ".join(ctx.get("high_activation_regions", []))  or "—"],
        ["Mid Activation",    ", ".join(ctx.get("medium_activation_regions", [])) or "—"],
        ["Low Activation",    ", ".join(ctx.get("low_activation_regions", []))   or "—"],
        ["VLM Caption",       str(ctx.get("vlm_caption", "N/A"))],
    ]
    vlm_table = Table(vlm_data, colWidths=[T.CONTENT_W * 0.30, T.CONTENT_W * 0.70])
    vlm_table.setStyle(_std_table_style())
    story.append(vlm_table)
    story.append(Spacer(1, 3 * mm))

    # Heatmap image embed
    heatmap_path = ctx.get("heatmap_path", "")
    story.append(HeatmapEmbed(
        path=str(heatmap_path),
        caption="Figure 1 — Grad-CAM activation overlay on EfficientNet-B4 (FF++ v3). "
                "RED = high activation (critical deepfake features). "
                "BLUE = secondary manipulation zone.",
    ))
    story.append(Spacer(1, 4 * mm))

    # ── §5.5 Biological ───────────────────────────────────────
    story.append(SectionHeader("§5.5  BIOLOGICAL PLAUSIBILITY ASSESSMENT", ref="BiologicalAgent"))
    story.append(Spacer(1, 2 * mm))

    bio_data = [
        ["Signal", "Measured", "Authentic Baseline", "Finding"],
        ["rPPG Cardiac SNR",
         str(ctx.get("rppg_snr", "N/A")),
         "> 0.45",
         "ABSENT" if float(ctx.get("rppg_snr", 1) or 1) < 0.45 else "PRESENT"],
        ["Corneal Highlight Deviation (°)",
         str(ctx.get("corneal_deviation_deg", "N/A")),
         "< 5°",
         "COMPOSITE ILLUMINATION" if float(ctx.get("corneal_deviation_deg", 0) or 0) > 5 else "CONSISTENT"],
        ["Perioral Micro-texture Variance",
         str(ctx.get("micro_texture_var", "N/A")),
         "~0.031 (mean)",
         "GAN SMOOTHED" if float(ctx.get("micro_texture_var", 0.031) or 0.031) < 0.020 else "NORMAL"],
        ["Vascular Pearson r",
         str(ctx.get("vascular_pearson_r", "N/A")),
         "> 0.88",
         "MISMATCH" if float(ctx.get("vascular_pearson_r", 1) or 1) < 0.88 else "MATCH"],
        ["Biological Anomaly Score",
         str(ctx.get("biological_anomaly_score", ctx.get("anomaly_score", "N/A"))),
         "", ""],
    ]
    bio_table = Table(bio_data,
                      colWidths=[T.CONTENT_W*0.38, T.CONTENT_W*0.18, T.CONTENT_W*0.22, T.CONTENT_W*0.22])
    bio_table.setStyle(_std_table_style())
    for i, row in enumerate(bio_data[1:], 1):
        if row[3] in ("ABSENT", "COMPOSITE ILLUMINATION", "GAN SMOOTHED", "MISMATCH"):
            bio_table.setStyle(TableStyle([("TEXTCOLOR", (3, i), (3, i), T.RED_ALERT),
                                            ("FONTNAME",  (3, i), (3, i), T.FONT_BOLD)]))
    story.append(bio_table)
    story.append(Spacer(1, 4 * mm))

    return story


def _build_metadata_reference(ctx, styles):
    """
    §5.6 + §7 — Metadata & Comparative Reference Analysis.
    Maps to MetadataAgent outputs.
    """
    story = []
    story.append(SectionHeader("§5.6 + §7  METADATA & COMPARATIVE REFERENCE ANALYSIS",
                               ref="MetadataAgent"))
    story.append(Spacer(1, 2 * mm))

    meta_data = [
        ["Field", "Value", "Flag"],
        ["EXIF Camera Present",     str(ctx.get("exif_camera_present", "N/A")),
         "YES" if not ctx.get("exif_camera_present", True) else ""],
        ["ELA Chi²",                f"{ctx.get('ela_chi2', 'N/A')}",
         "YES" if float(ctx.get("ela_chi2", 0) or 0) > 200 else ""],
        ["Thumbnail Mismatch",      str(ctx.get("thumbnail_mismatch", "N/A")),
         "YES" if ctx.get("thumbnail_mismatch") else ""],
        ["PRNU Absent",             str(ctx.get("prnu_absent", "N/A")),
         "YES" if ctx.get("prnu_absent") else ""],
        ["Software Tag",            str(ctx.get("software_tag", "")) or "—", ""],
        ["JPEG Quantisation Anomaly", str(ctx.get("jpeg_quantisation_anomaly", "N/A")),
         "YES" if ctx.get("jpeg_quantisation_anomaly") else ""],
        ["Cosine Dist to Authentic", str(ctx.get("cosine_dist_authentic", "N/A")),
         "HIGH" if float(ctx.get("cosine_dist_authentic", 0) or 0) > 0.40 else ""],
        ["Cosine Dist to Fake Set", str(ctx.get("cosine_dist_fake", "N/A")), ""],
        ["FaceNet Distance",        str(ctx.get("facenet_dist", "N/A")), ""],
        ["ArcFace Distance",        str(ctx.get("arcface_dist", "N/A")), ""],
        ["3DMM Shape Distance",     str(ctx.get("shape_3dmm_dist", "N/A")), ""],
        ["Reference Verdict",       str(ctx.get("reference_verdict", "N/A")), ""],
    ]
    col_w = [T.CONTENT_W * 0.45, T.CONTENT_W * 0.40, T.CONTENT_W * 0.15]
    meta_table = Table(meta_data, colWidths=col_w)
    meta_table.setStyle(_std_table_style())
    for i, row in enumerate(meta_data[1:], 1):
        if row[2] in ("YES", "HIGH"):
            meta_table.setStyle(TableStyle([("TEXTCOLOR", (2, i), (2, i), T.RED_ALERT),
                                             ("FONTNAME",  (2, i), (2, i), T.FONT_BOLD)]))
    story.append(meta_table)
    story.append(Spacer(1, 4 * mm))
    return story


def _build_fusion(ctx, styles):
    """
    §6 — Bayesian Confidence Scoring & Statistical Analysis.
    Maps to BayesianFusion outputs.
    Large verdict badge + per-module score bars + stats table.
    """
    story = []
    story.append(SectionHeader("§6  BAYESIAN CONFIDENCE SCORING & STATISTICAL ANALYSIS",
                               ref="BayesianFusion"))
    story.append(Spacer(1, 3 * mm))

    # Big verdict badge again (now with full width)
    decision = ctx.get("decision", "UNCERTAIN")
    final_score = ctx.get("final_score", 0.5)
    ci = ctx.get("confidence_interval", [0.0, 1.0])
    story.append(VerdictBadge(decision, final_score, ci))
    story.append(Spacer(1, 5 * mm))

    # Interpretation line
    interp = ctx.get("interpretation", "")
    if interp:
        interp_style = ParagraphStyle("interp", fontName=T.FONT_BOLD, fontSize=9,
                                       textColor=colors.HexColor("#333355"),
                                       alignment=TA_CENTER, spaceAfter=4*mm)
        story.append(Paragraph(f"Interpretation: {interp}", interp_style))

    # Per-module score bars
    story.append(Paragraph("PER-MODULE ANOMALY SCORES", styles["section_heading"]))
    per_module = ctx.get("per_module_scores", {})
    module_labels = {
        "geometry":     "§5.1  Geometry",
        "gan_artefact": "§5.2  GAN Artefact",
        "frequency":    "§5.2  Frequency",
        "texture":      "§5.3  Texture",
        "vlm":          "§5.4  VLM Explainability",
        "biological":   "§5.5  Biological",
        "metadata":     "§5.6  Metadata",
    }
    for key, label in module_labels.items():
        score = per_module.get(key, 0.5)
        story.append(ScoreBar(label, score))
    story.append(Spacer(1, 4 * mm))

    # Model statistics table
    story.append(Paragraph("MODEL STATISTICS (§6 FIXED CONSTANTS)", styles["section_heading"]))
    stats_data = [
        ["Statistic", "Value", "Description"],
        ["Model AUC-ROC",        str(ctx.get("model_auc_roc", "0.983")),
         "Area under ROC curve — classifier discrimination"],
        ["False Positive Rate",  f"{float(ctx.get('false_positive_rate', 0.021))*100:.1f}%",
         "Probability of wrongly flagging authentic image"],
        ["Calibration ECE",      str(ctx.get("calibration_ece", "0.014")),
         "Expected Calibration Error — lower is better"],
        ["Decision Threshold",   str(ctx.get("decision_threshold", "0.70")),
         "≥ 0.70 → DEEPFAKE  |  ≤ 0.35 → AUTHENTIC"],
    ]
    stats_table = Table(stats_data,
                        colWidths=[T.CONTENT_W*0.28, T.CONTENT_W*0.17, T.CONTENT_W*0.55])
    stats_table.setStyle(_std_table_style())
    story.append(stats_table)
    story.append(Spacer(1, 4 * mm))
    return story


def _build_narrative(ctx, styles):
    """
    §8–§10 — Narrative section (executive summary + findings).
    Text comes from Mistral-7B via Ollama (generator.py).
    Falls back to structured auto-text if LLM narrative not provided.
    """
    story = []
    story.append(SectionHeader("§8  EXECUTIVE SUMMARY & ANALYST FINDINGS", ref="§8–§10"))
    story.append(Spacer(1, 3 * mm))

    narrative = ctx.get("narrative_text", "")
    if narrative:
        story.append(Paragraph(narrative, styles["body"]))
    else:
        # Auto-generated fallback summary
        decision = ctx.get("decision", "UNCERTAIN")
        score = float(ctx.get("final_score", 0.5))
        ci = ctx.get("confidence_interval", [0.0, 1.0])

        auto_text = (
            f"Analysis of the submitted image was conducted across seven independent forensic "
            f"modules using the MFAD Multi-Factor AI Deepfake Detection pipeline. The Bayesian "
            f"ensemble fusion model returned a DeepFake Prediction Score of "
            f"<b>{score*100:.1f}%</b> with a 95% confidence interval of "
            f"[{ci[0]*100:.1f}%\u2013{ci[1]*100:.1f}%], yielding a primary verdict of "
            f"<b>{decision}</b>. "
        )

        if decision == "DEEPFAKE":
            auto_text += (
                "Multiple independent forensic signals corroborate this finding: "
                "spectral frequency anomalies consistent with GAN upsampling artefacts were "
                "identified in the mid and high frequency bands; texture consistency analysis "
                "revealed Earth Mover Distance values significantly above the authentic baseline "
                "at the neck-face boundary; biological plausibility checks detected absence of "
                "rPPG cardiac signal variation and implausible corneal highlight geometry. "
                "The combined weight of evidence is consistent with a synthetically generated "
                "or significantly manipulated facial image."
            )
        elif decision == "AUTHENTIC":
            auto_text += (
                "All independent forensic modules returned measurements consistent with "
                "authentic photographic origin. No significant GAN frequency artefacts were "
                "detected; texture boundary analysis revealed EMD values within the authentic "
                "baseline range; biological plausibility signals including rPPG cardiac variation "
                "and corneal highlight geometry were consistent with real physiological origin."
            )
        else:
            auto_text += (
                "Forensic evidence is inconclusive. Some modules returned anomalous readings "
                "while others did not. The image may have undergone partial enhancement or "
                "compression that affects some but not all detection signals. Manual review by "
                "a qualified forensic analyst is recommended before any evidential conclusion "
                "is drawn."
            )

        story.append(Paragraph(auto_text, styles["body"]))

    story.append(Spacer(1, 4 * mm))
    return story


def _build_legal_certification(ctx, styles):
    """
    §9–§10 — Legal Certification, Compliance Standards, Analyst Declaration.
    Maps to ReportGenerator outputs.
    """
    story = []
    story.append(SectionHeader("§9–§10  LEGAL CERTIFICATION & ANALYST DECLARATION",
                               ref="§9–§10"))
    story.append(Spacer(1, 3 * mm))

    # Compliance table
    standards = ctx.get("compliance_standards",
                        ["ISO/IEC 27037:2012", "SWGDE Best Practices 2023",
                         "NIST SP 800-101r1", "ACPO v5", "FRE Rule 702"])
    std_data = [["Reference Standard", "Applicability"]]
    std_applicability = {
        "ISO/IEC 27037:2012":    "Digital evidence identification, collection, preservation",
        "SWGDE Best Practices 2023": "Scientific working group — digital/multimedia evidence",
        "NIST SP 800-101r1":     "Guidelines on mobile device forensics",
        "ACPO v5":               "Association of Chief Police Officers digital evidence principles",
        "FRE Rule 702":          "Federal Rules of Evidence — admissibility of expert testimony",
    }
    for std in standards:
        std_data.append([std, std_applicability.get(std, "Applied as reference standard")])
    std_table = Table(std_data, colWidths=[T.CONTENT_W * 0.40, T.CONTENT_W * 0.60])
    std_table.setStyle(_std_table_style())
    story.append(std_table)
    story.append(Spacer(1, 5 * mm))

    # Declaration box
    analyst    = ctx.get("analyst_name", "[ANALYST NAME]")
    lab        = ctx.get("lab_accreditation", "[LABORATORY ACCREDITATION]")
    report_id  = ctx.get("report_id", "DFA-XXXX-TC-XXXXXX")
    generated  = ctx.get("generated_at", datetime.utcnow().isoformat())

    decl_text = (
        f"I, <b>{analyst}</b>, employed by / accredited through <b>{lab}</b>, hereby certify that "
        f"the forensic analysis described in this report (ID: <b>{report_id}</b>, generated "
        f"{generated}) was conducted in accordance with the above referenced standards. "
        f"The methodology, measurements, and conclusions presented herein represent my "
        f"independent professional opinion based on the analytical outputs of the MFAD system. "
        f"This report is submitted for lawful forensic purposes and is admissible as expert "
        f"testimony under the applicable rules of evidence."
    )
    decl_style = ParagraphStyle(
        "decl", fontName=T.FONT_BODY, fontSize=8.5,
        textColor=colors.HexColor("#1A1A2E"), leading=14,
        alignment=TA_JUSTIFY, borderPadding=(6, 8, 6, 8),
        borderColor=T.CYAN_DIM, borderWidth=0.8,
        backColor=colors.HexColor("#F4F8FF"),
        leftIndent=6, rightIndent=6,
    )
    story.append(Paragraph(decl_text, decl_style))
    story.append(Spacer(1, 6 * mm))

    # Signature block
    sig_data = [
        [
            Paragraph("<b>Analyst Signature</b>", styles["label"]),
            Paragraph("<b>Date</b>", styles["label"]),
            Paragraph("<b>Lab Seal / Stamp</b>", styles["label"]),
        ],
        [
            Paragraph("_______________________________", styles["body"]),
            Paragraph(generated[:10], styles["body"]),
            Paragraph("_______________________________", styles["body"]),
        ],
        [
            Paragraph(analyst, styles["body"]),
            Paragraph("", styles["body"]),
            Paragraph(lab, styles["body"]),
        ],
    ]
    sig_table = Table(sig_data, colWidths=[T.CONTENT_W / 3] * 3)
    sig_table.setStyle(TableStyle([
        ("ALIGN",     (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, T.BORDER),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, T.BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 4 * mm))

    # Watermark notice
    notice_style = ParagraphStyle(
        "notice", fontName=T.FONT_OBLIQ, fontSize=7.5,
        textColor=T.GREY_MID, alignment=TA_CENTER, leading=12,
    )
    story.append(Paragraph(
        "This report was generated automatically by the MFAD Forensic AI system. "
        "Findings are probabilistic in nature and must be reviewed by a qualified human forensic analyst "
        "before being submitted as evidence in any legal proceeding.",
        notice_style
    ))
    return story


# ─────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

def build_report(ctx: dict, output_path: str) -> str:
    """
    Assemble and render the full forensic PDF report.

    Args:
        ctx:         The fully merged context dict from master_agent.py
                     (all agent outputs + fusion result).
        output_path: Destination path for the output PDF file.

    Returns:
        output_path — confirmed path to the generated PDF.

    Report structure:
        Cover Page          — full-bleed dark design, case metadata, verdict badge
        §1                  — Chain of Custody & Provenance
        §5.1                — Facial Geometry
        §5.2                — Frequency + GAN Artefact
        §5.3                — Texture Consistency
        §5.4                — VLM Explainability + Heatmap
        §5.5                — Biological Plausibility
        §5.6 + §7           — Metadata & Comparative Reference
        §6                  — Bayesian Fusion Confidence Scores
        §8                  — Executive Summary & Narrative
        §9–§10              — Legal Certification & Declaration
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    styles = build_styles()

    # ── Document setup ───────────────────────────────────────
    doc = BaseDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=T.MARGIN_L,
        rightMargin=T.MARGIN_R,
        topMargin=T.MARGIN_T,
        bottomMargin=T.MARGIN_B,
    )
    # Attach report_id so header/footer can access it
    doc._report_id = ctx.get("report_id", "DFA-XXXX-TC-XXXXXX")

    # Cover frame (full page, very small margins so the bg bleeds)
    cover_frame = Frame(
        8 * mm, 20 * mm, T.PAGE_W - 16 * mm, T.PAGE_H - 24 * mm,
        id="cover_frame", showBoundary=0,
    )
    # Inner content frame
    inner_frame = Frame(
        T.MARGIN_L, T.MARGIN_B,
        T.CONTENT_W, T.PAGE_H - T.MARGIN_T - T.MARGIN_B,
        id="inner_frame", showBoundary=0,
    )

    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=_draw_cover_page),
        PageTemplate(id="Inner", frames=[inner_frame], onPage=_draw_inner_page),
    ])

    # ── Story assembly ────────────────────────────────────────
    story = []

    # Cover
    story.extend(_build_cover(ctx, styles))

    # Switch to inner template after cover page break
    story.append(NextPageTemplate("Inner"))
    story.append(PageBreak())

    # §1 Chain of Custody
    story.extend(_build_chain_of_custody(ctx, styles))
    story.append(PageBreak())

    # §5.1 – §5.5 agent findings
    story.extend(_build_per_module_findings(ctx, styles))
    story.append(PageBreak())

    # §5.6 + §7 Metadata
    story.extend(_build_metadata_reference(ctx, styles))

    # §6 Fusion
    story.extend(_build_fusion(ctx, styles))
    story.append(PageBreak())

    # §8 Narrative
    story.extend(_build_narrative(ctx, styles))

    # §9–§10 Legal
    story.extend(_build_legal_certification(ctx, styles))

    # ── Render ───────────────────────────────────────────────
    doc.build(story)
    return output_path


# ─────────────────────────────────────────────
#  STANDALONE TEST  — run with sample JSON
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from mfad_project_spec_sample import SAMPLE_CTX  # noqa — import sample if available

    SAMPLE = {
        # Report metadata
        "report_id":          "DFA-2025-TC-A3F1B2",
        "generated_at":       "2026-03-10T06:10:00Z",
        "analyst_name":       "Dr. A. Sharma",
        "lab_accreditation":  "MFAD Forensic AI Lab — ISO/IEC 17025 Accredited",
        "compliance_standards": ["ISO/IEC 27037:2012", "SWGDE Best Practices 2023",
                                  "NIST SP 800-101r1", "ACPO v5", "FRE Rule 702"],
        # Preprocessing
        "image_path":         "/evidence/TC-A3F1B2.jpg",
        "face_bbox":          [412, 180, 1640, 1900],
        "hash_sha256":        "3a7f2c91d84b0e56f123aa98cd7e4512b093f6a8210d7c3e9b5f014d82ea76c1",
        "hash_sha256_verified": True,
        "face_crop_path":     "temp/face_crop.jpg",
        "normalized_path":    "temp/normalized.jpg",
        "landmarks_path":     "temp/landmarks.json",
        "exif_camera_present": False,
        "ela_chi2":           847.0,
        "thumbnail_mismatch": True,
        "prnu_absent":        True,
        "software_tag":       "",
        "icc_profile":        "sRGB IEC61966-2.1",
        # Geometry
        "symmetry_index":       0.74,
        "jaw_curvature_deg":    11.2,
        "ear_alignment_px":     8.7,
        "philtrum_length_mm":   20.4,
        "interocular_dist_px":  64.3,
        "eye_aspect_ratio_l":   0.31,
        "eye_aspect_ratio_r":   0.29,
        "lip_thickness_ratio":  0.22,
        "neck_face_boundary":   "sharp_edge",
        "geometry_anomaly_score": 0.884,
        # Frequency
        "fft_mid_anomaly_db":      9.4,
        "fft_high_anomaly_db":     13.3,
        "fft_ultrahigh_anomaly_db": 15.6,
        "gan_probability":         0.967,
        "upsampling_grid_detected": True,
        "frequency_anomaly_score":  0.912,
        # Texture
        "forehead_cheek_emd":  0.061,
        "cheek_jaw_emd_l":     0.193,
        "cheek_jaw_emd_r":     0.211,
        "periorbital_nasal_emd": 0.072,
        "lip_chin_emd":        0.148,
        "neck_face_emd":       0.274,
        "lbp_uniformity":      0.51,
        "seam_detected":       True,
        "texture_anomaly_score": 0.895,
        # VLM
        "heatmap_path":            "",
        "vlm_caption":             "Central facial region shows high GAN probability; peripheral zones show identity-augmenting texture modifications.",
        "vlm_verdict":             "FAKE",
        "vlm_confidence":          0.93,
        "saliency_score":          0.91,
        "high_activation_regions":   ["eyes", "nose", "mouth"],
        "medium_activation_regions": ["cheeks", "brow", "chin"],
        "low_activation_regions":    ["background", "hair", "shoulders"],
        "zone_gan_probability":      0.93,
        "vlm_anomaly_score":         0.931,
        # Biological
        "rppg_snr":             0.09,
        "corneal_deviation_deg": 14.3,
        "micro_texture_var":    0.012,
        "vascular_pearson_r":   0.41,
        "biological_anomaly_score": 0.826,
        # Metadata
        "jpeg_quantisation_anomaly": True,
        "cosine_dist_authentic":     0.71,
        "cosine_dist_fake":          0.18,
        "facenet_dist":              0.71,
        "arcface_dist":              0.68,
        "shape_3dmm_dist":           0.58,
        "reference_verdict":         "HIGH_DISSIMILARITY_TO_AUTHENTIC",
        "metadata_anomaly_score":    0.973,
        # Fusion
        "per_module_scores": {
            "geometry":    0.884,
            "gan_artefact": 0.967,
            "frequency":   0.912,
            "texture":     0.895,
            "vlm":         0.931,
            "biological":  0.826,
            "metadata":    0.973,
        },
        "final_score":         0.950,
        "confidence_interval": [0.931, 0.966],
        "decision":            "DEEPFAKE",
        "interpretation":      "Very High Confidence",
        "model_auc_roc":       0.983,
        "false_positive_rate": 0.021,
        "calibration_ece":     0.014,
        "decision_threshold":  0.70,
    }

    out = build_report(SAMPLE, "reports/DFA-2025-TC-A3F1B2.pdf")
    print(f"Report generated: {out}")
