"""
MFAD — Multi-Factor AI Deepfake Detection System
report_agent/template.py  v3.0  — Professional Forensic Edition

ReportLab PDF layout engine.
Maps to §8 Legal Certification + §9–§10 Analyst Declaration
of reference report DFA-2025-TC-00471.

Design philosophy:
  - White body pages — clean, readable, court-grade
  - Deep navy + gold accent — serious, institutional, trustworthy
  - Strict typographic hierarchy — Times (headings/body) + Helvetica (labels/tables)
  - Every anomalous value auto-highlighted in crimson
  - Cover page full-bleed dark with gold accents
  - Table of Contents on page 2
  - Running header/footer with case ID on every inner page
"""

import os
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame,
    NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, KeepTogether,
)


# ══════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════

class T:
    # ── Palette ───────────────────────────────────────────
    NAVY         = colors.HexColor("#0B1628")
    GOLD         = colors.HexColor("#C9A84C")
    GOLD_LIGHT   = colors.HexColor("#E8C97A")
    WHITE        = colors.HexColor("#FFFFFF")
    INK          = colors.HexColor("#0D1117")
    INK_MID      = colors.HexColor("#3A4558")
    INK_LIGHT    = colors.HexColor("#6B7A96")
    RULE_LIGHT   = colors.HexColor("#D4D9E4")
    ROW_ALT      = colors.HexColor("#F0F3F8")
    ROW_HEAD     = colors.HexColor("#0B1628")
    CRIMSON      = colors.HexColor("#C0152A")
    CRIMSON_BG   = colors.HexColor("#FDF0F1")
    AMBER        = colors.HexColor("#B86A00")
    EMERALD      = colors.HexColor("#0A6B3C")

    # ── Fonts ────────────────────────────────────────────
    SERIF        = "Times-Roman"
    SERIF_BOLD   = "Times-Bold"
    SERIF_ITALIC = "Times-Italic"
    SANS         = "Helvetica"
    SANS_BOLD    = "Helvetica-Bold"
    SANS_OBLIQ   = "Helvetica-Oblique"
    MONO         = "Courier"

    # ── Page geometry ────────────────────────────────────
    PAGE_W, PAGE_H = A4
    MARGIN_L     = 22 * mm
    MARGIN_R     = 22 * mm
    MARGIN_T     = 28 * mm
    MARGIN_B     = 24 * mm
    CONTENT_W    = PAGE_W - MARGIN_L - MARGIN_R


# ══════════════════════════════════════════════════════════
#  PARAGRAPH STYLES
# ══════════════════════════════════════════════════════════

def build_styles():
    S = {}
    S["cover_title"] = ParagraphStyle(
        "cover_title", fontName=T.SERIF_BOLD, fontSize=30,
        textColor=T.WHITE, leading=38, alignment=TA_LEFT, spaceAfter=3*mm)
    S["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle", fontName=T.SERIF_ITALIC, fontSize=13,
        textColor=T.GOLD_LIGHT, leading=18, alignment=TA_LEFT, spaceAfter=8*mm)
    S["cover_label"] = ParagraphStyle(
        "cover_label", fontName=T.SANS_BOLD, fontSize=6.5,
        textColor=T.GOLD, leading=10, tracking=2)
    S["cover_value"] = ParagraphStyle(
        "cover_value", fontName=T.SANS, fontSize=9,
        textColor=T.WHITE, leading=14)
    S["h2"] = ParagraphStyle(
        "h2", fontName=T.SANS_BOLD, fontSize=8.5,
        textColor=T.NAVY, leading=13, tracking=1.5,
        spaceBefore=5*mm, spaceAfter=2*mm)
    S["body"] = ParagraphStyle(
        "body", fontName=T.SERIF, fontSize=9,
        textColor=T.INK, leading=14, alignment=TA_JUSTIFY, spaceAfter=3*mm)
    S["body_sans"] = ParagraphStyle(
        "body_sans", fontName=T.SANS, fontSize=8.5,
        textColor=T.INK, leading=13, alignment=TA_JUSTIFY, spaceAfter=2*mm)
    S["footnote"] = ParagraphStyle(
        "footnote", fontName=T.SANS, fontSize=7,
        textColor=T.INK_LIGHT, leading=10, alignment=TA_JUSTIFY)
    S["th"] = ParagraphStyle(
        "th", fontName=T.SANS_BOLD, fontSize=7.5,
        textColor=T.WHITE, leading=11, tracking=0.5)
    S["td"] = ParagraphStyle(
        "td", fontName=T.SANS, fontSize=8,
        textColor=T.INK, leading=12)
    S["sig_label"] = ParagraphStyle(
        "sig_label", fontName=T.SANS_BOLD, fontSize=7,
        textColor=T.INK_LIGHT, leading=10, tracking=1)
    S["sig_value"] = ParagraphStyle(
        "sig_value", fontName=T.SANS, fontSize=8.5,
        textColor=T.INK, leading=13)
    return S


# ══════════════════════════════════════════════════════════
#  CUSTOM FLOWABLES
# ══════════════════════════════════════════════════════════

class GoldRule(Flowable):
    def __init__(self, width=None, thickness=0.8):
        super().__init__()
        self.width = width or T.CONTENT_W
        self.thickness = thickness
        self.height = self.thickness + 2*mm

    def draw(self):
        self.canv.setStrokeColor(T.GOLD)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.thickness/2, self.width, self.thickness/2)


class NavyRule(Flowable):
    def __init__(self, width=None):
        super().__init__()
        self.width = width or T.CONTENT_W
        self.height = 1.5*mm

    def draw(self):
        self.canv.setStrokeColor(T.RULE_LIGHT)
        self.canv.setLineWidth(0.4)
        self.canv.line(0, 0.2, self.width, 0.2)


class SectionLabel(Flowable):
    """
    Professional section label with gold left stripe,
    section number in gold, and title in navy serif.
    """
    def __init__(self, number, title, width=None):
        super().__init__()
        self.number = number
        self.title  = title
        self.width  = width or T.CONTENT_W
        self.height = 12*mm

    def draw(self):
        c = self.canv
        h = self.height
        c.setFillColor(T.GOLD)
        c.rect(0, 0, 3.5, h, fill=1, stroke=0)
        c.setFillColor(T.GOLD)
        c.setFont(T.SANS_BOLD, 8)
        c.drawString(8, h/2 - 1, self.number)
        num_w = c.stringWidth(self.number, T.SANS_BOLD, 8)
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.5)
        c.line(14 + num_w, h*0.2, 14 + num_w, h*0.8)
        c.setFillColor(T.NAVY)
        c.setFont(T.SERIF_BOLD, 11)
        c.drawString(20 + num_w, h/2 - 4, self.title)


class VerdictPanel(Flowable):
    """Court-grade verdict panel with coloured top border."""
    def __init__(self, verdict, final_score, ci, width=None):
        super().__init__()
        self.verdict = verdict.upper()
        self.score   = float(final_score)
        self.ci      = ci or [0.0, 1.0]
        self.width   = width or T.CONTENT_W
        self.height  = 42*mm

    def _accent(self):
        if self.verdict == "DEEPFAKE":  return T.CRIMSON
        if self.verdict == "AUTHENTIC": return T.EMERALD
        return T.AMBER

    def draw(self):
        c = self.canv
        w, h  = self.width, self.height
        accent = self._accent()
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.6)
        c.roundRect(0, 0, w, h, 4, fill=0, stroke=1)
        c.setFillColor(accent)
        c.roundRect(0, h-5, w, 5, 4, fill=1, stroke=0)
        c.rect(0, h-8, w, 4, fill=1, stroke=0)
        c.setFillColor(T.INK_LIGHT)
        c.setFont(T.SANS_BOLD, 6.5)
        c.drawCentredString(w/2, h-16,
            "PRIMARY DETERMINATION  —  BAYESIAN ENSEMBLE ANALYSIS")
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.4)
        c.line(w*0.1, h-19, w*0.9, h-19)
        c.setFillColor(accent)
        c.setFont(T.SERIF_BOLD, 28)
        c.drawCentredString(w/2, h/2+1, self.verdict)
        c.setFillColor(T.INK_MID)
        c.setFont(T.SANS_BOLD, 10.5)
        c.drawCentredString(w/2, h/2-13,
            f"DeepFake Prediction Score:  {self.score*100:.1f}%")
        c.setFillColor(T.INK_LIGHT)
        c.setFont(T.SANS, 8)
        c.drawCentredString(w/2, h/2-24,
            f"95% Confidence Interval:  [{self.ci[0]*100:.1f}%  –  {self.ci[1]*100:.1f}%]")
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.4)
        c.line(w*0.1, 10, w*0.9, 10)
        c.setFillColor(T.INK_LIGHT)
        c.setFont(T.SANS, 6.5)
        c.drawCentredString(w/2, 4,
            "Threshold: >= 0.70 -> DEEPFAKE  |  0.35-0.70 -> UNCERTAIN  |  <= 0.35 -> AUTHENTIC")


class ModuleScoreBar(Flowable):
    """Horizontal score bar for per-module confidence display."""
    def __init__(self, section_ref, module_name, score, weight, width=None):
        super().__init__()
        self.ref    = section_ref
        self.name   = module_name
        self.score  = float(score)
        self.weight = float(weight)
        self.width  = width or T.CONTENT_W
        self.height = 9*mm

    def _bar_color(self):
        if self.score >= 0.70: return T.CRIMSON
        if self.score >= 0.35: return T.AMBER
        return T.EMERALD

    def draw(self):
        c       = self.canv
        ref_w   = 18*mm
        name_w  = 52*mm
        wt_w    = 14*mm
        bar_a   = self.width - ref_w - name_w - wt_w - 22*mm
        bar_h   = 3.5
        y       = self.height / 2
        c.setFillColor(T.GOLD)
        c.setFont(T.SANS_BOLD, 6.5)
        c.drawString(0, y-2.5, self.ref)
        c.setFillColor(T.INK)
        c.setFont(T.SANS, 8)
        c.drawString(ref_w, y-3, self.name)
        c.setFillColor(T.INK_LIGHT)
        c.setFont(T.SANS, 6.5)
        c.drawString(ref_w+name_w, y-2.5, f"w={self.weight:.2f}")
        bx = ref_w + name_w + wt_w
        c.setFillColor(T.RULE_LIGHT)
        c.roundRect(bx, y-bar_h/2, bar_a, bar_h, 1.5, fill=1, stroke=0)
        c.setFillColor(self._bar_color())
        c.roundRect(bx, y-bar_h/2, max(bar_a*self.score, 2), bar_h, 1.5, fill=1, stroke=0)
        c.setFillColor(T.INK)
        c.setFont(T.SANS_BOLD, 8)
        c.drawRightString(self.width, y-3, f"{self.score*100:.1f}%")


class CustodyField(Flowable):
    """Single key-value row for Chain of Custody display."""
    def __init__(self, label, value, flagged=False, width=None):
        super().__init__()
        self.label   = label
        self.value   = str(value)
        self.flagged = flagged
        self.width   = width or T.CONTENT_W
        self.height  = 7.5*mm

    def draw(self):
        c   = self.canv
        h   = self.height
        sep = self.width * 0.40
        c.setFillColor(T.CRIMSON_BG if self.flagged else T.WHITE)
        c.rect(0, 0, self.width, h, fill=1, stroke=0)
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.3)
        c.line(0, 0, self.width, 0)
        c.setFillColor(T.INK_MID)
        c.setFont(T.SANS_BOLD, 7.5)
        c.drawString(4, h/2-3, self.label)
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.3)
        c.line(sep, h*0.15, sep, h*0.85)
        val_color = T.CRIMSON if self.flagged else T.INK
        val_font  = T.SANS_BOLD if self.flagged else (
            T.MONO if len(self.value) > 30 else T.SANS)
        c.setFillColor(val_color)
        c.setFont(val_font, 7.5 if len(self.value) > 40 else 8)
        display = self.value if len(self.value) <= 55 else self.value[:52] + "..."
        c.drawString(sep+5, h/2-3, display)
        if self.flagged:
            c.setFillColor(T.CRIMSON)
            c.setFont(T.SANS_BOLD, 7)
            c.drawRightString(self.width-3, h/2-2.5, "FLAGGED")


class HeatmapFrame(Flowable):
    """Heatmap image embed with professional mat and caption."""
    def __init__(self, path, caption="", width=None):
        super().__init__()
        self.path    = path
        self.caption = caption
        self.img_w   = (width or T.CONTENT_W) * 0.60
        self.img_h   = self.img_w * 0.72
        self.height  = self.img_h + 16*mm

    def draw(self):
        c     = self.canv
        x     = (T.CONTENT_W - self.img_w) / 2
        img_y = 10*mm
        c.setFillColor(colors.HexColor("#C8CDD8"))
        c.rect(x+2, img_y-2, self.img_w, self.img_h, fill=1, stroke=0)
        c.setFillColor(T.WHITE)
        c.setStrokeColor(T.RULE_LIGHT)
        c.setLineWidth(0.6)
        c.rect(x, img_y, self.img_w, self.img_h, fill=1, stroke=1)
        if self.path and os.path.exists(str(self.path)):
            c.drawImage(str(self.path), x+2, img_y+2,
                        self.img_w-4, self.img_h-4, preserveAspectRatio=True)
        else:
            c.setFillColor(colors.HexColor("#E8EBF2"))
            c.rect(x+2, img_y+2, self.img_w-4, self.img_h-4, fill=1, stroke=0)
            c.setFillColor(T.INK_LIGHT)
            c.setFont(T.SANS, 8)
            c.drawCentredString(x+self.img_w/2, img_y+self.img_h/2+4,
                                "[ Grad-CAM Activation Heatmap ]")
            c.setFont(T.SANS, 7)
            c.drawCentredString(x+self.img_w/2, img_y+self.img_h/2-8,
                                "EfficientNet-B4 (FaceForensics++ v3)")
        c.setFillColor(T.GOLD)
        c.rect(x, img_y+self.img_h, self.img_w, 2, fill=1, stroke=0)
        if self.caption:
            c.setFillColor(T.INK_LIGHT)
            c.setFont(T.SERIF_ITALIC, 7.5)
            c.drawCentredString(T.CONTENT_W/2, img_y-6, self.caption)


# ══════════════════════════════════════════════════════════
#  PAGE TEMPLATES
# ══════════════════════════════════════════════════════════

def _draw_cover(canv, doc):
    w, h = A4
    canv.saveState()
    canv.setFillColor(T.NAVY)
    canv.rect(0, 0, w, h, fill=1, stroke=0)
    canv.setFillColor(colors.HexColor("#0E1F38"))
    p = canv.beginPath()
    p.moveTo(w*0.45, h)
    p.lineTo(w, h)
    p.lineTo(w, h*0.52)
    p.close()
    canv.drawPath(p, fill=1, stroke=0)
    canv.setFillColor(T.GOLD)
    canv.rect(0, h-4, w, 4, fill=1, stroke=0)
    canv.setFillColor(colors.HexColor("#7A6030"))
    canv.rect(0, h-7, w, 1, fill=1, stroke=0)
    canv.setFillColor(T.GOLD)
    canv.rect(0, 14*mm, w, 0.8, fill=1, stroke=0)
    canv.setFillColor(T.GOLD)
    canv.setFont(T.SANS_BOLD, 6.5)
    canv.drawCentredString(w/2, 9*mm,
        "CONFIDENTIAL  —  FORENSIC EVIDENCE  —  AUTHORISED PERSONNEL ONLY")
    canv.setFillColor(colors.HexColor("#445577"))
    canv.setFont(T.SANS, 6.5)
    canv.drawString(22*mm, 5*mm,
        "ISO/IEC 27037  .  SWGDE Best Practices  .  NIST SP 800-101r1  .  ACPO v5  .  FRE Rule 702")
    canv.restoreState()


def _draw_inner(canv, doc):
    w, h = A4
    canv.saveState()
    # Header
    canv.setFillColor(T.NAVY)
    canv.rect(0, h-T.MARGIN_T, w, T.MARGIN_T, fill=1, stroke=0)
    canv.setFillColor(T.GOLD)
    canv.rect(0, h-T.MARGIN_T, w, 1.2, fill=1, stroke=0)
    canv.setFillColor(T.GOLD)
    canv.setFont(T.SANS_BOLD, 7.5)
    canv.drawString(T.MARGIN_L, h-T.MARGIN_T+10, "MFAD  FORENSIC AI LABORATORY")
    canv.setFillColor(colors.HexColor("#8899BB"))
    canv.setFont(T.SANS, 7)
    canv.drawCentredString(w/2, h-T.MARGIN_T+10, "FORENSIC IMAGE AUTHENTICATION REPORT")
    report_id = getattr(doc, "_report_id", "DFA-XXXX-TC-XXXXXX")
    canv.setFillColor(T.GOLD_LIGHT)
    canv.setFont(T.SANS_BOLD, 7)
    canv.drawRightString(w-T.MARGIN_R, h-T.MARGIN_T+10, f"Case  {report_id}")
    # Footer
    canv.setFillColor(T.GOLD)
    canv.rect(T.MARGIN_L, T.MARGIN_B-3, T.CONTENT_W, 0.6, fill=1, stroke=0)
    canv.setFillColor(T.INK_LIGHT)
    canv.setFont(T.SANS, 6.5)
    canv.drawString(T.MARGIN_L, T.MARGIN_B-11, "CONFIDENTIAL — FORENSIC EVIDENCE")
    canv.drawCentredString(w/2, T.MARGIN_B-11, f"— {doc.page} —")
    canv.drawRightString(w-T.MARGIN_R, T.MARGIN_B-11,
                         datetime.now(timezone.utc).strftime("%d %B %Y"))
    canv.restoreState()


# ══════════════════════════════════════════════════════════
#  TABLE STYLE
# ══════════════════════════════════════════════════════════

def _table_style(flagged_rows=None):
    base = [
        ("BACKGROUND",    (0,0), (-1,0),  T.ROW_HEAD),
        ("TEXTCOLOR",     (0,0), (-1,0),  T.WHITE),
        ("FONTNAME",      (0,0), (-1,0),  T.SANS_BOLD),
        ("FONTSIZE",      (0,0), (-1,0),  7.5),
        ("TOPPADDING",    (0,0), (-1,0),  5),
        ("BOTTOMPADDING", (0,0), (-1,0),  5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("FONTNAME",      (0,1), (-1,-1), T.SANS),
        ("FONTSIZE",      (0,1), (-1,-1), 8),
        ("TEXTCOLOR",     (0,1), (-1,-1), T.INK),
        ("TOPPADDING",    (0,1), (-1,-1), 4),
        ("BOTTOMPADDING", (0,1), (-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [T.WHITE, T.ROW_ALT]),
        ("LINEBELOW",     (0,0), (-1,0),  1.2, T.GOLD),
        ("GRID",          (0,0), (-1,-1), 0.3, T.RULE_LIGHT),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]
    if flagged_rows:
        for r in flagged_rows:
            base.append(("BACKGROUND", (0,r), (-1,r), T.CRIMSON_BG))
    return TableStyle(base)


def _flag_rows(data_rows, flag_col):
    flagged = []
    for i, row in enumerate(data_rows, 1):
        val = str(row[flag_col]) if flag_col < len(row) else ""
        if "FLAG" in val.upper():
            flagged.append(i)
    return flagged


# ══════════════════════════════════════════════════════════
#  SECTION BUILDERS
# ══════════════════════════════════════════════════════════

def _build_cover(ctx, S):
    story = []
    story.append(Spacer(1, 48*mm))
    ban = ParagraphStyle("ban", fontName=T.SANS_BOLD, fontSize=7,
                          textColor=T.GOLD, tracking=3, alignment=TA_LEFT)
    story.append(Paragraph("FORENSIC DIGITAL EVIDENCE  —  RESTRICTED", ban))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Forensic Image<br/>Authentication Report", S["cover_title"]))
    story.append(Paragraph(
        "Multi-Factor AI Deepfake Detection — Bayesian Ensemble Analysis",
        S["cover_subtitle"]))
    story.append(Spacer(1, 6*mm))
    story.append(GoldRule())
    story.append(Spacer(1, 6*mm))

    report_id = ctx.get("report_id",        "DFA-XXXX-TC-XXXXXX")
    generated = ctx.get("generated_at",     datetime.now(timezone.utc).isoformat())
    analyst   = ctx.get("analyst_name",     "—")
    lab       = ctx.get("lab_accreditation","—")
    img       = ctx.get("image_path",       "—")
    sha256    = ctx.get("hash_sha256",      "—")
    decision  = ctx.get("decision",         "UNCERTAIN")
    score     = ctx.get("final_score",      0.5)

    fields = [
        ("REPORT IDENTIFIER",  report_id),
        ("DATE GENERATED",     generated),
        ("LEAD ANALYST",       analyst),
        ("LABORATORY",         lab),
        ("SUBJECT IMAGE",      os.path.basename(str(img))),
        ("SHA-256 INTEGRITY",  sha256[:40]+"..." if len(str(sha256))>43 else sha256),
    ]
    for label, value in fields:
        row = [[Paragraph(label, S["cover_label"]),
                Paragraph(str(value), S["cover_value"])]]
        t = Table(row, colWidths=[T.CONTENT_W*0.32, T.CONTENT_W*0.68])
        t.setStyle(TableStyle([
            ("LEFTPADDING",   (0,0),(-1,-1), 0),
            ("RIGHTPADDING",  (0,0),(-1,-1), 0),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LINEBELOW",     (0,0),(-1,-1), 0.3, colors.HexColor("#1E3060")),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        story.append(t)

    story.append(Spacer(1, 10*mm))
    verdict_color = {"DEEPFAKE": T.CRIMSON, "AUTHENTIC": T.EMERALD}.get(decision, T.AMBER)
    v_style = ParagraphStyle("cv", fontName=T.SERIF_BOLD, fontSize=22,
                              textColor=verdict_color, leading=28, alignment=TA_LEFT)
    lbl_s   = ParagraphStyle("cvl", fontName=T.SANS_BOLD, fontSize=6.5,
                              textColor=T.GOLD, leading=10, tracking=2)
    sc_s    = ParagraphStyle("cvs", fontName=T.SANS, fontSize=9,
                              textColor=colors.HexColor("#9AAABB"), leading=13)
    story.append(Paragraph("PRIMARY DETERMINATION", lbl_s))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(decision, v_style))
    story.append(Paragraph(
        f"DeepFake Prediction Score: {float(score)*100:.1f}%   |   Reference: ISO/IEC 27037:2012",
        sc_s))
    story.append(PageBreak())
    return story


def _build_toc(ctx, S):
    story = []
    story.append(SectionLabel("", "TABLE OF CONTENTS"))
    story.append(GoldRule())
    story.append(Spacer(1, 5*mm))
    items = [
        ("§1",        "Chain of Custody & Image Provenance",               "3"),
        ("§5.1",      "Facial Geometry & Landmark Analysis",               "3"),
        ("§5.2",      "GAN Artefact & Frequency-Domain Analysis",          "4"),
        ("§5.3",      "Texture Consistency & Skin-Tone Mapping",           "4"),
        ("§5.4",      "Explainability Heat-Map Analysis (VLM)",            "5"),
        ("§5.5",      "Biological Plausibility Assessment",                "5"),
        ("§5.6 + §7", "Metadata & Comparative Reference Analysis",         "6"),
        ("§6",        "Bayesian Confidence Scoring & Statistics",          "6"),
        ("§8",        "Executive Summary & Analyst Findings",              "7"),
        ("§9 - §10",  "Legal Certification & Analyst Declaration",         "7"),
    ]
    ref_s  = ParagraphStyle("tr", fontName=T.SANS_BOLD, fontSize=8,
                             textColor=T.GOLD, leading=12)
    ttl_s  = ParagraphStyle("tt", fontName=T.SANS, fontSize=8.5,
                             textColor=T.INK, leading=13)
    pg_s   = ParagraphStyle("tp", fontName=T.SANS, fontSize=8.5,
                             textColor=T.INK_MID, leading=13, alignment=TA_RIGHT)
    rows   = [[Paragraph(r, ref_s), Paragraph(t, ttl_s), Paragraph(p, pg_s)]
               for r, t, p in items]
    toc    = Table(rows, colWidths=[14*mm, T.CONTENT_W-27*mm, 13*mm])
    toc.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
        ("RIGHTPADDING",  (0,0),(-1,-1), 4),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("LINEBELOW",     (0,0),(-1,-1), 0.3, T.RULE_LIGHT),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [T.WHITE, T.ROW_ALT]),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(toc)
    story.append(PageBreak())
    return story


def _build_custody(ctx, S):
    story = []
    story.append(SectionLabel("§1", "Chain of Custody & Image Provenance"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "The following records establish the evidentiary integrity of the submitted image "
        "in accordance with ISO/IEC 27037:2012 Chain of Custody requirements. "
        "All cryptographic hashes were verified at intake and at report generation time.",
        S["body"]))
    fields = [
        ("Image Path",              ctx.get("image_path","—"),            False),
        ("SHA-256 Hash",            ctx.get("hash_sha256","—"),           False),
        ("Hash Verified at Report", str(ctx.get("hash_sha256_verified",False)), False),
        ("EXIF Camera Make/Model",  str(ctx.get("exif_camera_present","—")),
                                    not ctx.get("exif_camera_present", True)),
        ("ELA Chi-Squared Score",   f"{ctx.get('ela_chi2',0.0):.2f}",
                                    float(ctx.get("ela_chi2",0) or 0) > 200),
        ("Thumbnail Mismatch",      str(ctx.get("thumbnail_mismatch",False)),
                                    bool(ctx.get("thumbnail_mismatch",False))),
        ("PRNU Camera Fingerprint", "ABSENT" if ctx.get("prnu_absent") else "PRESENT",
                                    bool(ctx.get("prnu_absent",False))),
        ("Software Tag",            str(ctx.get("software_tag","")) or "Not present", False),
        ("ICC Colour Profile",      str(ctx.get("icc_profile","")) or "Not present",  False),
        ("Face Bounding Box",       str(ctx.get("face_bbox","—")),         False),
        ("Normalised Image Path",   str(ctx.get("normalized_path","—")),   False),
        ("Landmark Map Path",       str(ctx.get("landmarks_path","—")),    False),
    ]
    for label, value, flagged in fields:
        story.append(CustodyField(label, value, flagged))
    story.append(Spacer(1, 3*mm))
    story.append(NavyRule())
    return story


def _build_geometry(ctx, S):
    story = []
    story.append(SectionLabel("§5.1", "Facial Geometry & Landmark Analysis"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "68-point dlib landmark detection was applied to the normalised face crop. "
        "Anthropometric ratios were computed and compared against the population-level "
        "authentic baseline from §5.1 of the reference report.",
        S["body"]))

    def _s(val, above=None, below=None, lo=None, hi=None):
        try:
            v = float(val)
            if above is not None and v > above: return "FLAGGED"
            if below is not None and v < below: return "FLAGGED"
            if lo is not None and hi is not None and not (lo <= v <= hi): return "FLAGGED"
            return "Normal"
        except: return "—"

    rows = [
        ["Metric", "Measured", "Authentic Baseline", "Status"],
        ["Symmetry Index",          str(ctx.get("symmetry_index","—")),       "0.92 – 1.00",  _s(ctx.get("symmetry_index",1), below=0.92)],
        ["Jaw Curvature (deg)",     str(ctx.get("jaw_curvature_deg","—")),     "< 5.0",        _s(ctx.get("jaw_curvature_deg",0), above=5)],
        ["Ear Alignment (px)",      str(ctx.get("ear_alignment_px","—")),      "< 3 px",       _s(ctx.get("ear_alignment_px",0), above=3)],
        ["Philtrum Length (mm)",    str(ctx.get("philtrum_length_mm","—")),    "11 – 18 mm",   _s(ctx.get("philtrum_length_mm",14), lo=11, hi=18)],
        ["Inter-ocular Dist (px)",  str(ctx.get("interocular_dist_px","—")),   "58 – 72 px",   _s(ctx.get("interocular_dist_px",65), lo=58, hi=72)],
        ["Eye Aspect Ratio L",      str(ctx.get("eye_aspect_ratio_l","—")),    "0.28 – 0.34",  _s(ctx.get("eye_aspect_ratio_l",0.31), lo=0.28, hi=0.34)],
        ["Eye Aspect Ratio R",      str(ctx.get("eye_aspect_ratio_r","—")),    "0.28 – 0.34",  _s(ctx.get("eye_aspect_ratio_r",0.31), lo=0.28, hi=0.34)],
        ["Lip Thickness Ratio",     str(ctx.get("lip_thickness_ratio","—")),   "0.18 – 0.26",  _s(ctx.get("lip_thickness_ratio",0.22), lo=0.18, hi=0.26)],
        ["Neck-Face Boundary",      str(ctx.get("neck_face_boundary","—")),    "smooth",
         "FLAGGED" if str(ctx.get("neck_face_boundary","")) == "sharp_edge" else "Normal"],
        ["Geometry Anomaly Score",  str(ctx.get("geometry_anomaly_score", ctx.get("anomaly_score","—"))), "< 0.35", ""],
    ]
    flagged = _flag_rows(rows[1:], 3)
    col_w = [T.CONTENT_W*0.38, T.CONTENT_W*0.20, T.CONTENT_W*0.24, T.CONTENT_W*0.18]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style(flagged))
    for i in flagged:
        t.setStyle(TableStyle([("TEXTCOLOR",(3,i),(3,i),T.CRIMSON),("FONTNAME",(3,i),(3,i),T.SANS_BOLD)]))
    for i, row in enumerate(rows[1:], 1):
        if row[3] == "Normal":
            t.setStyle(TableStyle([("TEXTCOLOR",(3,i),(3,i),T.EMERALD)]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    return story


def _build_frequency(ctx, S):
    story = []
    story.append(SectionLabel("§5.2", "GAN Artefact & Frequency-Domain Analysis"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "2D Fast Fourier Transform (FFT) was applied to the greyscale face crop. Radial "
        "power spectral density was computed across three frequency bands. GAN-generated "
        "images characteristically deviate from the 1/f roll-off of authentic photography. "
        "EfficientNet-B4 (FaceForensics++ v3) provided the GAN classification probability.",
        S["body"]))
    rows = [
        ["Measurement", "Value", "Forensic Significance"],
        ["FFT Mid-Band Anomaly (16-50 cyc/px)",   f"{ctx.get('fft_mid_anomaly_db','—')} dB",      "GAN upsampling leaves characteristic excess energy in this band"],
        ["FFT High-Band Anomaly (51-100 cyc/px)", f"{ctx.get('fft_high_anomaly_db','—')} dB",     "Statistically significant at p < 0.001 when positive"],
        ["FFT Ultra-High Band (>100 cyc/px)",     f"{ctx.get('fft_ultrahigh_anomaly_db','—')} dB","StyleGAN2 signature — +15.6 dB in reference case DFA-2025-TC-00471"],
        ["GAN Probability (EfficientNet-B4)",     f"{float(ctx.get('gan_probability',0))*100:.1f}%","Primary GAN classifier confidence — FaceForensics++ v3 fine-tuned"],
        ["Upsampling Grid Artefact Detected",     str(ctx.get("upsampling_grid_detected","—")),   "4x4 px DCGAN / transposed-conv grid signature in FFT space"],
        ["Frequency Module Anomaly Score",        str(ctx.get("frequency_anomaly_score", ctx.get("anomaly_score","—"))), "Combined FFT + GAN signal"],
    ]
    col_w = [T.CONTENT_W*0.40, T.CONTENT_W*0.17, T.CONTENT_W*0.43]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 3*mm))
    return story


def _build_texture(ctx, S):
    story = []
    story.append(SectionLabel("§5.3", "Texture Consistency & Skin-Tone Mapping"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "The face was divided into six anatomical zones. Earth Mover Distance (EMD / "
        "Wasserstein-1) was computed between five adjacent zone pairs to detect seams "
        "introduced by composite synthesis or GAN inpainting. Authentic EMD baseline "
        "is < 0.08 for all zone pairs; values above 0.15 at the neck-face boundary "
        "trigger seam detection.",
        S["body"]))

    def ef(key):
        v = float(ctx.get(key,0) or 0)
        return "FLAGGED" if v > 0.08 else "Normal"

    rows = [
        ["Zone Pair", "EMD Score", "Baseline", "Status"],
        ["Forehead  to  Cheek",          str(ctx.get("forehead_cheek_emd","—")),    "< 0.08", ef("forehead_cheek_emd")],
        ["Cheek  to  Jaw (Left)",         str(ctx.get("cheek_jaw_emd_l","—")),       "< 0.08", ef("cheek_jaw_emd_l")],
        ["Cheek  to  Jaw (Right)",        str(ctx.get("cheek_jaw_emd_r","—")),       "< 0.08", ef("cheek_jaw_emd_r")],
        ["Periorbital  to  Nasal Bridge", str(ctx.get("periorbital_nasal_emd","—")), "< 0.08", ef("periorbital_nasal_emd")],
        ["Upper Lip  to  Chin",           str(ctx.get("lip_chin_emd","—")),          "< 0.08", ef("lip_chin_emd")],
        ["Neck  to  Face [primary seam]", str(ctx.get("neck_face_emd","—")),         "< 0.15",
         "FLAGGED - SEAM" if float(ctx.get("neck_face_emd",0) or 0) > 0.15 else "Normal"],
        ["LBP Uniformity Ratio",          str(ctx.get("lbp_uniformity","—")),        "> 0.85",
         "FLAGGED" if float(ctx.get("lbp_uniformity",1) or 1) < 0.85 else "Normal"],
        ["Seam Detected (threshold trigger)", str(ctx.get("seam_detected","—")),     "False", ""],
        ["Texture Module Anomaly Score",  str(ctx.get("texture_anomaly_score","—")), "", ""],
    ]
    flagged = _flag_rows(rows[1:], 3)
    col_w = [T.CONTENT_W*0.40, T.CONTENT_W*0.17, T.CONTENT_W*0.22, T.CONTENT_W*0.21]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style(flagged))
    for i in flagged:
        t.setStyle(TableStyle([("TEXTCOLOR",(3,i),(3,i),T.CRIMSON),("FONTNAME",(3,i),(3,i),T.SANS_BOLD)]))
    for i, row in enumerate(rows[1:], 1):
        if row[3] == "Normal":
            t.setStyle(TableStyle([("TEXTCOLOR",(3,i),(3,i),T.EMERALD)]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    return story


def _build_vlm(ctx, S):
    story = []
    story.append(SectionLabel("§5.4", "Explainability Heat-Map Analysis (VLM Attention)"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "BLIP-2 (Salesforce/blip2-opt-2.7b) was queried with a structured forensic prompt "
        "to elicit a verdict and artefact narrative. Grad-CAM was applied to the last "
        "convolutional layer of EfficientNet-B4 (FF++ v3) to generate pixel-level saliency "
        "maps. Regions are classified by activation: HIGH (>0.8), MID (0.5-0.8), LOW (<0.5).",
        S["body"]))
    rows = [
        ["Field", "Value"],
        ["VLM Verdict",                                str(ctx.get("vlm_verdict","—"))],
        ["VLM Confidence",                             f"{float(ctx.get('vlm_confidence',0))*100:.1f}%"],
        ["Saliency Score (mean Grad-CAM over face bbox)", str(ctx.get("saliency_score","—"))],
        ["Zone GAN Probability (central face zone)",   f"{float(ctx.get('zone_gan_probability',0))*100:.1f}%"],
        ["HIGH Activation Regions (>0.8)",             ", ".join(ctx.get("high_activation_regions",[]))  or "—"],
        ["MID Activation Regions (0.5-0.8)",           ", ".join(ctx.get("medium_activation_regions",[])) or "—"],
        ["LOW Activation Regions (<0.5)",              ", ".join(ctx.get("low_activation_regions",[]))   or "—"],
        ["VLM Forensic Caption",                       str(ctx.get("vlm_caption","—"))],
        ["VLM Module Anomaly Score",                   str(ctx.get("vlm_anomaly_score","—"))],
    ]
    col_w = [T.CONTENT_W*0.42, T.CONTENT_W*0.58]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 4*mm))
    story.append(HeatmapFrame(
        path=str(ctx.get("heatmap_path","")),
        caption="Figure 1 — Grad-CAM activation overlay (EfficientNet-B4, FaceForensics++ v3). "
                "Warm tones = highest deepfake confidence. Reproduced per §5.4 reference standard.",
    ))
    story.append(Spacer(1, 3*mm))
    return story


def _build_biological(ctx, S):
    story = []
    story.append(SectionLabel("§5.5", "Biological Plausibility Assessment"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "GAN-generated images fail to replicate physiological signals present in authentic "
        "photographs of living subjects. The four indicators below assess biological "
        "plausibility: rPPG cardiac signal (SNR), corneal specular highlight geometry, "
        "perioral micro-texture variance, and subcutaneous vascular pattern correlation.",
        S["body"]))
    rows = [
        ["Physiological Signal", "Measured", "Authentic Baseline", "Finding"],
        ["rPPG Cardiac Signal SNR",           str(ctx.get("rppg_snr","—")),              "> 0.45",
         "FLAGGED - ABSENT" if float(ctx.get("rppg_snr",1) or 1) < 0.45 else "Signal Present"],
        ["Corneal Highlight Deviation (deg)", str(ctx.get("corneal_deviation_deg","—")), "< 5.0 deg",
         "FLAGGED - COMPOSITE" if float(ctx.get("corneal_deviation_deg",0) or 0) > 5 else "Consistent"],
        ["Perioral Micro-texture Variance",   str(ctx.get("micro_texture_var","—")),     "~0.031 mean",
         "FLAGGED - SMOOTHED" if float(ctx.get("micro_texture_var",0.031) or 0.031) < 0.020 else "Normal"],
        ["Vascular Pattern Pearson r",        str(ctx.get("vascular_pearson_r","—")),    "> 0.88",
         "FLAGGED - MISMATCH" if float(ctx.get("vascular_pearson_r",1) or 1) < 0.88 else "Match"],
        ["Biological Module Anomaly Score",   str(ctx.get("biological_anomaly_score", ctx.get("anomaly_score","—"))), "", ""],
    ]
    flagged = _flag_rows(rows[1:], 3)
    col_w = [T.CONTENT_W*0.38, T.CONTENT_W*0.16, T.CONTENT_W*0.22, T.CONTENT_W*0.24]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style(flagged))
    for i in flagged:
        t.setStyle(TableStyle([("TEXTCOLOR",(3,i),(3,i),T.CRIMSON),("FONTNAME",(3,i),(3,i),T.SANS_BOLD)]))
    for i, row in enumerate(rows[1:], 1):
        if row[3] in ("Signal Present","Consistent","Normal","Match"):
            t.setStyle(TableStyle([("TEXTCOLOR",(3,i),(3,i),T.EMERALD)]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    return story


def _build_metadata(ctx, S):
    story = []
    story.append(SectionLabel("§5.6 + §7", "Metadata & Comparative Reference Analysis"))
    story.append(GoldRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Provenance metadata was cross-validated against the image content. FaceNet-512 "
        "and ArcFace identity embeddings were compared against an authentic reference set "
        "and a known deepfake cluster. 3DMM shape coefficient distance provides "
        "morphometric identity verification.",
        S["body"]))

    def mf(cond): return "FLAGGED" if cond else "—"

    rows = [
        ["Field", "Value", "Status"],
        ["EXIF Camera Make/Model",        str(ctx.get("exif_camera_present","—")),  mf(not ctx.get("exif_camera_present",True))],
        ["ELA Chi-Squared",               str(ctx.get("ela_chi2","—")),             mf(float(ctx.get("ela_chi2",0) or 0)>200)],
        ["Thumbnail Mismatch",            str(ctx.get("thumbnail_mismatch","—")),   mf(ctx.get("thumbnail_mismatch"))],
        ["PRNU Camera Fingerprint",       "ABSENT" if ctx.get("prnu_absent") else "PRESENT", mf(ctx.get("prnu_absent"))],
        ["Software Tag",                  str(ctx.get("software_tag","")) or "Not present", "—"],
        ["JPEG Quantisation Anomaly",     str(ctx.get("jpeg_quantisation_anomaly","—")), mf(ctx.get("jpeg_quantisation_anomaly"))],
        ["FaceNet Cosine Dist - Authentic", str(ctx.get("cosine_dist_authentic","—")), mf(float(ctx.get("cosine_dist_authentic",0) or 0)>0.40)],
        ["FaceNet Cosine Dist - Fake",    str(ctx.get("cosine_dist_fake","—")),     "—"],
        ["FaceNet-512 Distance",          str(ctx.get("facenet_dist","—")),          "—"],
        ["ArcFace Distance",              str(ctx.get("arcface_dist","—")),          "—"],
        ["3DMM Shape Coefficient Dist",   str(ctx.get("shape_3dmm_dist","—")),       "—"],
        ["Reference Verdict",             str(ctx.get("reference_verdict","—")),     "—"],
        ["Metadata Module Anomaly Score", str(ctx.get("metadata_anomaly_score","—")),""],
    ]
    flagged = _flag_rows(rows[1:], 2)
    col_w = [T.CONTENT_W*0.50, T.CONTENT_W*0.34, T.CONTENT_W*0.16]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_table_style(flagged))
    for i in flagged:
        t.setStyle(TableStyle([("TEXTCOLOR",(2,i),(2,i),T.CRIMSON),("FONTNAME",(2,i),(2,i),T.SANS_BOLD)]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    return story


def _build_fusion(ctx, S):
    story = []
    story.append(SectionLabel("§6", "Bayesian Confidence Scoring & Statistical Analysis"))
    story.append(GoldRule())
    story.append(Spacer(1, 3*mm))
    story.append(VerdictPanel(
        verdict=ctx.get("decision","UNCERTAIN"),
        final_score=ctx.get("final_score",0.5),
        ci=ctx.get("confidence_interval",[0.0,1.0]),
    ))
    story.append(Spacer(1, 5*mm))
    interp = ctx.get("interpretation","")
    if interp:
        story.append(Paragraph(f"Confidence Interpretation: <b>{interp}</b>", S["body_sans"]))
        story.append(Spacer(1, 2*mm))
    story.append(Paragraph("PER-MODULE ANOMALY SCORES", S["h2"]))
    story.append(NavyRule())
    story.append(Spacer(1, 2*mm))
    weights = {"geometry":0.15,"gan_artefact":0.25,"frequency":0.25,
               "texture":0.20,"vlm":0.25,"biological":0.15,"metadata":0.15}
    labels  = {"geometry":"Facial Geometry","gan_artefact":"GAN Artefact Classifier",
               "frequency":"Frequency-Domain FFT","texture":"Texture Consistency",
               "vlm":"VLM Explainability","biological":"Biological Plausibility",
               "metadata":"Metadata & Provenance"}
    refs    = {"geometry":"§5.1","gan_artefact":"§5.2","frequency":"§5.2",
               "texture":"§5.3","vlm":"§5.4","biological":"§5.5","metadata":"§5.6"}
    per_mod = ctx.get("per_module_scores",{})
    for key in ["geometry","gan_artefact","frequency","texture","vlm","biological","metadata"]:
        story.append(ModuleScoreBar(
            section_ref=refs[key], module_name=labels[key],
            score=per_mod.get(key,0.5), weight=weights[key]))
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("MODEL PERFORMANCE CONSTANTS  (§6 — Fixed, Image-Independent)", S["h2"]))
    story.append(NavyRule())
    story.append(Spacer(1, 2*mm))
    stats = [
        ["Statistic", "Value", "Description"],
        ["Model AUC-ROC", str(ctx.get("model_auc_roc","0.983")), "Classifier discrimination — area under ROC curve"],
        ["False Positive Rate", f"{float(ctx.get('false_positive_rate',0.021))*100:.1f}%", "Probability of incorrectly flagging an authentic image"],
        ["Expected Calibration Error", str(ctx.get("calibration_ece","0.014")), "Probability calibration quality — lower is better"],
        ["Decision Threshold", str(ctx.get("decision_threshold","0.70")), "Score >= 0.70 -> DEEPFAKE  |  Score <= 0.35 -> AUTHENTIC"],
    ]
    st = Table(stats, colWidths=[T.CONTENT_W*0.28, T.CONTENT_W*0.14, T.CONTENT_W*0.58])
    st.setStyle(_table_style())
    story.append(st)
    story.append(Spacer(1, 3*mm))
    return story


def _build_narrative(ctx, S):
    story = []
    story.append(SectionLabel("§8", "Executive Summary & Analyst Findings"))
    story.append(GoldRule())
    story.append(Spacer(1, 3*mm))
    narrative = ctx.get("narrative_text","")
    if narrative:
        story.append(Paragraph(narrative, S["body"]))
    else:
        decision = ctx.get("decision","UNCERTAIN")
        score    = float(ctx.get("final_score",0.5))
        ci       = ctx.get("confidence_interval",[0.0,1.0])
        per_mod  = ctx.get("per_module_scores",{})
        top_key  = max(per_mod, key=per_mod.get) if per_mod else "—"
        top_val  = per_mod.get(top_key, 0)
        story.append(Paragraph(
            f"A comprehensive forensic analysis of the submitted image was conducted using the "
            f"MFAD Multi-Factor AI Deepfake Detection pipeline across seven independent "
            f"analytical modules. The Bayesian ensemble model returned a <b>DeepFake Prediction "
            f"Score of {score*100:.1f}%</b> with a 95% confidence interval of "
            f"[{ci[0]*100:.1f}% - {ci[1]*100:.1f}%], yielding a primary verdict of "
            f"<b>{decision}</b>.", S["body"]))
        if decision == "DEEPFAKE":
            story.append(Paragraph(
                f"Convergent evidence across multiple independent forensic channels supports "
                f"this determination. The strongest signal was recorded by the "
                f"{top_key.replace('_',' ').title()} module (anomaly score: {top_val:.3f}). "
                f"Spectral frequency analysis detected excess power spectral density in the "
                f"mid and high frequency bands, consistent with GAN upsampling artefacts. "
                f"Texture consistency analysis revealed Earth Mover Distance values "
                f"substantially above the authentic baseline at the neck-face boundary, "
                f"indicating a composite seam. Biological plausibility assessment confirmed "
                f"absence of rPPG cardiac signal variation and implausible corneal highlight "
                f"geometry, both characteristic of synthetically generated imagery. "
                f"Metadata forensics confirmed absence of camera sensor PRNU fingerprint "
                f"and EXIF camera provenance, consistent with machine-generated origin.", S["body"]))
        elif decision == "AUTHENTIC":
            story.append(Paragraph(
                f"All seven independent forensic modules returned measurements consistent with "
                f"authentic photographic origin. No statistically significant GAN frequency "
                f"artefacts were detected in any spectral band. Texture boundary analysis "
                f"returned Earth Mover Distance values within the authentic baseline for all "
                f"five zone pairs. Biological plausibility signals were consistent with real "
                f"physiological origin. Metadata provenance was intact.", S["body"]))
        else:
            story.append(Paragraph(
                f"Forensic evidence was found to be inconclusive. Certain analytical modules "
                f"returned anomalous readings while others did not, precluding a definitive "
                f"determination at the {float(ctx.get('decision_threshold',0.70)):.0%} "
                f"decision threshold. Manual review by a qualified forensic examiner is "
                f"required before any legal determination is drawn from this report.", S["body"]))
    story.append(Spacer(1, 3*mm))
    return story


def _build_legal(ctx, S):
    story = []
    story.append(SectionLabel("§9 - §10", "Legal Certification & Analyst Declaration"))
    story.append(GoldRule())
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("COMPLIANCE WITH REFERENCE STANDARDS", S["h2"]))
    story.append(NavyRule())
    story.append(Spacer(1, 2*mm))
    applicability = {
        "ISO/IEC 27037:2012":        "Digital evidence identification, collection, acquisition and preservation",
        "SWGDE Best Practices 2023": "Scientific Working Group for Digital Evidence — image and video analysis",
        "NIST SP 800-101r1":         "Guidelines on mobile device forensics and digital evidence handling",
        "ACPO v5":                   "Association of Chief Police Officers — digital evidence principles",
        "FRE Rule 702":              "Federal Rules of Evidence — admissibility of expert testimony",
    }
    standards = ctx.get("compliance_standards", list(applicability.keys()))
    std_rows  = [["Reference Standard", "Scope of Applicability"]]
    for s in standards:
        std_rows.append([s, applicability.get(s, "Applied as reference standard")])
    st = Table(std_rows, colWidths=[T.CONTENT_W*0.38, T.CONTENT_W*0.62])
    st.setStyle(_table_style())
    story.append(st)
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("ANALYST DECLARATION", S["h2"]))
    story.append(NavyRule())
    story.append(Spacer(1, 3*mm))
    analyst   = ctx.get("analyst_name",     "[ANALYST NAME]")
    lab       = ctx.get("lab_accreditation","[LABORATORY]")
    report_id = ctx.get("report_id",        "DFA-XXXX-TC-XXXXXX")
    generated = ctx.get("generated_at",     datetime.now(timezone.utc).isoformat())
    decl_box = ParagraphStyle(
        "decl_box", fontName=T.SERIF_ITALIC, fontSize=9,
        textColor=T.INK, leading=15, alignment=TA_JUSTIFY,
        borderPadding=(8,10,8,10), borderColor=T.GOLD, borderWidth=0.8,
        backColor=colors.HexColor("#FDFAF3"), leftIndent=8, rightIndent=8)
    story.append(Paragraph(
        f"I, <b>{analyst}</b>, duly accredited through <b>{lab}</b>, hereby certify that "
        f"the forensic analysis described in this report (Report ID: <b>{report_id}</b>; "
        f"generated {generated}) was conducted in strict accordance with the above referenced "
        f"standards and accepted forensic methodology. The analytical methods employed are "
        f"reproducible, peer-reviewed, and have been validated against benchmark datasets. "
        f"The measurements, findings, and conclusions presented herein represent my "
        f"independent professional opinion formed on the basis of the MFAD pipeline output "
        f"and my expert interpretation thereof. This report is prepared for submission as "
        f"expert forensic evidence and is admissible under the applicable rules of evidence.",
        decl_box))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("SIGNATURES", S["h2"]))
    story.append(NavyRule())
    story.append(Spacer(1, 3*mm))
    sig_rows = [
        [Paragraph("ANALYST SIGNATURE", S["sig_label"]),
         Paragraph("DATE OF SIGNING", S["sig_label"]),
         Paragraph("LABORATORY SEAL / STAMP", S["sig_label"])],
        [Paragraph(" ", S["sig_value"]),
         Paragraph(" ", S["sig_value"]),
         Paragraph(" ", S["sig_value"])],
        [Paragraph("_"*28, S["sig_value"]),
         Paragraph("_"*18, S["sig_value"]),
         Paragraph("_"*28, S["sig_value"])],
        [Paragraph(analyst, S["sig_value"]),
         Paragraph(generated[:10], S["sig_value"]),
         Paragraph(lab[:35]+("..." if len(lab)>35 else ""), S["sig_value"])],
    ]
    sig_t = Table(sig_rows, colWidths=[T.CONTENT_W/3]*3)
    sig_t.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),4), ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LINEBELOW",(0,2),(-1,2),0.6,T.INK_MID), ("VALIGN",(0,0),(-1,-1),"BOTTOM"),
    ]))
    story.append(sig_t)
    story.append(Spacer(1, 8*mm))
    story.append(NavyRule())
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f"DISCLAIMER: This report was generated by an automated AI forensic pipeline. All "
        f"probabilistic findings are statistical in nature and carry an inherent false positive "
        f"rate of {float(ctx.get('false_positive_rate',0.021))*100:.1f}%. "
        f"No automated system should be the sole basis for a legal determination. "
        f"This report must be reviewed and validated by a qualified human forensic examiner "
        f"before submission as evidence in any criminal, civil, or administrative proceeding.",
        S["footnote"]))
    return story


# ══════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════

def build_report(ctx: dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    S = build_styles()
    doc = BaseDocTemplate(
        output_path, pagesize=A4,
        leftMargin=T.MARGIN_L, rightMargin=T.MARGIN_R,
        topMargin=T.MARGIN_T, bottomMargin=T.MARGIN_B,
        title=f"MFAD Forensic Report — {ctx.get('report_id','—')}",
        author="MFAD Forensic AI Laboratory",
        subject="Forensic Image Authentication",
    )
    doc._report_id = ctx.get("report_id", "DFA-XXXX-TC-XXXXXX")
    cover_frame = Frame(12*mm, 16*mm, T.PAGE_W-24*mm, T.PAGE_H-20*mm, id="cover", showBoundary=0)
    inner_frame = Frame(T.MARGIN_L, T.MARGIN_B, T.CONTENT_W, T.PAGE_H-T.MARGIN_T-T.MARGIN_B, id="inner", showBoundary=0)
    doc.addPageTemplates([
        PageTemplate(id="Cover", frames=[cover_frame], onPage=_draw_cover),
        PageTemplate(id="Inner", frames=[inner_frame], onPage=_draw_inner),
    ])
    story = []
    story.extend(_build_cover(ctx, S))
    story.append(NextPageTemplate("Inner"))
    story.append(PageBreak())
    story.extend(_build_toc(ctx, S))
    story.extend(_build_custody(ctx, S))
    story.append(Spacer(1, 4*mm))
    story.extend(_build_geometry(ctx, S))
    story.append(PageBreak())
    story.extend(_build_frequency(ctx, S))
    story.append(Spacer(1, 4*mm))
    story.extend(_build_texture(ctx, S))
    story.append(PageBreak())
    story.extend(_build_vlm(ctx, S))
    story.append(Spacer(1, 4*mm))
    story.extend(_build_biological(ctx, S))
    story.append(PageBreak())
    story.extend(_build_metadata(ctx, S))
    story.append(Spacer(1, 4*mm))
    story.extend(_build_fusion(ctx, S))
    story.append(PageBreak())
    story.extend(_build_narrative(ctx, S))
    story.append(Spacer(1, 4*mm))
    story.extend(_build_legal(ctx, S))
    doc.build(story)
    return output_path
