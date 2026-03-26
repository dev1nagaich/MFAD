"""
report_agent/generate.py — MFAD Multi-Factor AI Deepfake Detection System
==========================================================================
ReportGenerator agent — §8 Legal Certification + §9–§10 Analyst Declaration.

This is the FINAL agent in the pipeline. It receives the fully merged ctx dict
from master_agent.py (all 7 agent outputs + Bayesian fusion result) and:

  Step A — Generates a unique report ID and timestamp
  Step B — Calls Mistral-7B via Ollama to write the narrative text sections
  Step C — Calls build_report() from template.py to render the final PDF
  Step D — Validates output and returns REPORT_KEYS dict to master_agent

No models are trained here. Two tools are used:
  1. ReportLab  — PDF drawing library (handled entirely in template.py)
  2. Mistral-7B — Local LLM via Ollama for narrative text generation

Dependencies:
    pip install ollama reportlab

Ollama one-time setup:
    curl -fsSL https://ollama.com/install.sh | sh   (Mac / Linux)
    Then download installer from https://ollama.com  (Windows)

    ollama pull mistral      <- downloads ~4 GB once, never again
    ollama serve             <- starts the local server (keep running)
"""

import datetime
import os
import sys
import random

# ── Path setup so imports work from anywhere ──────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from contracts import validate, REPORT_KEYS
from report_agent.template import build_report


# ══════════════════════════════════════════════════════════════
#  OLLAMA NARRATIVE GENERATOR
#  Calls Mistral-7B locally to produce the narrative sections.
#  Falls back to empty string so template.py uses its auto-text.
# ══════════════════════════════════════════════════════════════

def _call_ollama(prompt: str, model: str = "mistral") -> str:
    """
    Send a prompt to Mistral-7B running locally via Ollama.

    Args:
        prompt: The full text prompt to send.
        model:  Ollama model name. "mistral" is fine for this use case.

    Returns:
        The model's text response as a plain string.
        Returns empty string if Ollama is not available.

    How Ollama works under the hood:
        - Ollama runs a local HTTP server on port 11434
        - The 'ollama' Python library sends your prompt to that server
        - The server runs Mistral-7B on your GPU (or CPU if no GPU)
        - The response comes back as text
        - Nothing leaves your machine — fully offline
    """
    try:
        import ollama  # pip install ollama

        response = ollama.chat(
            model=model,
            messages=[
                {
                    # System message — sets the role and writing style
                    # This is like giving instructions to the LLM before it starts
                    "role": "system",
                    "content": (
                        "You are a senior forensic analyst writing an official court-grade "
                        "forensic evidence report. Your writing must be precise, formal, "
                        "authoritative, and technically accurate. Write in the third person. "
                        "Do not use bullet points. Write only in continuous prose paragraphs. "
                        "Do not add any preamble, headings, or sign-off. "
                        "Only write the requested paragraphs and nothing else."
                    ),
                },
                {
                    # User message — the actual prompt with all the case data
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # The response comes back as a dict — extract the text
        return response["message"]["content"].strip()

    except ImportError:
        # The ollama Python library is not installed
        print(
            "[ReportGenerator] WARNING: 'ollama' Python library is not installed.\n"
            "  Run:  pip install ollama\n"
            "  Falling back to auto-generated narrative text in template.py."
        )
        return ""

    except Exception as e:
        # Ollama server not running, model not downloaded, GPU out of memory, etc.
        print(
            f"[ReportGenerator] WARNING: Ollama call failed — {e}\n"
            "  Make sure Ollama is running:  ollama serve\n"
            "  Make sure model is pulled:    ollama pull mistral\n"
            "  Falling back to auto-generated narrative text in template.py."
        )
        return ""


def _build_prompt(ctx: dict) -> str:
    """
    Build the full prompt that gets sent to Mistral-7B.

    The key insight here: we inject ALL the real measurements from ctx
    into the prompt so Mistral-7B writes about the ACTUAL findings of
    THIS specific image — not a generic template.

    Every number that goes into this prompt comes from the real agents.
    The LLM turns those numbers into readable forensic English.
    """

    # Pull every key measurement from ctx
    decision    = ctx.get("decision",                    "UNCERTAIN")
    score       = float(ctx.get("final_score",           0.5))
    ci          = ctx.get("confidence_interval",         [0.0, 1.0])
    per_mod     = ctx.get("per_module_scores",           {})
    report_id   = ctx.get("report_id",                   "—")

    # §5.1 Geometry
    sym         = ctx.get("symmetry_index",              "N/A")
    jaw         = ctx.get("jaw_curvature_deg",           "N/A")
    ear         = ctx.get("ear_alignment_px",            "N/A")
    neck_bound  = ctx.get("neck_face_boundary",          "N/A")

    # §5.2 Frequency
    fft_mid     = ctx.get("fft_mid_anomaly_db",          "N/A")
    fft_high    = ctx.get("fft_high_anomaly_db",         "N/A")
    fft_ultra   = ctx.get("fft_ultrahigh_anomaly_db",    "N/A")
    gan_prob    = float(ctx.get("gan_probability",       0))
    grid        = ctx.get("upsampling_grid_detected",    False)

    # §5.3 Texture
    neck_emd    = ctx.get("neck_face_emd",               "N/A")
    lbp         = ctx.get("lbp_uniformity",              "N/A")
    seam        = ctx.get("seam_detected",               False)

    # §5.4 VLM
    vlm_verdict = ctx.get("vlm_verdict",                 "N/A")
    vlm_conf    = float(ctx.get("vlm_confidence",        0))
    vlm_caption = ctx.get("vlm_caption",                 "N/A")
    saliency    = ctx.get("saliency_score",              "N/A")

    # §5.5 Biological
    rppg        = ctx.get("rppg_snr",                    "N/A")
    corneal     = ctx.get("corneal_deviation_deg",       "N/A")
    micro_tex   = ctx.get("micro_texture_var",           "N/A")
    vascular    = ctx.get("vascular_pearson_r",          "N/A")

    # §5.6 + §7 Metadata
    exif        = ctx.get("exif_camera_present",         False)
    ela         = ctx.get("ela_chi2",                    "N/A")
    prnu        = ctx.get("prnu_absent",                 False)
    jpeg_q      = ctx.get("jpeg_quantisation_anomaly",   False)
    ref_verdict = ctx.get("reference_verdict",           "N/A")
    cos_auth    = ctx.get("cosine_dist_authentic",       "N/A")

    prompt = f"""
Write the forensic narrative for report {report_id}.

CASE DATA:
Verdict: {decision}
DeepFake Prediction Score: {score * 100:.1f}%
95% Confidence Interval: {ci[0]*100:.1f}% to {ci[1]*100:.1f}%
Per-module anomaly scores: {per_mod}

MEASUREMENTS FROM EACH FORENSIC MODULE:

Facial Geometry (§5.1):
  Symmetry index: {sym}  [authentic range: 0.92 to 1.00]
  Jaw curvature: {jaw} degrees  [authentic: below 5 degrees]
  Ear alignment: {ear} px  [authentic: below 3 px]
  Neck-face boundary: {neck_bound}  [authentic: smooth]

Frequency and GAN Analysis (§5.2):
  FFT mid-band anomaly (16-50 cycles/px): {fft_mid} dB
  FFT high-band anomaly (51-100 cycles/px): {fft_high} dB
  FFT ultra-high band anomaly (above 100 cycles/px): {fft_ultra} dB
  GAN probability from EfficientNet-B4 FaceForensics++ v3: {gan_prob * 100:.1f}%
  Upsampling grid artefact detected: {grid}

Texture Consistency (§5.3):
  Neck-to-face boundary EMD: {neck_emd}  [authentic below 0.08; seam threshold 0.15]
  LBP uniformity: {lbp}  [authentic above 0.85]
  Seam detected: {seam}

VLM Explainability (§5.4):
  BLIP-2 verdict: {vlm_verdict}  [confidence: {vlm_conf * 100:.1f}%]
  Saliency score: {saliency}
  BLIP-2 caption: {vlm_caption}

Biological Plausibility (§5.5):
  rPPG cardiac SNR: {rppg}  [authentic above 0.45]
  Corneal highlight deviation: {corneal} degrees  [authentic below 5 degrees]
  Perioral micro-texture variance: {micro_tex}  [authentic mean approximately 0.031]
  Vascular Pearson r: {vascular}  [authentic above 0.88]

Metadata and Reference Comparison (§5.6 + §7):
  EXIF camera metadata present: {exif}
  ELA chi-squared: {ela}
  PRNU camera fingerprint absent: {prnu}
  JPEG quantisation anomaly: {jpeg_q}
  FaceNet cosine distance to authentic reference: {cos_auth}  [same person below 0.40]
  Reference verdict: {ref_verdict}

TASK:
Write exactly two paragraphs as described below. No headings, no bullet points.

PARAGRAPH 1 — Executive Summary (for a judge, lawyer, or investigator who has no technical background):
Explain what the image is suspected to be, what the overall finding is, how confident the system is,
and what the practical implication of this finding is. Use plain English. 4 to 6 sentences.

PARAGRAPH 2 — Technical Findings (for a forensic expert reviewing the evidence):
Describe the specific measurements that deviate most significantly from authentic baselines.
Explain what each deviation indicates forensically. Explain how the convergence of findings
across multiple independent analytical modules supports the primary determination.
Reference specific module scores and measurements. 6 to 8 sentences.

Write only the two paragraphs separated by one blank line. Nothing else.
""".strip()

    return prompt


def _generate_narrative(ctx: dict, model: str = "mistral") -> str:
    """
    Generate narrative text for the report using Mistral-7B.

    Args:
        ctx:   Fully merged context dict.
        model: Ollama model name to use.

    Returns:
        String containing the two narrative paragraphs.
        Empty string if Ollama is unavailable (template.py handles fallback).
    """
    prompt = _build_prompt(ctx)
    text   = _call_ollama(prompt, model=model)
    return text


# ══════════════════════════════════════════════════════════════
#  REPORT ID GENERATOR
#  Format:  DFA-{YYYY}-TC-{6-char uppercase hex}
#  Example: DFA-2026-TC-A3F1B2
# ══════════════════════════════════════════════════════════════

def _new_report_id() -> str:
    year   = datetime.datetime.now(datetime.timezone.utc).year
    hex_id = "".join(random.choices("0123456789ABCDEF", k=6))
    return f"DFA-{year}-TC-{hex_id}"


# ══════════════════════════════════════════════════════════════
#  REPORT GENERATOR AGENT
# ══════════════════════════════════════════════════════════════

class ReportGenerator:
    """
    Final agent in the MFAD pipeline.
    Maps to: §8 Legal Certification + §9-§10 Analyst Declaration.

    This agent takes no image input and runs no computer vision.
    It is a pure orchestration agent:
        - reads numbers from ctx
        - sends them to Mistral-7B for narrative generation
        - passes everything to template.py for PDF rendering
        - returns the REPORT_KEYS contract to master_agent

    Configuration:
        Set ANALYST_NAME and LAB_ACCREDIT before deploying.
        These will come from a config file in production.

    Ollama requirement:
        Mistral-7B must be running via Ollama for narrative generation.
        If not available, the report still generates — template.py
        produces auto-generated narrative text as fallback.
    """

    # ── Deployment config ─────────────────────────────────────
    # Change these to match your actual analyst and lab details.
    ANALYST_NAME  = "To be configured in deployment"
    LAB_ACCREDIT  = "MFAD Forensic AI Laboratory"
    REPORTS_DIR   = "reports"
    OLLAMA_MODEL  = "mistral"   # or "mistral:7b-instruct" for the instruction-tuned version

    COMPLIANCE_STANDARDS = [
        "ISO/IEC 27037:2012",
        "SWGDE Best Practices 2023",
        "NIST SP 800-101r1",
        "ACPO Good Practice Guide v5",
        "FRE Rule 702",
    ]

    def generate(self, ctx: dict) -> dict:
        """
        Main entry point. Called by master_agent.py after all other agents complete.

        This method is intentionally simple — it just orchestrates the three steps.
        All the complexity lives in _generate_narrative() and build_report().

        Args:
            ctx: Fully merged context dict from master_agent.py.
                 Contains every output key from every agent that ran before this.

        Returns:
            dict matching REPORT_KEYS contract.
        """

        # ── Step A: Generate report ID and timestamp ──────────
        report_id   = _new_report_id()
        generated   = datetime.datetime.now(datetime.timezone.utc).isoformat()
        os.makedirs(self.REPORTS_DIR, exist_ok=True)
        report_path = os.path.join(self.REPORTS_DIR, f"{report_id}.pdf")

        print(f"\n  [ReportGenerator] ─────────────────────────────────")
        print(f"  [ReportGenerator] Report ID  : {report_id}")
        print(f"  [ReportGenerator] Output     : {report_path}")
        print(f"  [ReportGenerator] Decision   : {ctx.get('decision','—')}")
        print(f"  [ReportGenerator] Final Score: {ctx.get('final_score','—')}")

        # ── Step B: Narrative text via Mistral-7B ─────────────
        # This is the ONLY step that uses a language model.
        # If Mistral is not available, narrative = "" and template.py
        # generates the summary text automatically from the numbers.
        print("  [ReportGenerator] Requesting narrative from Mistral-7B...")
        narrative = _generate_narrative(ctx, model=self.OLLAMA_MODEL)

        if narrative:
            print("  [ReportGenerator] Narrative generated by Mistral-7B.")
        else:
            print("  [ReportGenerator] Using template.py auto-narrative (Ollama unavailable).")

        # ── Step C: Render PDF via template.py ────────────────
        # We copy ctx and add report metadata + narrative before passing to template.
        # We copy (don't mutate) because master_agent.py still holds a reference to ctx.
        ctx_for_pdf = dict(ctx)
        ctx_for_pdf.update({
            "report_id":            report_id,
            "generated_at":         generated,
            "analyst_name":         self.ANALYST_NAME,
            "lab_accreditation":    self.LAB_ACCREDIT,
            "compliance_standards": self.COMPLIANCE_STANDARDS,
            "narrative_text":       narrative,  # empty string = template uses auto-text
        })

        print("  [ReportGenerator] Rendering PDF with ReportLab...")
        build_report(ctx_for_pdf, report_path)
        print(f"  [ReportGenerator] PDF saved to: {report_path}")

        # ── Step D: Verify integrity hash ─────────────────────
        # Check that the SHA-256 hash recorded at intake is not empty or a stub value.
        # In production this would re-hash the file and compare — for now we check
        # that a real hash was recorded.
        intake_hash   = ctx.get("hash_sha256", "")
        hash_verified = bool(intake_hash) and intake_hash != "stub_sha256_hash"

        # ── Build and validate the output dict ────────────────
        output = {
            "report_path":          report_path,
            "report_id":            report_id,
            "generated_at":         generated,
            "compliance_standards": self.COMPLIANCE_STANDARDS,
            "analyst_name":         self.ANALYST_NAME,
            "lab_accreditation":    self.LAB_ACCREDIT,
            "hash_sha256_verified": hash_verified,
        }

        validate(output, REPORT_KEYS, "ReportGenerator")
        return output
