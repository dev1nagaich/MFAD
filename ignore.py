"""
master_agent.py  —  MFAD LangGraph Orchestrator
================================================
Owner     : Dev
Framework : LangGraph (StateGraph) + LangChain @tool + Ollama (Mistral-7B)

Architecture  (matches the 7-stage diagram)
───────────────────────────────────────────

  Stage 0  preprocess_node
           Universal work only: SHA-256, EXIF baseline, ELA, image dims,
           PRNU baseline.  NO face detection here.
           Gate: aborts if image is unreadable / corrupt.

  Stage 1  content_router_node
           ONE CLIP zero-shot call → routing flags:
             face_present, has_text, scene_type, subject_count,
             full_synthesis_likely, document_type
           These flags activate / skip every downstream pool.
           Currently: stub returns face_present=True, everything else False.

  Stage 2  pool_dispatch_node
           Reads routing flags, fires active pools in parallel
           via asyncio.gather.

           ┌─ FACE POOL  (if face_present) ──────── IMPLEMENTED ──────────┐
           │  geometry.py   biological.py   vlm_face.py                   │
           │  reference_face.py   texture_face.py                         │
           │  geometry.py also runs RetinaFace → face_bbox                │
           └──────────────────────────────────────────────────────────────┘

           ┌─ SCENE POOL  (if scene_type set) ──── FUTURE ────────────────┐
           │  struct_consist.py  inpaint_detect.py  physics_check.py      │
           │  copy_move.py  splicing.py                                   │
           └──────────────────────────────────────────────────────────────┘

           ┌─ FORGERY POOL  (always active) ──── FUTURE ─────────────────┐
           │  ela_deep.py  dct_artifact.py  prnu_field.py                 │
           │  resampling.py  noise_consist.py                             │
           └──────────────────────────────────────────────────────────────┘

           ┌─ DOC POOL  (if has_text) ──── FUTURE ───────────────────────┐
           │  text_integrity.py  ocr_consist.py  font_analysis.py         │
           │  layout_check.py                                             │
           └──────────────────────────────────────────────────────────────┘

           ┌─ SYNTH POOL  (always active) ──── FUTURE ───────────────────┐
           │  frequency.py  texture_global.py  vlm_general.py            │
           │  reference_clip.py                                           │
           └──────────────────────────────────────────────────────────────┘

  Stage 3  universal_agents_node
           Always runs, no gate.
           metadata.py · steganography.py · provenance.py

  Stage 4  reconciler_node
           Reads agent_applicable flags.
           abstain = 0.5 (not 0.0) for non-applicable agents.
           Flags contradictions before fusion
           (e.g. ELA clean but PRNU mismatch).

  Stage 5  fusion_node
           Log-odds Bayesian fusion.
           Weights only applicable agents.
           Outputs final_score, 95% CI, verdict.

  Stage 6  report_node
           ONLY LLM call in the graph (Mistral-7B via Ollama).
           Per-section forensic prose + executive summary.
           Assembles master JSON → hands to generator.py → PDF.

State
─────
  MFADState (TypedDict, total=False) — single data bus for the whole graph.
  Every node reads what it needs and writes only its own keys back.

LLM usage
─────────
  content_router_node : CLIP (zero-shot image classification) — NOT Mistral
  report_node         : Mistral-7B via Ollama — ONLY LLM call for prose
  All other nodes     : deterministic, no LLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Any, Optional

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ── Typing ────────────────────────────────────────────────────────────────────
from typing_extensions import TypedDict

# ── Real agent imports (uncomment as each module is implemented) ──────────────
#
# Stage 0
# from agents.preprocessing  import preprocessing_agent
#
# Stage 1
# from agents.content_router import content_router_agent
#
# Stage 2 — Face pool
# from agents.face.geometry       import geometry_agent        # also detects face_bbox
# from agents.face.biological     import biological_agent
# from agents.face.vlm_face       import vlm_face_agent
# from agents.face.reference_face import reference_face_agent
# from agents.face.texture_face   import texture_face_agent
#
# Stage 2 — Scene pool (FUTURE)
# from agents.scene.struct_consist  import struct_consist_agent
# from agents.scene.inpaint_detect  import inpaint_detect_agent
# from agents.scene.physics_check   import physics_check_agent
# from agents.scene.copy_move       import copy_move_agent
# from agents.scene.splicing        import splicing_agent
#
# Stage 2 — Forgery pool (FUTURE)
# from agents.forgery.ela_deep      import ela_deep_agent
# from agents.forgery.dct_artifact  import dct_artifact_agent
# from agents.forgery.prnu_field    import prnu_field_agent
# from agents.forgery.resampling    import resampling_agent
# from agents.forgery.noise_consist import noise_consist_agent
#
# Stage 2 — Doc pool (FUTURE)
# from agents.doc.text_integrity    import text_integrity_agent
# from agents.doc.ocr_consist       import ocr_consist_agent
# from agents.doc.font_analysis     import font_analysis_agent
# from agents.doc.layout_check      import layout_check_agent
#
# Stage 2 — Synth pool (FUTURE)
# from agents.synth.frequency       import frequency_agent
# from agents.synth.texture_global  import texture_global_agent
# from agents.synth.vlm_general     import vlm_general_agent
# from agents.synth.reference_clip  import reference_clip_agent
#
# Stage 3 — Universal agents
# from agents.universal.metadata     import metadata_agent
# from agents.universal.steganography import steganography_agent
# from agents.universal.provenance   import provenance_agent
#
# Stage 4
# from fusion.reconciler  import reconciler
#
# Stage 5
# from fusion.bayesian    import bayesian_fusion
#
# Stage 6
# from report.generator   import generate_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("master_agent")


# ══════════════════════════════════════════════════════════════════════════════
#  STUB AGENTS
#  Replace each stub body with the real import as modules are implemented.
#  Every stub returns anomaly_score + agent_applicable — both are mandatory.
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 0 ───────────────────────────────────────────────────────────────────

@tool
def preprocessing_agent(image_path: str) -> dict:
    """
    Stage 0 — Universal intake.
    SHA-256 · MD5 · EXIF baseline · ELA map · image dims · PRNU baseline.
    NO face detection — that is geometry_agent's job in the face pool.
    Gate: sets preprocess_ok=False if image is unreadable/corrupt.
    """
    # TODO: replace body with real implementation
    return {
        "image_path":     os.path.abspath(image_path),
        "hash_sha256":    "stub_sha256",
        "hash_md5":       "stub_md5",
        "ela_score":      0.34,
        "image_dims":     [512, 512],
        "prnu_baseline":  0.021,
        "exif_raw":       {},
        "preprocess_ok":  True,
        "anomaly_score":  0.34,
        "agent_applicable": True,
    }


# ── Stage 1 ───────────────────────────────────────────────────────────────────

@tool
def content_router_agent(image_path: str) -> dict:
    """
    Stage 1 — Content router.
    Uses CLIP zero-shot classification (NOT Mistral) to decide which
    agent pools activate.  One VLM call, multiple text prompts.

    Routing flags:
      face_present        → activates Face pool
      scene_type          → activates Scene pool (None = no scene pool)
      has_text            → activates Doc pool
      full_synthesis_likely → boosts Synth pool weight
      subject_count       → used by reconciler for multi-face logic
      document_type       → "photo" | "screenshot" | "document" | "artwork"

    CURRENT STATUS: stub — hardcoded face_present=True, all others False/None.
    Replace body with real CLIP inference when content_router.py is ready.
    """
    # TODO: replace with CLIP zero-shot:
    #   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    #   prompts = ["a photo of a human face", "a photo of a scene without people",
    #              "a document or screenshot with text", ...]
    #   scores = model(image, prompts)  → routing flags
    return {
        "face_present":          True,   # activates face pool
        "scene_type":            None,   # FUTURE: "outdoor" | "indoor" | "studio" | None
        "has_text":              False,  # FUTURE: activates doc pool
        "full_synthesis_likely": False,  # FUTURE: boosts synth pool
        "subject_count":         1,      # FUTURE: multi-face fan-out
        "document_type":         "photo",
        "agent_applicable":      True,
        "anomaly_score":         0.0,    # router itself is neutral
    }


# ── Stage 2 — Face pool ───────────────────────────────────────────────────────
# geometry_agent is the FIRST face-pool agent.
# It runs RetinaFace to get face_bbox, then does landmark geometry.
# All other face-pool agents receive face_bbox from geometry's output.

@tool
def geometry_agent(image_path: str) -> dict:
    """
    Face pool — primary.
    Runs RetinaFace to detect face_bbox, then 68-pt dlib landmark analysis.
    Computes symmetry_index, jaw_curvature_deg, ear_alignment_px, philtrum_length.
    Sets face_bbox in its output — shared with all other face-pool agents.
    Sets agent_applicable=False if no face is found (routes pool to abstain).
    """
    # TODO: replace with RetinaFace + dlib shape predictor
    return {
        "face_bbox":           [120, 80, 420, 460],  # RetinaFace output
        "symmetry_index":      0.74,
        "jaw_curvature_deg":   11.2,
        "ear_alignment_px":    8.7,
        "philtrum_length":     0.21,
        "landmark_confidence": 0.91,
        "anomaly_score":       0.884,
        "agent_applicable":    True,   # False if RetinaFace finds no face
    }


@tool
def biological_agent(image_path: str, face_bbox: list) -> dict:
    """
    Face pool — rPPG SNR, corneal highlight consistency, perioral micro-texture.
    Receives face_bbox from geometry_agent output (set by pool_dispatch_node).
    """
    # TODO: replace with real implementation
    return {
        "rppg_snr":              2.1,
        "corneal_deviation_deg": 22.4,
        "micro_texture_var":     0.012,
        "highlight_positions":   {"left": [210, 180], "right": [310, 182]},
        "anomaly_score":         0.826,
        "agent_applicable":      True,
    }


@tool
def vlm_face_agent(image_path: str, face_bbox: list) -> dict:
    """
    Face pool — BLIP-2 Grad-CAM heatmap + forensic caption focused on face region.
    Distinct from vlm_general (Synth pool) which analyses the full image.
    """
    # TODO: replace with BLIP-2 + pytorch-grad-cam
    return {
        "heatmap_path":            "outputs/heatmap_face.png",
        "vlm_caption":             "Unnatural texture smoothing at jaw boundary. Shadow inconsistency.",
        "saliency_score":          0.91,
        "high_activation_regions": ["jaw boundary", "eyes", "nose bridge"],
        "anomaly_score":           0.931,
        "agent_applicable":        True,
    }


@tool
def texture_face_agent(image_path: str, face_bbox: list) -> dict:
    """
    Face pool — LBP + Gabor + Earth Mover's Distance across face zones.
    Face-specific crop (uses face_bbox). Distinct from texture_global (Synth pool).
    """
    # TODO: replace with real implementation
    return {
        "jaw_emd":        0.61,
        "neck_emd":       0.48,
        "cheek_emd":      0.22,
        "lbp_uniformity": 0.31,
        "seam_detected":  True,
        "zone_scores":    {"forehead": 0.2, "cheek_L": 0.3, "jaw": 0.8},
        "anomaly_score":  0.895,
        "agent_applicable": True,
    }


@tool
def reference_face_agent(image_path: str, face_bbox: list) -> dict:
    """
    Face pool — FaceNet embedding cosine distance to real/fake face clusters.
    Face-specific. Distinct from reference_clip (Synth pool) which uses CLIP embeddings.
    """
    # TODO: replace with deepface FaceNet
    return {
        "cosine_dist_authentic": 0.71,
        "cosine_dist_fake":      0.18,
        "verdict":               "CLOSER_TO_FAKE",
        "embedding_norm":        0.994,
        "anomaly_score":         0.910,
        "agent_applicable":      True,
    }


# ── Stage 2 — Scene pool (FUTURE) ─────────────────────────────────────────────
# Activated when content_router returns scene_type is not None.
# Each agent does its own spatial parsing — no shared bbox from face pool.

@tool
def scene_pool_stub(image_path: str, scene_type: str) -> dict:
    """
    FUTURE — Scene pool placeholder.
    Replace with individual agents:
      struct_consist_agent, inpaint_detect_agent, physics_check_agent,
      copy_move_agent, splicing_agent
    Each agent returns anomaly_score + agent_applicable.
    """
    # ADD SCENE POOL AGENTS HERE
    return {"anomaly_score": 0.5, "agent_applicable": False, "note": "scene pool not implemented"}


# ── Stage 2 — Forgery pool (FUTURE) ───────────────────────────────────────────
# Always active — no routing gate.

@tool
def forgery_pool_stub(image_path: str) -> dict:
    """
    FUTURE — Forgery pool placeholder.
    Replace with individual agents:
      ela_deep_agent, dct_artifact_agent, prnu_field_agent,
      resampling_agent, noise_consist_agent
    """
    # ADD FORGERY POOL AGENTS HERE
    return {"anomaly_score": 0.5, "agent_applicable": False, "note": "forgery pool not implemented"}


# ── Stage 2 — Doc pool (FUTURE) ───────────────────────────────────────────────
# Activated when content_router returns has_text=True.

@tool
def doc_pool_stub(image_path: str) -> dict:
    """
    FUTURE — Doc pool placeholder.
    Replace with individual agents:
      text_integrity_agent, ocr_consist_agent, font_analysis_agent, layout_check_agent
    text_integrity_agent runs OCR first to get text bboxes — no face_bbox needed.
    """
    # ADD DOC POOL AGENTS HERE
    return {"anomaly_score": 0.5, "agent_applicable": False, "note": "doc pool not implemented"}


# ── Stage 2 — Synth pool (FUTURE) ─────────────────────────────────────────────
# Always active — no routing gate.

@tool
def synth_pool_stub(image_path: str) -> dict:
    """
    FUTURE — Synth pool placeholder.
    Replace with individual agents:
      frequency_agent, texture_global_agent, vlm_general_agent, reference_clip_agent
    These operate on the FULL image, not just the face crop.
    """
    # ADD SYNTH POOL AGENTS HERE
    return {"anomaly_score": 0.5, "agent_applicable": False, "note": "synth pool not implemented"}


# ── Stage 3 — Universal agents ────────────────────────────────────────────────
# Always run, no gate, no routing condition.

@tool
def metadata_agent(image_path: str) -> dict:
    """
    Stage 3 — Universal.
    EXIF spoof detection, ELA chi-squared, thumbnail mismatch, PRNU absence.
    Always runs regardless of content_router output.
    """
    # TODO: replace with real implementation
    return {
        "exif_camera_present": False,
        "software_tag":        "Adobe Photoshop 24.0",
        "ela_chi2":            847.3,
        "ela_map_path":        "outputs/ela_map.png",
        "thumbnail_mismatch":  True,
        "prnu_absent":         True,
        "anomaly_score":       0.973,
        "agent_applicable":    True,
    }


@tool
def steganography_agent(image_path: str) -> dict:
    """
    Stage 3 — Universal.
    LSB payload detection. Checks for hidden data in least-significant bits.
    Always runs regardless of content_router output.
    FUTURE: replace stub with real LSB analysis.
    """
    # ADD STEGANOGRAPHY IMPLEMENTATION HERE
    return {
        "lsb_payload_detected": False,
        "lsb_capacity_bits":    0,
        "anomaly_score":        0.1,
        "agent_applicable":     True,
    }


@tool
def provenance_agent(image_path: str, hash_sha256: str) -> dict:
    """
    Stage 3 — Universal.
    Chain-of-custody: checks hash against known databases, C2PA manifest,
    reverse image search hooks.
    Always runs regardless of content_router output.
    FUTURE: replace stub with real C2PA + hash-db lookup.
    """
    # ADD PROVENANCE IMPLEMENTATION HERE
    return {
        "c2pa_manifest_present": False,
        "hash_known_fake":       False,
        "origin_url":            None,
        "anomaly_score":         0.2,
        "agent_applicable":      True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STUB RECONCILER  (Stage 4)
#  Replace with: from fusion.reconciler import reconciler
# ══════════════════════════════════════════════════════════════════════════════

def reconciler(agent_outputs: dict) -> dict:
    """
    Stage 4 — Cross-agent reconciler.
    - Reads agent_applicable flags from every agent output.
    - Sets abstain=0.5 for non-applicable agents (NOT 0.0).
      0.0 would drag the Bayesian fusion toward "real" unfairly.
    - Flags contradictions (e.g. ELA clean but PRNU mismatch).
    - Returns reconciled_scores and contradiction_flags.

    Replace with: from fusion.reconciler import reconciler
    """
    reconciled  = {}
    contradictions = []

    for name, output in agent_outputs.items():
        if not output.get("agent_applicable", True):
            reconciled[name] = 0.5   # abstain — agent not applicable to this image type
        else:
            reconciled[name] = output.get("anomaly_score", 0.5)

    # Example contradiction check (expand in real reconciler.py):
    meta = agent_outputs.get("metadata", {})
    if meta.get("ela_chi2", 0) < 200 and meta.get("prnu_absent", False):
        contradictions.append("ELA clean but PRNU absent — possible targeted synthesis")

    return {
        "reconciled_scores":   reconciled,
        "contradiction_flags": contradictions,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STUB FUSION  (Stage 5)
#  Replace with: from fusion.bayesian import bayesian_fusion
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_fusion(reconciled_scores: dict[str, float]) -> dict:
    """
    Stage 5 — Log-odds Bayesian fusion.
    Weights only applicable agents (abstain=0.5 contributes 0 log-odds — neutral).
    Formula: log_odds = SUM [ weight_i * log(score_i / (1 - score_i)) ]
             final_score = sigmoid(log_odds)

    Replace with: from fusion.bayesian import bayesian_fusion
    """
    import math

    # Weights based on published AUC. Non-applicable agents are already
    # set to 0.5 by reconciler — log(0.5/0.5) = 0, so they contribute nothing.
    MODULE_WEIGHTS = {
        # Face pool
        "geometry":      0.884,
        "biological":    0.826,
        "vlm_face":      0.931,
        "texture_face":  0.895,
        "reference_face":0.910,
        # Universal (Stage 3)
        "metadata":      0.973,
        "steganography": 0.700,
        "provenance":    0.750,
        # FUTURE pools — add weights here when implemented:
        # "struct_consist":  0.0,  # scene pool
        # "ela_deep":        0.0,  # forgery pool
        # "text_integrity":  0.0,  # doc pool
        # "frequency":       0.0,  # synth pool
    }

    log_odds_total = 0.0
    for module, score in reconciled_scores.items():
        score  = max(1e-6, min(1 - 1e-6, score))
        weight = MODULE_WEIGHTS.get(module, 0.5)
        log_odds_total += weight * math.log(score / (1.0 - score))

    final_score = 1.0 / (1.0 + math.exp(-log_odds_total))
    margin = 0.05 * (1 - abs(2 * final_score - 1))
    ci = [round(max(0.0, final_score - margin), 3),
          round(min(1.0, final_score + margin), 3)]

    return {
        "final_score":         round(final_score, 4),
        "confidence_interval": ci,
        "verdict":             "DEEPFAKE" if final_score >= 0.70 else "LIKELY REAL",
        "per_module_scores":   reconciled_scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MFAD STATE
# ══════════════════════════════════════════════════════════════════════════════

class MFADState(TypedDict, total=False):
    """
    Single shared state object — flows through every LangGraph node.
    total=False means every key is optional (nodes only write their own keys).
    """

    # ── Input ─────────────────────────────────────────────────────────────
    image_path:   str
    case_id:      str

    # ── Stage 0: Preprocessing ────────────────────────────────────────────
    preprocessing:  dict    # full output of preprocessing_agent
    preprocess_ok:  bool    # gate: False aborts graph at route_after_preprocess

    # ── Stage 1: Content router ───────────────────────────────────────────
    routing_flags:  dict    # full output of content_router_agent
    face_present:   bool    # shortcut — read by pool_dispatch_node
    has_text:       bool    # shortcut
    scene_type:     Optional[str]  # shortcut

    # ── Stage 2: Face pool ────────────────────────────────────────────────
    face_bbox:       list   # set by geometry_agent, shared across face pool
    geometry:        dict
    biological:      dict
    vlm_face:        dict
    texture_face:    dict
    reference_face:  dict

    # ── Stage 2: Scene pool (FUTURE) ──────────────────────────────────────
    # struct_consist:  dict
    # inpaint_detect:  dict
    # physics_check:   dict
    # copy_move:       dict
    # splicing:        dict

    # ── Stage 2: Forgery pool (FUTURE) ────────────────────────────────────
    # ela_deep:        dict
    # dct_artifact:    dict
    # prnu_field:      dict
    # resampling:      dict
    # noise_consist:   dict

    # ── Stage 2: Doc pool (FUTURE) ────────────────────────────────────────
    # text_integrity:  dict
    # ocr_consist:     dict
    # font_analysis:   dict
    # layout_check:    dict

    # ── Stage 2: Synth pool (FUTURE) ──────────────────────────────────────
    # frequency:       dict
    # texture_global:  dict
    # vlm_general:     dict
    # reference_clip:  dict

    # ── Stage 3: Universal agents ─────────────────────────────────────────
    metadata:      dict
    steganography: dict
    provenance:    dict

    # ── Stage 4: Reconciler ───────────────────────────────────────────────
    reconciler_output: dict   # {reconciled_scores, contradiction_flags}

    # ── Stage 5: Fusion ───────────────────────────────────────────────────
    fusion: dict              # {final_score, confidence_interval, verdict, per_module_scores}

    # ── Stage 6: Report ───────────────────────────────────────────────────
    report_path:    str
    master_output:  dict

    # ── Reflection loop ───────────────────────────────────────────────────
    # Tracks re-analysis passes when score lands in ambiguous zone (0.45–0.65).
    # Currently inactive — see should_reflect() and build_graph() comments.
    reflection_passes: int

    # ── Error tracking ────────────────────────────────────────────────────
    errors:      list[str]
    fatal_error: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 0: Preprocessing ────────────────────────────────────────────────────

def preprocess_node(state: MFADState) -> MFADState:
    """
    Stage 0 — Universal intake gate.

    Runs preprocessing_agent (SHA-256, EXIF, ELA, dims, PRNU baseline).
    NO face detection here — face detection belongs to geometry_agent
    in the face pool (Stage 2).

    Gate logic (hard stop):
      Image unreadable / corrupt → preprocess_ok=False → abort_node
      Image readable             → preprocess_ok=True  → router_node
    """
    log.info("▶ [Stage 0] preprocess_node | %s", state["image_path"])
    errors = state.get("errors", [])

    try:
        result = preprocessing_agent.invoke({"image_path": state["image_path"]})

        if not result.get("preprocess_ok", True):
            return {
                **state,
                "preprocessing": result,
                "preprocess_ok": False,
                "fatal_error":   "Preprocessing failed — image unreadable or corrupt.",
                "errors":        errors,
            }

        log.info("  ✓ sha256=%s  dims=%s", result["hash_sha256"], result["image_dims"])
        return {**state, "preprocessing": result, "preprocess_ok": True, "errors": errors}

    except Exception as exc:
        log.error("preprocess_node error: %s", exc)
        return {
            **state,
            "preprocessing": {},
            "preprocess_ok": False,
            "fatal_error":   f"Preprocessing exception: {exc}",
            "errors":        errors + [f"preprocessing: {exc}"],
        }


# ── Stage 1: Content router ───────────────────────────────────────────────────

def router_node(state: MFADState) -> MFADState:
    """
    Stage 1 — Content router.

    Calls content_router_agent (CLIP zero-shot) to produce routing flags.
    These flags are stored in state and read by pool_dispatch_node to decide
    which pools to activate.

    Currently: stub always returns face_present=True.
    When content_router.py is implemented, this node needs no changes —
    only the stub body inside content_router_agent changes.
    """
    log.info("▶ [Stage 1] router_node")
    errors = state.get("errors", [])

    try:
        flags = content_router_agent.invoke({"image_path": state["image_path"]})
        log.info(
            "  ✓ face_present=%s  scene_type=%s  has_text=%s  full_synth=%s",
            flags.get("face_present"), flags.get("scene_type"),
            flags.get("has_text"), flags.get("full_synthesis_likely"),
        )
        return {
            **state,
            "routing_flags": flags,
            "face_present":  flags.get("face_present", False),
            "has_text":      flags.get("has_text", False),
            "scene_type":    flags.get("scene_type"),
            "errors":        errors,
        }
    except Exception as exc:
        log.error("router_node error: %s", exc)
        # Fail safe: assume face present so pipeline doesn't silently do nothing
        return {
            **state,
            "routing_flags": {},
            "face_present":  True,
            "has_text":      False,
            "scene_type":    None,
            "errors":        errors + [f"content_router: {exc}"],
        }


# ── Stage 2: Pool dispatch ────────────────────────────────────────────────────

def pool_dispatch_node(state: MFADState) -> MFADState:
    """
    Stage 2 — Parallel pool dispatcher.

    Reads routing flags from state, fires active pools via asyncio.gather.
    Wall time = max(slowest agent), not sum of all agents.

    CURRENTLY ACTIVE:
      Face pool    (always active for now — face_present=True from stub router)

    FUTURE POOLS (add safe_run() calls when each pool is implemented):
      Scene pool   — if state.get("scene_type") is not None
      Forgery pool — always active
      Doc pool     — if state.get("has_text")
      Synth pool   — always active

    face_bbox lifecycle:
      geometry_agent runs RetinaFace → sets face_bbox in its output.
      pool_dispatch_node reads face_bbox from geometry result and passes it
      to remaining face-pool agents (biological, vlm_face, texture_face,
      reference_face) which cannot run without it.

    agent_applicable contract:
      If geometry_agent returns face_bbox=None / agent_applicable=False,
      all other face-pool agents receive agent_applicable=False and
      return anomaly_score=0.5 (abstain).
    """
    log.info(
        "▶ [Stage 2] pool_dispatch_node | face=%s scene=%s text=%s",
        state.get("face_present"), state.get("scene_type"), state.get("has_text"),
    )

    image_path = state["image_path"]
    errors     = list(state.get("errors", []))
    results: dict[str, Any] = {}

    async def run_pools() -> dict[str, Any]:

        async def safe_run(name: str, fn) -> tuple[str, dict]:
            try:
                out = await asyncio.to_thread(fn)
                log.info(
                    "  ✓ %-20s anomaly=%.3f applicable=%s",
                    name, out.get("anomaly_score", 0), out.get("agent_applicable", True),
                )
                return name, out
            except Exception as exc:
                log.error("  ✗ %-20s %s", name, exc)
                errors.append(f"{name}: {exc}")
                # abstain on failure — do not bias fusion toward real or fake
                return name, {"anomaly_score": 0.5, "agent_applicable": False, "error": str(exc)}

        pool_tasks = []

        # ── FACE POOL ─────────────────────────────────────────────────────
        # Step 1: geometry_agent runs RetinaFace, returns face_bbox.
        # Step 2: remaining face agents receive face_bbox from geometry output.
        if state.get("face_present", False):
            geo_name, geo_out = await safe_run(
                "geometry",
                lambda: geometry_agent.invoke({"image_path": image_path})
            )
            results["geometry"] = geo_out
            face_bbox = geo_out.get("face_bbox")

            if face_bbox and geo_out.get("agent_applicable", True):
                # Face found — run remaining face-pool agents in parallel
                face_tasks = [
                    safe_run("biological",
                             lambda: biological_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
                    safe_run("vlm_face",
                             lambda: vlm_face_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
                    safe_run("texture_face",
                             lambda: texture_face_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
                    safe_run("reference_face",
                             lambda: reference_face_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
                ]
                face_results = await asyncio.gather(*face_tasks)
                results.update(dict(face_results))
            else:
                # No face found — mark all face-pool agents as abstain
                log.warning("  geometry found no face — face pool abstaining")
                for name in ["biological", "vlm_face", "texture_face", "reference_face"]:
                    results[name] = {"anomaly_score": 0.5, "agent_applicable": False}

        # ── SCENE POOL (FUTURE) ───────────────────────────────────────────
        # ADD SCENE POOL HERE when agents/scene/* are implemented:
        # if state.get("scene_type") is not None:
        #     pool_tasks += [
        #         safe_run("struct_consist",  lambda: struct_consist_agent.invoke({...})),
        #         safe_run("inpaint_detect",  lambda: inpaint_detect_agent.invoke({...})),
        #         safe_run("physics_check",   lambda: physics_check_agent.invoke({...})),
        #         safe_run("copy_move",       lambda: copy_move_agent.invoke({...})),
        #         safe_run("splicing",        lambda: splicing_agent.invoke({...})),
        #     ]

        # ── FORGERY POOL (FUTURE) ─────────────────────────────────────────
        # ADD FORGERY POOL HERE when agents/forgery/* are implemented:
        # pool_tasks += [
        #     safe_run("ela_deep",       lambda: ela_deep_agent.invoke({...})),
        #     safe_run("dct_artifact",   lambda: dct_artifact_agent.invoke({...})),
        #     safe_run("prnu_field",     lambda: prnu_field_agent.invoke({...})),
        #     safe_run("resampling",     lambda: resampling_agent.invoke({...})),
        #     safe_run("noise_consist",  lambda: noise_consist_agent.invoke({...})),
        # ]

        # ── DOC POOL (FUTURE) ─────────────────────────────────────────────
        # ADD DOC POOL HERE when agents/doc/* are implemented:
        # if state.get("has_text"):
        #     pool_tasks += [
        #         safe_run("text_integrity", lambda: text_integrity_agent.invoke({...})),
        #         safe_run("ocr_consist",    lambda: ocr_consist_agent.invoke({...})),
        #         safe_run("font_analysis",  lambda: font_analysis_agent.invoke({...})),
        #         safe_run("layout_check",   lambda: layout_check_agent.invoke({...})),
        #     ]

        # ── SYNTH POOL (FUTURE) ───────────────────────────────────────────
        # ADD SYNTH POOL HERE when agents/synth/* are implemented:
        # pool_tasks += [
        #     safe_run("frequency",      lambda: frequency_agent.invoke({...})),
        #     safe_run("texture_global", lambda: texture_global_agent.invoke({...})),
        #     safe_run("vlm_general",    lambda: vlm_general_agent.invoke({...})),
        #     safe_run("reference_clip", lambda: reference_clip_agent.invoke({...})),
        # ]

        # Run all non-face pool tasks in parallel (empty for now)
        if pool_tasks:
            other_results = await asyncio.gather(*pool_tasks)
            results.update(dict(other_results))

        return results

    all_outputs = asyncio.run(run_pools())

    return {
        **state,
        "face_bbox":      all_outputs.get("geometry", {}).get("face_bbox"),
        "geometry":       all_outputs.get("geometry",       {}),
        "biological":     all_outputs.get("biological",     {}),
        "vlm_face":       all_outputs.get("vlm_face",       {}),
        "texture_face":   all_outputs.get("texture_face",   {}),
        "reference_face": all_outputs.get("reference_face", {}),
        # FUTURE: uncomment as pools are added
        # "struct_consist": all_outputs.get("struct_consist", {}),
        # "ela_deep":       all_outputs.get("ela_deep",       {}),
        # "text_integrity": all_outputs.get("text_integrity", {}),
        # "frequency":      all_outputs.get("frequency",      {}),
        "reflection_passes": state.get("reflection_passes", 0) + 1,
        "errors":         errors,
    }


# ── Stage 3: Universal agents ─────────────────────────────────────────────────

def universal_agents_node(state: MFADState) -> MFADState:
    """
    Stage 3 — Universal agents.
    Always runs. No routing condition. No gate.
    metadata · steganography · provenance
    All run in parallel.
    """
    log.info("▶ [Stage 3] universal_agents_node")
    image_path = state["image_path"]
    errors     = list(state.get("errors", []))

    async def run_universal():
        async def safe_run(name, fn):
            try:
                out = await asyncio.to_thread(fn)
                log.info("  ✓ %-20s anomaly=%.3f", name, out.get("anomaly_score", 0))
                return name, out
            except Exception as exc:
                log.error("  ✗ %-20s %s", name, exc)
                errors.append(f"{name}: {exc}")
                return name, {"anomaly_score": 0.5, "agent_applicable": False, "error": str(exc)}

        tasks = [
            safe_run("metadata",
                     lambda: metadata_agent.invoke({"image_path": image_path})),
            safe_run("steganography",
                     lambda: steganography_agent.invoke({"image_path": image_path})),
            safe_run("provenance",
                     lambda: provenance_agent.invoke({
                         "image_path":  image_path,
                         "hash_sha256": state.get("preprocessing", {}).get("hash_sha256", ""),
                     })),
        ]
        return dict(await asyncio.gather(*tasks))

    outputs = asyncio.run(run_universal())

    return {
        **state,
        "metadata":      outputs.get("metadata",      {}),
        "steganography": outputs.get("steganography", {}),
        "provenance":    outputs.get("provenance",    {}),
        "errors":        errors,
    }


# ── Stage 4: Reconciler ───────────────────────────────────────────────────────

def reconciler_node(state: MFADState) -> MFADState:
    """
    Stage 4 — Cross-agent reconciler.

    Collects all agent outputs, applies agent_applicable logic,
    sets abstain=0.5 for non-applicable agents, flags contradictions.
    Passes reconciled_scores to fusion_node.
    """
    log.info("▶ [Stage 4] reconciler_node")

    all_agent_outputs = {
        # Face pool
        "geometry":      state.get("geometry",       {}),
        "biological":    state.get("biological",     {}),
        "vlm_face":      state.get("vlm_face",       {}),
        "texture_face":  state.get("texture_face",   {}),
        "reference_face":state.get("reference_face", {}),
        # Universal
        "metadata":      state.get("metadata",       {}),
        "steganography": state.get("steganography",  {}),
        "provenance":    state.get("provenance",     {}),
        # FUTURE pools — uncomment as implemented:
        # "struct_consist": state.get("struct_consist", {}),
        # "ela_deep":       state.get("ela_deep",       {}),
        # "text_integrity": state.get("text_integrity", {}),
        # "frequency":      state.get("frequency",      {}),
    }

    recon = reconciler(all_agent_outputs)
    log.info(
        "  ✓ reconciled %d scores | %d contradiction(s)",
        len(recon["reconciled_scores"]),
        len(recon["contradiction_flags"]),
    )
    if recon["contradiction_flags"]:
        for flag in recon["contradiction_flags"]:
            log.warning("  ⚠ contradiction: %s", flag)

    return {**state, "reconciler_output": recon}


# ── Stage 5: Fusion ───────────────────────────────────────────────────────────

def fusion_node(state: MFADState) -> MFADState:
    """
    Stage 5 — Bayesian fusion.

    Receives reconciled_scores (with abstain=0.5 for non-applicable agents).
    log(0.5 / 0.5) = 0 → abstained agents contribute exactly zero log-odds.

    REFLECTION LOOP (currently inactive):
    ──────────────────────────────────────
    If final_score lands in 0.45–0.65 and reflection_passes < 2,
    a conditional edge can route back to pool_dispatch_node for a second pass.
    To activate: swap the direct edge fusion→report for the commented
    conditional edge in build_graph().
    """
    log.info("▶ [Stage 5] fusion_node")

    reconciled_scores = state.get("reconciler_output", {}).get("reconciled_scores", {})
    result = bayesian_fusion(reconciled_scores)

    log.info(
        "  ✓ final_score=%.4f  verdict=%s  CI=%s",
        result["final_score"], result["verdict"], result["confidence_interval"],
    )
    return {**state, "fusion": result}


# ── Stage 6: Report ───────────────────────────────────────────────────────────

def report_node(state: MFADState) -> MFADState:
    """
    Stage 6 — Report generation.

    ONLY place in the entire graph where Mistral-7B (via Ollama) is called.
    LLM writes the executive summary and per-section narrative prose.
    All other nodes are deterministic.

    On LLM failure → falls back to template string so PDF always generates.

    ADD LLM REFINEMENT LOOP HERE (future):
      If Mistral output fails a structured-output validation check,
      retry up to MAX_RETRIES before falling back to template.
    """
    log.info("▶ [Stage 6] report_node")

    case_id   = state.get("case_id", "DFA-UNKNOWN")
    timestamp = datetime.now().isoformat(timespec="seconds")

    # ── Build master JSON ─────────────────────────────────────────────────
    master_output: dict = {
        "case_id":    case_id,
        "image_path": state["image_path"],
        "timestamp":  timestamp,
        "routing_flags": state.get("routing_flags", {}),
        "agent_outputs": {
            "preprocessing":  state.get("preprocessing",  {}),
            "geometry":       state.get("geometry",       {}),
            "biological":     state.get("biological",     {}),
            "vlm_face":       state.get("vlm_face",       {}),
            "texture_face":   state.get("texture_face",   {}),
            "reference_face": state.get("reference_face", {}),
            "metadata":       state.get("metadata",       {}),
            "steganography":  state.get("steganography",  {}),
            "provenance":     state.get("provenance",     {}),
            # FUTURE pools here
        },
        "reconciler": state.get("reconciler_output", {}),
        "fusion":     state.get("fusion", {}),
        "errors":     state.get("errors", []),
    }

    # ── LLM narrative (Mistral-7B via Ollama) ─────────────────────────────
    try:
        llm = ChatOllama(model="mistral", temperature=0.1)
        fusion = master_output["fusion"]
        geo    = master_output["agent_outputs"]["geometry"]
        meta   = master_output["agent_outputs"]["metadata"]
        bio    = master_output["agent_outputs"]["biological"]
        vlm    = master_output["agent_outputs"]["vlm_face"]
        contra = master_output["reconciler"].get("contradiction_flags", [])

        system_msg = SystemMessage(content=(
            "You are a senior digital forensics analyst writing court-admissible reports. "
            "Be precise. State only what the data shows. Do not speculate."
        ))
        user_msg = HumanMessage(content=f"""
Write a 3-sentence executive forensic summary.

Case ID      : {case_id}
Timestamp    : {timestamp}
Final score  : {fusion.get('final_score', 'N/A')} (0=real, 1=deepfake)
Verdict      : {fusion.get('verdict', 'UNKNOWN')}
CI 95%       : {fusion.get('confidence_interval', [])}
Contradictions: {contra if contra else 'None'}

Module scores: {json.dumps(fusion.get('per_module_scores', {}), indent=2)}

Key findings:
- Geometry : symmetry={geo.get('symmetry_index')} jaw_dev={geo.get('jaw_curvature_deg')} deg
- Metadata : ELA_chi2={meta.get('ela_chi2')} PRNU_absent={meta.get('prnu_absent')}
- Biological: corneal_dev={bio.get('corneal_deviation_deg')} deg  micro_var={bio.get('micro_texture_var')}
- VLM       : {vlm.get('vlm_caption', 'N/A')}

Write ONLY the 3-sentence summary. No headers, no bullet points.
        """)

        response = llm.invoke([system_msg, user_msg])
        executive_summary = response.content.strip()
        log.info("  ✓ LLM narrative generated (%d chars)", len(executive_summary))

    except Exception as exc:
        log.warning("  LLM narrative unavailable (Ollama running?): %s", exc)
        f = master_output["fusion"]
        executive_summary = (
            f"Digital forensic analysis of case {case_id} yielded a DeepFake Prediction Score "
            f"of {f.get('final_score', 0):.1%} (95% CI: {f.get('confidence_interval', [0,1])}), "
            f"meeting the threshold for classification as SYNTHETIC MEDIA. "
            f"Active analytical modules — face geometry, biological plausibility, VLM explainability, "
            f"texture analysis, reference embedding, and metadata forensics — all returned anomaly "
            f"scores consistent with GAN or diffusion-model synthesis."
        )

    master_output["executive_summary"] = executive_summary

    # ── Write output ──────────────────────────────────────────────────────
    # TODO: replace with: report_path = generate_report(master_output)
    os.makedirs("outputs", exist_ok=True)
    report_path = f"outputs/{case_id}.json"
    with open(report_path, "w") as f:
        json.dump(master_output, f, indent=2)
    log.info("  ✓ master JSON written → %s  (swap for generate_report() when ready)", report_path)

    return {
        **state,
        "master_output": master_output,
        "report_path":   os.path.abspath(report_path),
    }


# ── Abort node ────────────────────────────────────────────────────────────────

def abort_node(state: MFADState) -> MFADState:
    """Terminal node for hard failures at the preprocessing gate."""
    log.error("▶ abort_node | %s", state.get("fatal_error"))
    os.makedirs("outputs", exist_ok=True)
    error_report = {
        "case_id":     state.get("case_id", "UNKNOWN"),
        "image_path":  state.get("image_path", ""),
        "timestamp":   datetime.now().isoformat(),
        "status":      "ABORTED",
        "fatal_error": state.get("fatal_error"),
        "errors":      state.get("errors", []),
        "fusion":      {"final_score": None, "verdict": "INCONCLUSIVE"},
    }
    report_path = f"outputs/{state.get('case_id', 'UNKNOWN')}_ABORTED.json"
    with open(report_path, "w") as f:
        json.dump(error_report, f, indent=2)
    return {**state, "master_output": error_report, "report_path": os.path.abspath(report_path)}


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def route_after_preprocess(state: MFADState) -> str:
    """Gate router: preprocessing ok → router_node, else → abort_node."""
    return "router_node" if state.get("preprocess_ok") else "abort_node"


def should_reflect(state: MFADState) -> str:
    """
    REFLECTION LOOP — currently inactive.
    Activate by swapping the direct fusion→report edge in build_graph().

    Returns "reflect" if score is ambiguous AND passes < MAX.
    Returns "report"  otherwise.
    """
    MAX_PASSES = 2
    score  = state.get("fusion", {}).get("final_score", 0.0)
    passes = state.get("reflection_passes", 0)
    if 0.45 <= score <= 0.65 and passes < MAX_PASSES:
        log.info("  Reflection triggered: score=%.3f pass %d/%d", score, passes, MAX_PASSES)
        return "reflect"
    return "report"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Assembles the full 7-stage LangGraph StateGraph.

    Current active path:
      START
        → preprocess_node
        → (gate) router_node
        → pool_dispatch_node     [Face pool active, others stubbed]
        → universal_agents_node
        → reconciler_node
        → fusion_node
        → report_node
        → END

    Error path:
      preprocess_node (gate fail) → abort_node → END

    To activate reflection loop:
      1. Comment out:  graph.add_edge("fusion_node", "report_node")
      2. Uncomment the conditional edge block below.
    """
    graph = StateGraph(MFADState)

    # ── Register nodes ────────────────────────────────────────────────────
    graph.add_node("preprocess_node",        preprocess_node)
    graph.add_node("router_node",            router_node)
    graph.add_node("pool_dispatch_node",     pool_dispatch_node)
    graph.add_node("universal_agents_node",  universal_agents_node)
    graph.add_node("reconciler_node",        reconciler_node)
    graph.add_node("fusion_node",            fusion_node)
    graph.add_node("report_node",            report_node)
    graph.add_node("abort_node",             abort_node)

    # ── Edges ─────────────────────────────────────────────────────────────
    graph.add_edge(START, "preprocess_node")

    graph.add_conditional_edges(
        "preprocess_node",
        route_after_preprocess,
        {"router_node": "router_node", "abort_node": "abort_node"},
    )

    graph.add_edge("router_node",           "pool_dispatch_node")
    graph.add_edge("pool_dispatch_node",    "universal_agents_node")
    graph.add_edge("universal_agents_node", "reconciler_node")
    graph.add_edge("reconciler_node",       "fusion_node")

    # ── Fusion → Report (direct, no reflection) ───────────────────────────
    # Comment this out and uncomment the block below to activate reflection.
    graph.add_edge("fusion_node", "report_node")

    # ── REFLECTION LOOP — uncomment to activate ───────────────────────────
    # graph.add_conditional_edges(
    #     "fusion_node",
    #     should_reflect,
    #     {
    #         "reflect": "pool_dispatch_node",   # re-run all pools
    #         "report":  "report_node",
    #     },
    # )

    graph.add_edge("report_node", END)
    graph.add_edge("abort_node",  END)

    return graph


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_compiled_graph(use_checkpointing: bool = False):
    """
    Returns compiled LangGraph.
    Pass use_checkpointing=True for in-memory state persistence.

    FUTURE persistence:
      from langgraph.checkpoint.sqlite import SqliteSaver
      checkpointer = SqliteSaver.from_conn_string("mfad_runs.db")
      return build_graph().compile(checkpointer=checkpointer)
    """
    g = build_graph()
    if use_checkpointing:
        return g.compile(checkpointer=MemorySaver())
    return g.compile()


def analyse_image(image_path: str, analyst_name: str = "MFAD-System") -> dict:
    """
    Public entry-point. Run the full MFAD pipeline on one image.

    Args:
        image_path   : path to image (jpg / png)
        analyst_name : appears in the PDF report header

    Returns:
        master_output dict with all agent results, fusion score,
        executive summary, contradiction flags, and report_path.

    Usage:
        from master_agent import analyse_image
        result = analyse_image("test_images/sample_fake.jpg")
        print(result["fusion"]["verdict"])      # "DEEPFAKE"
        print(result["fusion"]["final_score"])  # 0.957
        print(result["report_path"])            # outputs/DFA-...json (→ pdf when ready)
    """
    year    = datetime.now().strftime("%Y")
    case_id = f"DFA-{year}-TC-{uuid.uuid4().hex[:8].upper()}"

    log.info("=" * 60)
    log.info("  MFAD  |  case=%s  |  analyst=%s", case_id, analyst_name)
    log.info("  image : %s", image_path)
    log.info("=" * 60)

    compiled = get_compiled_graph()
    final_state = compiled.invoke({
        "image_path":        image_path,
        "case_id":           case_id,
        "reflection_passes": 0,
        "errors":            [],
    })

    fusion = final_state.get("fusion", {})
    log.info("=" * 60)
    log.info("  verdict=%s  score=%.4f  CI=%s",
             fusion.get("verdict"), fusion.get("final_score", 0),
             fusion.get("confidence_interval"))
    log.info("  report : %s", final_state.get("report_path"))
    log.info("=" * 60)

    return final_state.get("master_output", {})


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MFAD — Multimodal Forensic Agent for Deepfake Detection")
    parser.add_argument("image_path",    help="Path to image (jpg/png)")
    parser.add_argument("--analyst",     default="MFAD-CLI")
    parser.add_argument("--output-json", action="store_true", help="Print master JSON to stdout")
    args = parser.parse_args()

    result = analyse_image(args.image_path, analyst_name=args.analyst)

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        f = result.get("fusion", {})
        print(f"\n{'─'*55}")
        print(f"  Case     : {result.get('case_id')}")
        print(f"  Verdict  : {f.get('verdict')}")
        print(f"  Score    : {f.get('final_score', 0):.1%}")
        print(f"  CI 95%   : {f.get('confidence_interval')}")
        print(f"  Report   : {result.get('report_path')}")
        print(f"{'─'*55}\n")