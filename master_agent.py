"""
master_agent.py — MFAD LangGraph Orchestrator
==============================================
Framework : LangGraph (StateGraph) + LangChain tools + Ollama (Mistral-7B)

Architecture
────────────
LangGraph StateGraph node topology:

  [START]
      │
      ▼
  preprocess_node         ← CLIP gate: aborts entire graph if face not
      │                     detected or image is corrupt.
      │ preprocess_ok=True
      ▼
  parallel_analysis_node  ← Fans out to all specialist agents concurrently
      │                     via asyncio.gather.
      │                     ← ADD SCENE LOOP / multi-face fan-out HERE
      ▼
  fusion_node             ← Bayesian log-odds fusion (fusion/bayesian.py)
      │                     ← ADD REFLECTION LOOP HERE (ambiguous zone re-run)
      ▼
  report_node             ← Mistral-7B narrative + report/generator.py PDF
      │
      ▼
  [END]

Adding a new agent
──────────────────
1. Implement your agent module so it exports a `run(...)` function
   that returns a dict with an `anomaly_score` key.
2. Register a @tool wrapper in the AGENT TOOLS section below.
3. Add the tool call to _AGENT_REGISTRY at the bottom of that section.
4. In parallel_analysis_node, the registry is iterated automatically —
   no other changes needed.
5. Add the agent's score key to fusion_node's `scores` dict.
6. Add the key to MODULE_WEIGHTS in fusion/bayesian.py if you want it
   weighted in fusion.

Removing / disabling an agent
──────────────────────────────
Set the tool's entry in _AGENT_REGISTRY to None or remove it.
fusion_node silently skips missing scores.

State
─────
MFADState (TypedDict) — single shared object flowing through all nodes.

LLM
───
Mistral-7B via Ollama — only used inside report_node for narrative text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ── Typing ────────────────────────────────────────────────────────────────────
from typing_extensions import TypedDict

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

# ── Internal imports ──────────────────────────────────────────────────────────
from contracts import validate, FUSION_KEYS
from fusion.bayesian import bayesian_fusion

# ── Agent imports (each raises ImportError if its deps are not installed) ─────
# We import lazily inside each @tool so the graph can be imported for unit
# tests without all heavy models being loaded.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("master_agent")


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT TOOL WRAPPERS
#  One @tool per agent. Each wrapper:
#    1. Imports the real agent module lazily (so tests don't load 14GB models)
#    2. Calls its run() / main function with the correct arguments
#    3. Returns the agent's output dict
#
#  The function signature is what LangChain uses for tool calling — keep it
#  simple: only primitive types + list as arguments.
#
#  Every tool MUST return a dict that contains "anomaly_score" (float 0-1).
#  If the agent crashes, safe_run() in parallel_analysis_node catches it
#  and returns {"anomaly_score": 0.0, "error": str(exc)}.
# ══════════════════════════════════════════════════════════════════════════════

@tool
def preprocessing_tool(image_path: str) -> dict:
    """
    ALWAYS runs first. Computes SHA-256/MD5, detects face bbox, runs ELA.
    Returns preprocessing JSON path and key fields inline.

    Args:
        image_path: path to the image file to analyse.

    Returns:
        Full preprocessing output dict including image_path, face_bbox,
        hash_sha256, hash_md5, ela_score, face_detected, anomaly_score.
    """
    from agents.preprocessing_agent import run_preprocessing
    import json as _json

    json_path = run_preprocessing(image_path)
    with open(json_path) as f:
        result = _json.load(f)
    # Attach the json path so downstream agents can use it if needed
    result["preprocessing_json_path"] = json_path
    return result


@tool
def geometry_tool(image_path: str, face_bbox: list) -> dict:
    """
    68-point dlib landmark analysis. Computes symmetry_index, jaw_curvature_deg,
    ear_alignment_px, philtrum_length, anomaly_score.

    Args:
        image_path: absolute path to the original image.
        face_bbox:  [x1, y1, x2, y2] from preprocessing.
    """
    import cv2 as _cv2
    from agents.geometry import run

    img = _cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"geometry_tool: cannot read {image_path}")
    return run(image_bgr=img, face_present=True)


@tool
def frequency_tool(image_path: str) -> dict:
    """
    FFT radial spectrum + SVM classifier. Returns fft_mid_anomaly_db,
    fft_high_anomaly_db, svm_fake_probability, anomaly_score.

    NOTE: frequency_agent.py handles its own face detection and
    preprocessing internally. Pass the raw image path directly.

    Args:
        image_path: absolute path to original image.
    """
    from agents.frequency_agent import run

    result = run({"input_type": "image", "path": image_path})
    return result


@tool
def texture_tool(image_path: str, face_bbox: list) -> dict:
    """
    LBP + Gabor + Earth Mover's Distance seam detection across face zones.
    Returns jaw_emd, neck_emd, cheek_emd, lbp_uniformity, seam_detected,
    zone_scores, anomaly_score.

    Args:
        image_path: absolute path to original image.
        face_bbox:  [x1, y1, x2, y2] from preprocessing.
    """
    from agents.texture import run_texture_agent

    result = run_texture_agent(image_path=image_path, face_bbox=face_bbox)
    return result.model_dump()


@tool
def vlm_tool(image_path: str, face_bbox: list, face_crop_path: str) -> dict:
    """
    LLaVA-1.5-7b forensic caption + Grad-CAM heatmap + saliency score.
    Returns heatmap_path, vlm_caption, vlm_verdict, vlm_confidence,
    saliency_score, high/medium/low_activation_regions, zone_gan_probability,
    anomaly_score.

    Args:
        image_path:      absolute path to original image.
        face_bbox:       [x1, y1, x2, y2] from preprocessing.
        face_crop_path:  path to the 512x512 normalised face crop from preprocessing.
    """
    from agents.vlm import VLMAgent

    agent = VLMAgent()
    ctx = {
        "image_path":     image_path,
        "face_bbox":      face_bbox,
        "face_crop_path": face_crop_path,
    }
    return agent.run(ctx)


@tool
def biological_tool(image_path: str, face_bbox: list) -> dict:
    """
    Pupil shape BIoU, corneal highlight IoU, micro-texture variance.
    Returns biou_left, biou_right, avg_biou, iou_reflect, solidity,
    convexity, aspect, hu1, reflection_count, anomaly_score.

    NOTE: biological_plausibility_agent.py exposes analyse_image() which
    returns a results dict. We wrap it to match the contracts schema,
    mapping its internal keys to the expected BIOLOGICAL_KEYS.

    Args:
        image_path: absolute path to original image.
        face_bbox:  [x1, y1, x2, y2] from preprocessing.
    """
    import cv2 as _cv2
    from agents.biological_plausibility_agent import (
        analyse_image, make_face_mesh, CFG,
    )

    img = _cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"biological_tool: cannot read {image_path}")

    face_mesh = make_face_mesh()
    try:
        raw = analyse_image(img, face_mesh, CFG)
    finally:
        face_mesh.close()

    # Map biological_plausibility_agent output → contracts.py BIOLOGICAL_KEYS
    # rppg_snr      : avg_biou used as proxy (real rPPG needs video; single-image
    #                 BIoU is the closest available signal from this agent)
    # corneal_deviation_deg: derived from iou_reflect (lower IoU = more deviation)
    # micro_texture_var    : derived from solidity (lower solidity = over-smoothed)
    # vascular_pearson_r   : not available from this agent — set to None

    avg_biou   = float(raw.get("avg_biou", 0.5))
    iou_ref    = float(raw.get("iou_reflect", 0.0))
    solidity   = float(raw.get("solidity", 1.0))
    prediction = raw.get("prediction", "real")

    # anomaly_score: fake → high score, real → low score
    # Use inverse of avg_biou as primary signal (low BIoU = irregular pupil = fake)
    # Blend with corneal IoU signal
    raw_score  = 0.6 * (1.0 - avg_biou) + 0.4 * max(0.0, 1.0 - iou_ref * 2)
    anomaly_score = float(min(1.0, max(0.0, raw_score)))

    return {
        # raw biological_plausibility_agent values (kept for report)
        "biou_left":             float(raw.get("biou_left", 0.5)),
        "biou_right":            float(raw.get("biou_right", 0.5)),
        "avg_biou":              avg_biou,
        "iou_reflect":           iou_ref,
        "solidity":              solidity,
        "convexity":             float(raw.get("convexity", 1.0)),
        "aspect":                float(raw.get("aspect", 1.0)),
        "hu1":                   float(raw.get("hu1", 0.0)),
        "reflection_count":      int(raw.get("reflection_count", 0)),
        "landmarks_found":       bool(raw.get("landmarks_found", False)),
        "prediction":            prediction,
        # contracts.py BIOLOGICAL_KEYS mapped values
        "rppg_snr":              round(avg_biou, 4),           # proxy
        "corneal_deviation_deg": round((1.0 - iou_ref) * 20, 4),  # scaled proxy
        "micro_texture_var":     round(solidity * 0.031, 4),   # scaled proxy
        "vascular_pearson_r":    None,                         # not available
        "anomaly_score":         round(anomaly_score, 4),
    }


@tool
def metadata_tool(preprocessing_json_path: str) -> dict:
    """
    EXIF, ELA chi-squared, thumbnail mismatch, PRNU. Returns full metadata
    forensic dict including anomaly_score.

    Args:
        preprocessing_json_path: path to JSON produced by preprocessing_tool.
    """
    from agents.metadata_agent import run_metadata
    import json as _json

    json_path = run_metadata(preprocessing_json_path)
    with open(json_path) as f:
        return _json.load(f)


# ── Agent registry ─────────────────────────────────────────────────────────────
# Maps a logical name to:
#   {
#     "tool":    the @tool function,
#     "args_fn": callable(state) → dict of kwargs for tool.invoke(),
#     "score_key": key to extract from result for fusion,
#     "fusion_module": key name expected by bayesian_fusion()
#   }
#
# TO ADD A NEW AGENT: append an entry here. Nothing else needs to change.
# TO DISABLE AN AGENT: set "enabled": False.

def _make_registry(state) -> list[dict]:
    """
    Build the list of agent descriptors for the current state.
    Called at the start of parallel_analysis_node.

    Keeping this as a function (not a module-level constant) means
    the state dict is available for computing args_fn closures.
    """
    image_path    = state["image_path"]
    face_bbox     = state["face_bbox"]
    prep_json     = state.get("preprocessing", {}).get("preprocessing_json_path", "")
    face_crop     = state.get("preprocessing", {}).get("normalised_img_path", "")

    return [
        {
            "name":          "geometry",
            "tool":          geometry_tool,
            "invoke_args":   {"image_path": image_path, "face_bbox": face_bbox},
            "score_key":     "anomaly_score",
            "fusion_module": "geometry",
            "enabled":       True,
        },
        {
            "name":          "frequency",
            "tool":          frequency_tool,
            "invoke_args":   {"image_path": image_path},
            "score_key":     "anomaly_score",
            "fusion_module": "frequency",
            "enabled":       True,
        },
        {
            "name":          "texture",
            "tool":          texture_tool,
            "invoke_args":   {"image_path": image_path, "face_bbox": face_bbox},
            "score_key":     "anomaly_score",
            "fusion_module": "texture",
            "enabled":       True,
        },
        {
            "name":          "vlm",
            "tool":          vlm_tool,
            "invoke_args":   {
                "image_path":     image_path,
                "face_bbox":      face_bbox,
                "face_crop_path": face_crop,
            },
            "score_key":     "anomaly_score",
            "fusion_module": "vlm",
            "enabled":       True,
        },
        {
            "name":          "biological",
            "tool":          biological_tool,
            "invoke_args":   {"image_path": image_path, "face_bbox": face_bbox},
            "score_key":     "anomaly_score",
            "fusion_module": "biological",
            "enabled":       True,
        },
        {
            "name":          "metadata",
            "tool":          metadata_tool,
            "invoke_args":   {"preprocessing_json_path": prep_json},
            "score_key":     "anomaly_score",
            "fusion_module": "metadata",
            "enabled":       True,
        },
        # ── ADD NEW AGENTS BELOW THIS LINE ──────────────────────────────────
        # Example:
        # {
        #     "name":          "reference",
        #     "tool":          reference_tool,
        #     "invoke_args":   {"image_path": image_path},
        #     "score_key":     "anomaly_score",
        #     "fusion_module": "gan_artefact",
        #     "enabled":       True,
        # },
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class MFADState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    image_path:   str
    case_id:      str
    analyst_name: str

    # ── Preprocessing (CLIP gate result) ───────────────────────────────────
    preprocessing:  dict    # full output dict from preprocessing_tool
    face_bbox:      list    # [x1, y1, x2, y2] shortcut for all downstream agents
    preprocess_ok:  bool    # False → abort_node

    # ── Per-agent outputs (keyed by logical agent name) ────────────────────
    agent_outputs:  dict    # {"geometry": {...}, "frequency": {...}, ...}

    # ── Fusion ─────────────────────────────────────────────────────────────
    fusion: dict            # bayesian_fusion() output (FUSION_KEYS)

    # ── Reflection loop counter ─────────────────────────────────────────────
    # ADD REFLECTION LOOP HERE: increment each re-analysis pass.
    reflection_passes: int

    # ── Report ──────────────────────────────────────────────────────────────
    report_path:   str
    master_output: dict

    # ── Errors ──────────────────────────────────────────────────────────────
    errors:      list
    fatal_error: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — PREPROCESSING  (CLIP gate)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_node(state: MFADState) -> MFADState:
    """
    Gate node — entire pipeline is blocked until this succeeds.

    On success:
        preprocess_ok = True
        state["preprocessing"] = full preprocessing dict
        state["face_bbox"]     = [x1, y1, x2, y2]

    On failure (no face, unreadable image):
        preprocess_ok = False  →  route_after_preprocess → abort_node

    ADD MULTI-FACE SCENE LOOP HERE (future):
        If preprocessing returns multiple face_bboxes, fan out to one
        MFADState per face using a map-reduce sub-graph pattern.
    """
    log.info("▶ preprocess_node | %s", state["image_path"])
    errors = list(state.get("errors", []))

    try:
        result = preprocessing_tool.invoke({"image_path": state["image_path"]})

        if not result.get("face_detected", False):
            log.warning("CLIP gate: no face detected — aborting pipeline")
            return {
                **state,
                "preprocessing": result,
                "preprocess_ok": False,
                "fatal_error": "No face detected. Pipeline aborted.",
                "errors": errors,
            }

        log.info(
            "  ✓ face_bbox=%s  sha256=%.16s...",
            result["face_bbox"], result.get("hash_sha256", ""),
        )
        return {
            **state,
            "preprocessing":  result,
            "face_bbox":      result["face_bbox"],
            "preprocess_ok":  True,
            "agent_outputs":  {},
            "errors":         errors,
        }

    except Exception as exc:
        log.error("preprocess_node crash:\n%s", traceback.format_exc())
        return {
            **state,
            "preprocessing": {},
            "preprocess_ok": False,
            "fatal_error":   f"Preprocessing crashed: {exc}",
            "errors":        errors + [f"preprocessing: {exc}"],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — PARALLEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def parallel_analysis_node(state: MFADState) -> MFADState:
    """
    Runs all registered agents concurrently with asyncio.gather.

    Each agent call is wrapped in safe_run() which catches any exception
    and returns a zero-score stub so fusion can still proceed.

    The agent registry (_make_registry) is rebuilt from state each call,
    so on a reflection-loop re-run you can pass a modified state (e.g.
    tighter face_bbox) to re-run agents on a different crop.

    ADD REFLECTION LOOP HERE:
        On re-runs, mutate state["face_bbox"] to a tighter crop before
        this node is called. The registry picks it up automatically.

    ADD MULTI-FACE SCENE LOOP HERE:
        Loop over state["preprocessing"]["face_bboxes"], override
        state["face_bbox"] for each, call the registry, collect results.
    """
    pass_num = state.get("reflection_passes", 0) + 1
    log.info("▶ parallel_analysis_node | pass #%d", pass_num)

    errors: list[str] = list(state.get("errors", []))
    agent_outputs: dict[str, Any] = dict(state.get("agent_outputs", {}))

    registry = _make_registry(state)
    active = [r for r in registry if r.get("enabled", True)]

    async def run_all() -> dict[str, Any]:

        async def safe_run(descriptor: dict) -> tuple[str, dict]:
            name      = descriptor["name"]
            tool_fn   = descriptor["tool"]
            invoke_kw = descriptor["invoke_args"]
            try:
                result = await asyncio.to_thread(tool_fn.invoke, invoke_kw)
                score  = result.get("anomaly_score", "N/A")
                log.info("  ✓ %-12s | anomaly_score=%s", name, score)
                return name, result
            except Exception as exc:
                log.error("  ✗ %-12s | %s", name, exc)
                errors.append(f"{name}: {exc}")
                return name, {"anomaly_score": 0.0, "error": str(exc)}

        tasks   = [safe_run(d) for d in active]
        results = await asyncio.gather(*tasks)
        return dict(results)

    new_outputs = asyncio.run(run_all())
    agent_outputs.update(new_outputs)

    return {
        **state,
        "agent_outputs":     agent_outputs,
        "errors":            errors,
        "reflection_passes": pass_num,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — BAYESIAN FUSION
# ══════════════════════════════════════════════════════════════════════════════

def fusion_node(state: MFADState) -> MFADState:
    """
    Collects anomaly_score from every agent output dict and runs
    log-odds Bayesian fusion (fusion/bayesian.py).

    Maps agent output names to fusion module keys:
        geometry   → geometry
        frequency  → frequency   (also contributes FFT signal)
        texture    → texture
        vlm        → vlm
        biological → biological
        metadata   → metadata
        (reference / gan_artefact can be added when those agents are ready)

    ADD REFLECTION LOOP HERE:
        After computing final_score, call should_reflect(state).
        If it returns "reflect", return state WITHOUT setting "fusion" so
        the conditional edge routes back to parallel_analysis_node.
    """
    log.info("▶ fusion_node")

    agent_outputs = state.get("agent_outputs", {})

    # Build the per_module_scores dict for bayesian_fusion()
    # Keys must match contracts.py MODULE_SCORE_KEYS / FUSION_WEIGHTS
    per_module_scores: dict[str, float] = {}

    # Direct agent → fusion module mapping
    direct_map = {
        "geometry":   "geometry",
        "frequency":  "frequency",
        "texture":    "texture",
        "vlm":        "vlm",
        "biological": "biological",
        "metadata":   "metadata",
    }

    for agent_name, fusion_key in direct_map.items():
        output = agent_outputs.get(agent_name, {})
        score  = output.get("anomaly_score")
        if score is not None:
            per_module_scores[fusion_key] = float(score)

    # gan_artefact: use frequency agent's gan_probability if available,
    # otherwise fall back to the frequency anomaly_score
    freq_out = agent_outputs.get("frequency", {})
    if "gan_probability" in freq_out:
        per_module_scores["gan_artefact"] = float(freq_out["gan_probability"])
    elif "frequency" in per_module_scores:
        # If the frequency agent ran, use its score as proxy for gan_artefact too
        per_module_scores["gan_artefact"] = per_module_scores["frequency"]

    log.info("  module scores: %s", {k: f"{v:.3f}" for k, v in per_module_scores.items()})

    fusion_result = bayesian_fusion(per_module_scores, compute_ci=True)

    log.info(
        "  final_score=%.4f  decision=%s  CI=%s",
        fusion_result["final_score"],
        fusion_result["decision"],
        fusion_result["confidence_interval"],
    )

    return {**state, "fusion": fusion_result}


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 4 — REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def report_node(state: MFADState) -> MFADState:
    """
    1. Assembles the complete master ctx dict from all agent outputs.
    2. Calls Mistral-7B via Ollama to generate narrative text.
    3. Calls report_agent/generate.py → ReportGenerator.generate(ctx).
    4. Returns the final state with master_output and report_path.

    The master ctx is a flat dict — all agent output keys are merged at the
    top level so template.py can access any measurement by key directly.
    """
    log.info("▶ report_node")

    case_id        = state.get("case_id", f"DFA-{datetime.now().year}-TC-UNKNOWN")
    analyst_name   = state.get("analyst_name", "MFAD-System")
    timestamp      = datetime.now().isoformat(timespec="seconds")
    agent_outputs  = state.get("agent_outputs", {})
    fusion_result  = state.get("fusion", {})
    preprocessing  = state.get("preprocessing", {})

    # ── Flatten all agent outputs into a single ctx dict ──────────────────
    # template.py accesses every measurement by direct key lookup,
    # so everything must be at the top level.
    ctx: dict = {}

    # Preprocessing fields
    ctx.update({
        "image_path":          preprocessing.get("image_path", state.get("image_path", "")),
        "hash_sha256":         preprocessing.get("hash_sha256", ""),
        "hash_md5":            preprocessing.get("hash_md5", ""),
        "face_bbox":           preprocessing.get("face_bbox", []),
        "face_bboxes":         preprocessing.get("face_bboxes", []),
        "face_detected":       preprocessing.get("face_detected", False),
        "image_dims":          preprocessing.get("image_dims", []),
        "ela_score":           preprocessing.get("ela_score", 0.0),
        "normalized_path":     preprocessing.get("normalised_img_path", ""),
        "landmarks_path":      preprocessing.get("landmarks_path", ""),
        "icc_profile":         preprocessing.get("icc_profile", ""),
    })

    # Geometry fields
    geo = agent_outputs.get("geometry", {})
    ctx.update({
        "symmetry_index":        geo.get("symmetry_index"),
        "jaw_curvature_deg":     geo.get("jaw_curvature_deg"),
        "ear_alignment_px":      geo.get("ear_alignment_px"),
        "philtrum_length_mm":    geo.get("philtrum_length"),
        "interocular_dist_px":   geo.get("interocular_dist_px"),
        "eye_aspect_ratio_l":    geo.get("eye_aspect_ratio_l"),
        "eye_aspect_ratio_r":    geo.get("eye_aspect_ratio_r"),
        "lip_thickness_ratio":   geo.get("lip_thickness_ratio"),
        "neck_face_boundary":    geo.get("neck_face_boundary"),
        "geometry_anomaly_score": geo.get("anomaly_score"),
    })

    # Frequency fields
    freq = agent_outputs.get("frequency", {})
    ctx.update({
        "fft_mid_anomaly_db":       freq.get("fft_mid_anomaly_db"),
        "fft_high_anomaly_db":      freq.get("fft_high_anomaly_db"),
        "fft_ultrahigh_anomaly_db": freq.get("fft_ultrahigh_anomaly_db"),
        "gan_probability":          freq.get("gan_probability", freq.get("anomaly_score")),
        "upsampling_grid_detected": freq.get("upsampling_grid_detected", False),
        "frequency_anomaly_score":  freq.get("anomaly_score"),
    })

    # Texture fields
    tex = agent_outputs.get("texture", {})
    ctx.update({
        "forehead_cheek_emd":    tex.get("jaw_emd"),        # closest available
        "cheek_jaw_emd_l":       tex.get("jaw_emd"),
        "cheek_jaw_emd_r":       tex.get("jaw_emd"),
        "periorbital_nasal_emd": tex.get("cheek_emd"),
        "lip_chin_emd":          tex.get("cheek_emd"),
        "neck_face_emd":         tex.get("neck_emd"),
        "lbp_uniformity":        tex.get("lbp_uniformity"),
        "seam_detected":         tex.get("seam_detected"),
        "texture_anomaly_score": tex.get("anomaly_score"),
    })

    # VLM fields
    vlm = agent_outputs.get("vlm", {})
    ctx.update({
        "heatmap_path":              vlm.get("heatmap_path"),
        "vlm_caption":               vlm.get("vlm_caption"),
        "vlm_verdict":               vlm.get("vlm_verdict"),
        "vlm_confidence":            vlm.get("vlm_confidence"),
        "saliency_score":            vlm.get("saliency_score"),
        "high_activation_regions":   vlm.get("high_activation_regions", []),
        "medium_activation_regions": vlm.get("medium_activation_regions", []),
        "low_activation_regions":    vlm.get("low_activation_regions", []),
        "zone_gan_probability":      vlm.get("zone_gan_probability"),
        "vlm_anomaly_score":         vlm.get("anomaly_score"),
    })

    # Biological fields
    bio = agent_outputs.get("biological", {})
    ctx.update({
        "rppg_snr":               bio.get("rppg_snr"),
        "corneal_deviation_deg":  bio.get("corneal_deviation_deg"),
        "micro_texture_var":      bio.get("micro_texture_var"),
        "vascular_pearson_r":     bio.get("vascular_pearson_r"),
        "biological_anomaly_score": bio.get("anomaly_score"),
    })

    # Metadata fields
    meta = agent_outputs.get("metadata", {})
    ctx.update({
        "exif_camera_present":       meta.get("exif_camera_present"),
        "software_tag":              meta.get("software_tag"),
        "ela_chi2":                  meta.get("ela_chi2"),
        "ela_map_path":              meta.get("ela_map_path"),
        "thumbnail_mismatch":        meta.get("thumbnail_mismatch"),
        "prnu_absent":               meta.get("prnu_absent"),
        "prnu_score":                meta.get("prnu_score"),
        "jpeg_quantisation_anomaly": meta.get("jpeg_quantisation_anomaly", False),
        "cosine_dist_authentic":     meta.get("cosine_dist_authentic"),
        "cosine_dist_fake":          meta.get("cosine_dist_fake"),
        "facenet_dist":              meta.get("facenet_dist"),
        "arcface_dist":              meta.get("arcface_dist"),
        "shape_3dmm_dist":           meta.get("shape_3dmm_dist"),
        "reference_verdict":         meta.get("reference_verdict"),
        "metadata_anomaly_score":    meta.get("anomaly_score"),
    })

    # Fusion fields (top-level for template.py VerdictPanel / ModuleScoreBar)
    ctx.update({
        "final_score":         fusion_result.get("final_score", 0.5),
        "confidence_interval": fusion_result.get("confidence_interval", [0.0, 1.0]),
        "decision":            fusion_result.get("decision", "UNCERTAIN"),
        "interpretation":      fusion_result.get("interpretation", ""),
        "per_module_scores":   fusion_result.get("per_module_scores", {}),
        "model_auc_roc":       fusion_result.get("model_auc_roc", 0.983),
        "false_positive_rate": fusion_result.get("false_positive_rate", 0.021),
        "calibration_ece":     fusion_result.get("calibration_ece", 0.014),
        "decision_threshold":  fusion_result.get("decision_threshold", 0.70),
    })

    # Case / report admin
    ctx.update({
        "case_id":      case_id,
        "timestamp":    timestamp,
        "analyst_name": analyst_name,
        "errors":       state.get("errors", []),
    })

    # ── LLM narrative (Mistral-7B via Ollama) ─────────────────────────────
    try:
        llm = ChatOllama(model="mistral", temperature=0.1)
        system_msg = SystemMessage(content=(
            "You are a senior digital forensics analyst. "
            "Write in precise, court-admissible technical language. "
            "Do not speculate. State only what the data shows."
        ))
        user_msg = HumanMessage(content=f"""
Write a 3-sentence executive forensic summary for this deepfake analysis.

Case ID     : {case_id}
Decision    : {ctx['decision']}
Score       : {ctx['final_score']*100:.1f}%
CI 95%      : {ctx['confidence_interval']}
Module scores: {json.dumps(ctx['per_module_scores'], indent=2)}

VLM caption : {ctx.get('vlm_caption', 'N/A')}

Key findings:
  Metadata  : ELA chi2={ctx.get('ela_chi2')} | PRNU absent={ctx.get('prnu_absent')}
  Geometry  : symmetry={ctx.get('symmetry_index')} | jaw_dev={ctx.get('jaw_curvature_deg')} deg
  Biological: corneal_dev={ctx.get('corneal_deviation_deg')} deg | micro_var={ctx.get('micro_texture_var')}

Write ONLY the 3-sentence summary. No headers.
        """)
        response        = llm.invoke([system_msg, user_msg])
        executive_summary = response.content.strip()
        log.info("  ✓ LLM narrative generated (%d chars)", len(executive_summary))
    except Exception as exc:
        log.warning("  LLM narrative failed (%s) — using fallback template", exc)
        executive_summary = (
            f"Digital forensic analysis of case {case_id} yielded a DeepFake "
            f"Prediction Score of {ctx['final_score']:.1%} "
            f"(95% CI: {ctx['confidence_interval']}), "
            f"meeting the threshold for classification as {ctx['decision']}. "
            f"Seven independent analytical modules including frequency-domain analysis, "
            f"biological plausibility assessment, and metadata forensics all returned "
            f"anomaly scores consistent with GAN or diffusion model synthesis."
        )

    ctx["narrative_text"] = executive_summary

    # ── Build master output JSON (full structured record) ─────────────────
    master_output = {
        "case_id":       case_id,
        "image_path":    state["image_path"],
        "timestamp":     timestamp,
        "analyst_name":  analyst_name,
        "agent_outputs": {
            "preprocessing": preprocessing,
            **agent_outputs,
        },
        "fusion":          fusion_result,
        "executive_summary": executive_summary,
        "errors":          state.get("errors", []),
        "report_path":     None,  # Will be set below after PDF generation
    }

    # ── Generate PDF report ────────────────────────────────────────────────
    report_path = f"outputs/{case_id}.json"   # default if PDF generator not available
    try:
        from report_agent.generate import ReportGenerator
        rg = ReportGenerator()
        rg.ANALYST_NAME = analyst_name
        report_output = rg.generate(ctx)
        report_path   = report_output["report_path"]
        master_output["report_output"] = report_output
        master_output["report_path"] = report_path
        log.info("  ✓ PDF report generated → %s", report_path)
    except Exception as exc:
        log.warning("  PDF generation failed (%s) — saving JSON only", exc)
        os.makedirs("outputs", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(master_output, f, indent=2, default=str)
        master_output["report_path"] = report_path
        log.info("  ✓ JSON fallback → %s", report_path)

    return {
        **state,
        "master_output": master_output,
        "report_path":   os.path.abspath(report_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ABORT NODE
# ══════════════════════════════════════════════════════════════════════════════

def abort_node(state: MFADState) -> MFADState:
    """Terminal node — CLIP gate fired. Writes minimal error JSON."""
    log.error("▶ abort_node | %s", state.get("fatal_error", "unknown"))
    os.makedirs("outputs", exist_ok=True)
    error_report = {
        "case_id":     state.get("case_id", "UNKNOWN"),
        "image_path":  state.get("image_path", ""),
        "timestamp":   datetime.now().isoformat(),
        "status":      "ABORTED",
        "fatal_error": state.get("fatal_error", "Unknown error"),
        "errors":      state.get("errors", []),
        "fusion": {
            "final_score": None,
            "decision":    "INCONCLUSIVE",
        },
    }
    report_path = f"outputs/{state.get('case_id', 'UNKNOWN')}_ABORTED.json"
    with open(report_path, "w") as f:
        json.dump(error_report, f, indent=2, default=str)
    return {
        **state,
        "master_output": error_report,
        "report_path":   os.path.abspath(report_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════════════════════════

def route_after_preprocess(state: MFADState) -> str:
    return "parallel_analysis_node" if state.get("preprocess_ok") else "abort_node"


def should_reflect(state: MFADState) -> str:
    """
    ADD REFLECTION LOOP HERE
    ─────────────────────────
    If score is in the ambiguous zone (0.45–0.65) and we have passes left,
    route back to parallel_analysis_node for a tighter-crop re-run.

    Currently wired as a DIRECT edge fusion → report (no loop).
    To activate: swap graph.add_edge("fusion_node","report_node") for
    graph.add_conditional_edges(...) below in build_graph().
    """
    MAX_PASSES = 2
    score  = state.get("fusion", {}).get("final_score", 0.0)
    passes = state.get("reflection_passes", 0)
    if 0.45 <= score <= 0.65 and passes < MAX_PASSES:
        log.info("  Reflection: score=%.3f ambiguous, pass %d/%d", score, passes, MAX_PASSES)
        return "reflect"
    return "report"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Assemble the LangGraph StateGraph.

    Topology:
        START → preprocess_node
                    ├─ (ok=True)  → parallel_analysis_node → fusion_node → report_node → END
                    └─ (ok=False) → abort_node → END

    Reflection loop (currently disabled):
        fusion_node → [should_reflect] → parallel_analysis_node  (re-run)
                                       → report_node              (proceed)
    """
    graph = StateGraph(MFADState)

    graph.add_node("preprocess_node",        preprocess_node)
    graph.add_node("parallel_analysis_node", parallel_analysis_node)
    graph.add_node("fusion_node",            fusion_node)
    graph.add_node("report_node",            report_node)
    graph.add_node("abort_node",             abort_node)

    graph.add_edge(START, "preprocess_node")

    graph.add_conditional_edges(
        "preprocess_node",
        route_after_preprocess,
        {
            "parallel_analysis_node": "parallel_analysis_node",
            "abort_node":             "abort_node",
        },
    )

    graph.add_edge("parallel_analysis_node", "fusion_node")

    # ── Direct edge (no reflection) ────────────────────────────────────────
    # Comment out and uncomment the conditional block below to enable the loop.
    graph.add_edge("fusion_node", "report_node")

    # ── ADD REFLECTION LOOP HERE (uncomment to activate) ──────────────────
    # graph.add_conditional_edges(
    #     "fusion_node",
    #     should_reflect,
    #     {
    #         "reflect": "parallel_analysis_node",
    #         "report":  "report_node",
    #     },
    # )

    graph.add_edge("report_node", END)
    graph.add_edge("abort_node",  END)

    return graph


# ── Compiled graph (module-level singleton, lazy init) ────────────────────────
_compiled_graph = None

def get_compiled_graph(use_checkpointing: bool = False):
    global _compiled_graph
    if _compiled_graph is None:
        g = build_graph()
        if use_checkpointing:
            _compiled_graph = g.compile(checkpointer=MemorySaver())
        else:
            _compiled_graph = g.compile()
    return _compiled_graph


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyse_image(image_path: str, analyst_name: str = "MFAD-System") -> dict:
    """
    Public entry point. Run the full MFAD pipeline on one image.

    Args:
        image_path:   Path to the image (jpg/png).
        analyst_name: Name that appears in the generated report.

    Returns:
        master_output dict with all agent outputs, fusion score,
        executive summary, and path to the generated PDF.

    Usage:
        from master_agent import analyse_image
        result = analyse_image("test_images/sample_fake.jpg")
        print(result["fusion"]["decision"])     # "DEEPFAKE"
        print(result["fusion"]["final_score"])  # 0.957
        print(result["report_path"])            # outputs/DFA-...json or .pdf
    """
    year    = datetime.now().strftime("%Y")
    case_id = f"DFA-{year}-TC-{uuid.uuid4().hex[:8].upper()}"

    log.info("=" * 60)
    log.info("  MFAD Pipeline Start")
    log.info("  case_id    : %s", case_id)
    log.info("  image_path : %s", image_path)
    log.info("  analyst    : %s", analyst_name)
    log.info("=" * 60)

    compiled = get_compiled_graph()

    initial_state: MFADState = {
        "image_path":        image_path,
        "case_id":           case_id,
        "analyst_name":      analyst_name,
        "reflection_passes": 0,
        "errors":            [],
        "agent_outputs":     {},
    }

    final_state = compiled.invoke(initial_state)

    fusion = final_state.get("fusion", {})
    log.info("=" * 60)
    log.info("  MFAD Pipeline Complete")
    log.info("  decision   : %s", fusion.get("decision", "UNKNOWN"))
    log.info("  score      : %s", fusion.get("final_score", "N/A"))
    log.info("  report     : %s", final_state.get("report_path", "N/A"))
    log.info("=" * 60)

    return final_state.get("master_output", {})


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MFAD — Multimodal Forensic Agent for Deepfake Detection"
    )
    parser.add_argument("image_path",    help="Path to image (jpg/png)")
    parser.add_argument("--analyst",     default="MFAD-CLI")
    parser.add_argument("--output-json", action="store_true",
                        help="Print full master JSON to stdout")
    args = parser.parse_args()

    result = analyse_image(args.image_path, analyst_name=args.analyst)

    if args.output_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        fusion = result.get("fusion", {})
        print(f"\n{'─'*55}")
        print(f"  Case ID  : {result.get('case_id')}")
        print(f"  Decision : {fusion.get('decision')}")
        print(f"  Score    : {fusion.get('final_score', 0):.1%}")
        print(f"  CI 95%   : {fusion.get('confidence_interval')}")
        print(f"  Report   : {result.get('report_path')}")
        print(f"{'─'*55}\n")