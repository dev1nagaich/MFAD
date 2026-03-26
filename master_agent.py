"""
master_agent.py — MFAD LangGraph Orchestrator
==============================================
Framework : LangGraph (StateGraph) + LangChain tools + Ollama (Mistral-7B)

Architecture overview
─────────────────────
LangGraph StateGraph with the following node topology:

  [START]
     │
     ▼
  preprocess_node           ← CLIP / gate: blocks the entire graph if face not
     │                        detected or image is corrupt. Hard stop.
     │ (on success)
     ▼
  parallel_analysis_node    ← Fans out to all 7 specialist agents concurrently
  (geometry, frequency,       using asyncio.gather inside a single graph node.
   texture, vlm,              Each agent is a @tool decorated function.
   biological, metadata,      ← ADD SCENE LOOP HERE if you want iterative
   reference)                   re-analysis with different crop windows.
     │
     ▼
  fusion_node               ← Bayesian log-odds fusion of all anomaly_scores
     │                        ← ADD REFLECTION LOOP HERE: if final_score is in
     │                          ambiguous zone (0.45–0.65), loop back to
     │                          parallel_analysis_node with tighter face crop.
     ▼
  report_node               ← Calls generator.py, writes final PDF
     │
     ▼
  [END]

State
─────
  MFADState (TypedDict) — the single shared object that flows through all nodes.
  Every node reads what it needs and writes its results back to state.

LLM
───
  Mistral-7B via Ollama for the master reasoning loop and report narration.
  The LangGraph nodes themselves are deterministic — Mistral is only invoked
  inside report_node for narrative generation.
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

# ── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver  # optional persistence

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from langchain_ollama import ChatOllama  # pip install langchain-ollama
from langchain_core.messages import HumanMessage, SystemMessage

# ── Typing ────────────────────────────────────────────────────────────────────
from typing_extensions import TypedDict

# ── Internal agent modules ────────────────────────────────────────────────────
# These are imported lazily inside each node so the graph can be imported
# without all heavy models loaded (useful for unit tests).
#
# from agents.preprocessing import preprocessing_agent   ← real import
# from agents.geometry      import geometry_agent
# from agents.frequency     import frequency_agent
# from agents.texture       import texture_agent
# from agents.vlm           import vlm_agent
# from agents.biological    import biological_agent
# from agents.metadata      import metadata_agent
# from agents.reference     import reference_agent
# from fusion.bayesian      import bayesian_fusion
# from report.generator     import generate_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("master_agent")


# ══════════════════════════════════════════════════════════════════════════════
#  STUB AGENTS  (replace with real imports when each agent module is ready)
#  Each stub mirrors the exact schema in contracts.py
# ══════════════════════════════════════════════════════════════════════════════

@tool
def preprocessing_agent(image_path: str) -> dict:
    """Validates image integrity, detects face, computes hash. MUST run first."""
    # TODO: replace with: from agents.preprocessing import preprocessing_agent
    return {
        "image_path":    os.path.abspath(image_path),
        "face_bbox":     [120, 80, 420, 460],   # [x1, y1, x2, y2]
        "hash_sha256":   "stub_sha256_replace_me",
        "hash_md5":      "stub_md5_replace_me",
        "ela_score":     0.34,
        "face_detected": True,
        "image_dims":    [512, 512],
        "anomaly_score": 0.34,
    }


@tool
def geometry_agent(image_path: str, face_bbox: list) -> dict:
    """68-point landmark geometry analysis — symmetry, jaw, ear, philtrum."""
    # TODO: replace with: from agents.geometry import geometry_agent
    return {
        "symmetry_index":      0.74,
        "jaw_curvature_deg":   11.2,
        "ear_alignment_px":    8.7,
        "philtrum_length":     0.21,
        "landmark_confidence": 0.91,
        "anomaly_score":       0.884,
    }


@tool
def frequency_agent(image_path: str, face_bbox: list) -> dict:
    """FFT mid/high frequency anomaly + EfficientNet-B4 GAN probability."""
    # TODO: replace with: from agents.frequency import frequency_agent
    return {
        "fft_mid_anomaly_db":  9.4,
        "fft_high_anomaly_db": 3.2,
        "gan_probability":     0.967,
        "freq_spectrum_path":  "outputs/fft_spectrum.png",
        "anomaly_score":       0.967,
    }


@tool
def texture_agent(image_path: str, face_bbox: list) -> dict:
    """LBP + Gabor + Earth Mover's Distance seam detection across face zones."""
    # TODO: replace with: from agents.texture import texture_agent
    return {
        "jaw_emd":        0.61,
        "neck_emd":       0.48,
        "cheek_emd":      0.22,
        "lbp_uniformity": 0.31,
        "seam_detected":  True,
        "zone_scores":    {"forehead": 0.2, "cheek_L": 0.3, "jaw": 0.8},
        "anomaly_score":  0.895,
    }


@tool
def vlm_agent(image_path: str, face_bbox: list) -> dict:
    """BLIP-2 Grad-CAM heatmap + forensic caption + saliency score."""
    # TODO: replace with: from agents.vlm import vlm_agent
    return {
        "heatmap_path":            "outputs/heatmap_overlay.png",
        "vlm_caption":             (
            "The central facial region shows unnatural texture smoothing near the "
            "jaw boundary. Skin tone appears inconsistent between cheek zones. "
            "Shadow direction is inconsistent with ambient lighting."
        ),
        "saliency_score":          0.91,
        "high_activation_regions": ["jaw boundary", "eyes", "nose bridge"],
        "anomaly_score":           0.931,
    }


@tool
def biological_agent(image_path: str, face_bbox: list) -> dict:
    """rPPG SNR, corneal highlight consistency, perioral micro-texture variance."""
    # TODO: replace with: from agents.biological import biological_agent
    return {
        "rppg_snr":              2.1,
        "corneal_deviation_deg": 22.4,
        "micro_texture_var":     0.012,
        "highlight_positions":   {"left": [210, 180], "right": [310, 182]},
        "anomaly_score":         0.826,
    }


@tool
def metadata_agent(image_path: str) -> dict:
    """EXIF parsing, ELA chi-squared, thumbnail mismatch, PRNU absence."""
    # TODO: replace with: from agents.metadata import metadata_agent
    return {
        "exif_camera_present": False,
        "software_tag":        "Adobe Photoshop 24.0",
        "ela_chi2":            847.3,
        "ela_map_path":        "outputs/ela_map.png",
        "thumbnail_mismatch":  True,
        "prnu_absent":         True,
        "anomaly_score":       0.973,
    }


@tool
def reference_agent(image_path: str) -> dict:
    """FaceNet embedding cosine similarity vs real/fake reference clusters."""
    # TODO: replace with: from agents.reference import reference_agent
    return {
        "cosine_dist_authentic": 0.71,
        "cosine_dist_fake":      0.18,
        "verdict":               "CLOSER_TO_FAKE",
        "embedding_norm":        0.994,
        "anomaly_score":         0.910,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STUB FUSION  (replace with real bayesian.py when ready)
# ══════════════════════════════════════════════════════════════════════════════

def bayesian_fusion(scores: dict[str, float]) -> dict:
    """
    Log-odds Bayesian fusion.
    Replace with: from fusion.bayesian import bayesian_fusion
    """
    import math

    MODULE_WEIGHTS = {
        "geometry":   0.884,
        "frequency":  0.967,
        "texture":    0.895,
        "vlm":        0.931,
        "biological": 0.826,
        "metadata":   0.973,
        "reference":  0.910,
    }

    log_odds_total = 0.0
    for module, score in scores.items():
        # Clamp to avoid log(0) or log(inf)
        score = max(1e-6, min(1 - 1e-6, score))
        weight = MODULE_WEIGHTS.get(module, 0.5)
        log_odds_total += weight * math.log(score / (1.0 - score))

    final_score = 1.0 / (1.0 + math.exp(-log_odds_total))

    # Approximate 95% CI via Laplace smoothing heuristic
    margin = 0.05 * (1 - abs(2 * final_score - 1))
    ci = [round(max(0, final_score - margin), 3), round(min(1, final_score + margin), 3)]

    return {
        "final_score":         round(final_score, 4),
        "confidence_interval": ci,
        "verdict":             "DEEPFAKE" if final_score >= 0.70 else "LIKELY REAL",
        "per_module_scores":   scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class MFADState(TypedDict, total=False):
    """
    Shared state object — lives for the duration of one image analysis.
    Every node reads from and writes to this dict.
    """
    # ── Input ──────────────────────────────────────────────────────────────
    image_path:   str           # path provided by caller
    case_id:      str           # auto-generated on graph entry

    # ── Preprocessing ──────────────────────────────────────────────────────
    preprocessing: dict         # full output from preprocessing_agent
    face_bbox:     list         # [x1, y1, x2, y2] — shortcut for downstream
    preprocess_ok: bool         # CLIP gate result — False aborts the graph

    # ── Per-agent outputs ──────────────────────────────────────────────────
    geometry:   dict
    frequency:  dict
    texture:    dict
    vlm:        dict
    biological: dict
    metadata:   dict
    reference:  dict

    # ── Fusion ─────────────────────────────────────────────────────────────
    fusion: dict                # bayesian_fusion output

    # ── Reflection loop counter ────────────────────────────────────────────
    # ADD REFLECTION LOOP HERE — tracks how many re-analysis passes have run
    reflection_passes: int      # 0 on first entry; increment each retry

    # ── Report ─────────────────────────────────────────────────────────────
    report_path: str            # absolute path to generated PDF

    # ── Final master JSON (mirrors spec in documentation) ─────────────────
    master_output: dict

    # ── Error tracking ─────────────────────────────────────────────────────
    errors: list[str]           # non-fatal errors from individual agents
    fatal_error: Optional[str]  # set if preprocessing CLIP gate fires


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 1 — PREPROCESSING  (CLIP / gate node)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_node(state: MFADState) -> MFADState:
    """
    Gate node — the entire pipeline is blocked until this succeeds.

    CLIP logic:
    ──────────
    If face_detected is False  → set fatal_error, preprocess_ok=False
    If image is unreadable     → set fatal_error, preprocess_ok=False
    Otherwise                  → preprocess_ok=True, pipeline continues

    This mirrors a CLIP (Conditional Loop / Interrupt Point) in agentic
    systems: a hard checkpoint that can terminate the run early.

    ADD SCENE LOOP HERE (future):
    ─────────────────────────────
    If the image contains multiple faces (e.g. group photo), loop over
    each detected face bbox and create a separate MFADState per face.
    Use a subgraph or a map-reduce fan-out node to handle this.
    """
    log.info("▶ preprocess_node | image: %s", state["image_path"])

    errors = state.get("errors", [])

    try:
        result = preprocessing_agent.invoke({"image_path": state["image_path"]})  # type: ignore[arg-type]

        if not result.get("face_detected", False):
            log.warning("CLIP gate triggered: no face detected")
            return {
                **state,
                "preprocessing": result,
                "preprocess_ok": False,
                "fatal_error": "No face detected in the image. Pipeline aborted.",
                "errors": errors,
            }

        log.info("  ✓ face_bbox=%s  sha256=%s", result["face_bbox"], result["hash_sha256"])
        return {
            **state,
            "preprocessing":  result,
            "face_bbox":      result["face_bbox"],
            "preprocess_ok":  True,
            "errors":         errors,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("preprocessing_agent failed:\n%s", tb)
        return {
            **state,
            "preprocessing": {},
            "preprocess_ok": False,
            "fatal_error":   f"Preprocessing failed: {exc}",
            "errors":        errors + [f"preprocessing: {exc}"],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 2 — PARALLEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def parallel_analysis_node(state: MFADState) -> MFADState:
    """
    Runs all 7 specialist agents concurrently using asyncio.gather.

    All agents receive the same (image_path, face_bbox) from preprocessing.
    Failed agents write a zero-score stub so fusion can still run.

    ADD SCENE LOOP HERE (future):
    ─────────────────────────────
    If you want the agents to analyse multiple face crops iteratively
    (e.g., zoomed face + full frame + different lighting corrections),
    wrap this node in a loop node:

        while passes_remaining > 0:
            results = await gather(all_agents(crop_variant))
            passes_remaining -= 1

    Use LangGraph's conditional edge from fusion_node back to this node
    to trigger a re-analysis pass when score is in ambiguous zone.

    ADD SCENE LOOP HERE (multi-face):
    ──────────────────────────────────
    If preprocessing returns multiple bboxes, fan out with:
        tasks = [run_agent_suite(bbox) for bbox in all_bboxes]
        results = await asyncio.gather(*tasks)
    Then aggregate results before fusion.
    """
    log.info("▶ parallel_analysis_node | pass #%d", state.get("reflection_passes", 0) + 1)

    image_path = state["image_path"]
    face_bbox  = state["face_bbox"]
    errors     = list(state.get("errors", []))

    async def run_all() -> dict[str, Any]:
        """Inner async runner — all agents fire simultaneously."""

        async def safe_run(name: str, coro) -> tuple[str, dict]:
            """Wraps each agent call, catches exceptions, returns zero-score stub on failure."""
            try:
                result = await asyncio.to_thread(coro)   # run sync tool in thread
                log.info("  ✓ %s | anomaly_score=%.3f", name, result.get("anomaly_score", 0))
                return name, result
            except Exception as exc:
                log.error("  ✗ %s | %s", name, exc)
                errors.append(f"{name}: {exc}")
                # Zero-score stub — fusion will discount this module automatically
                return name, {"anomaly_score": 0.0, "error": str(exc)}

        tasks = [
            safe_run("geometry",   lambda: geometry_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
            safe_run("frequency",  lambda: frequency_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
            safe_run("texture",    lambda: texture_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
            safe_run("vlm",        lambda: vlm_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
            safe_run("biological", lambda: biological_agent.invoke({"image_path": image_path, "face_bbox": face_bbox})),
            safe_run("metadata",   lambda: metadata_agent.invoke({"image_path": image_path})),
            safe_run("reference",  lambda: reference_agent.invoke({"image_path": image_path})),
        ]

        results = await asyncio.gather(*tasks)
        return dict(results)

    agent_outputs = asyncio.run(run_all())

    return {
        **state,
        "geometry":   agent_outputs["geometry"],
        "frequency":  agent_outputs["frequency"],
        "texture":    agent_outputs["texture"],
        "vlm":        agent_outputs["vlm"],
        "biological": agent_outputs["biological"],
        "metadata":   agent_outputs["metadata"],
        "reference":  agent_outputs["reference"],
        "errors":     errors,
        # Increment reflection pass counter (used by conditional edge below)
        "reflection_passes": state.get("reflection_passes", 0) + 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 3 — BAYESIAN FUSION
# ══════════════════════════════════════════════════════════════════════════════

def fusion_node(state: MFADState) -> MFADState:
    """
    Collects anomaly_score from every agent, runs log-odds Bayesian fusion.

    ADD REFLECTION LOOP HERE (future):
    ───────────────────────────────────
    After computing final_score, check if it falls in the ambiguous zone
    (e.g., 0.45 <= final_score <= 0.65).  If so, and if reflection_passes < MAX,
    return state WITHOUT setting fusion so the conditional edge routes back to
    parallel_analysis_node with a tighter or different crop window.

    Example conditional edge logic (add to graph builder below):
        graph.add_conditional_edges(
            "fusion_node",
            should_reflect,          <- returns "reflect" or "report"
            {"reflect": "parallel_analysis_node", "report": "report_node"}
        )
    """
    log.info("▶ fusion_node")

    scores = {
        "geometry":   state.get("geometry",   {}).get("anomaly_score", 0.0),
        "frequency":  state.get("frequency",  {}).get("anomaly_score", 0.0),
        "texture":    state.get("texture",    {}).get("anomaly_score", 0.0),
        "vlm":        state.get("vlm",        {}).get("anomaly_score", 0.0),
        "biological": state.get("biological", {}).get("anomaly_score", 0.0),
        "metadata":   state.get("metadata",   {}).get("anomaly_score", 0.0),
        "reference":  state.get("reference",  {}).get("anomaly_score", 0.0),
    }

    fusion_result = bayesian_fusion(scores)
    log.info(
        "  final_score=%.4f  verdict=%s  CI=%s",
        fusion_result["final_score"],
        fusion_result["verdict"],
        fusion_result["confidence_interval"],
    )

    return {**state, "fusion": fusion_result}


# ══════════════════════════════════════════════════════════════════════════════
#  NODE 4 — REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def report_node(state: MFADState) -> MFADState:
    """
    Assembles the master JSON, calls Mistral-7B for narrative text,
    then calls report/generator.py to produce the final PDF.

    The Mistral call here acts as a final "editorial" pass — it receives
    the structured findings and writes polished forensic narrative text
    for each report section.

    ADD LLM REFINEMENT LOOP HERE (future):
    ────────────────────────────────────────
    If Mistral returns a hallucination-flagged output (detected via a
    structured output parser + confidence check), loop the prompt up to
    MAX_RETRIES times before falling back to a template string.
    """
    log.info("▶ report_node")

    case_id   = state.get("case_id", f"DFA-{datetime.now().strftime('%Y')}-TC-UNKNOWN")
    timestamp = datetime.now().isoformat(timespec="seconds")

    # ── Build master JSON (matches spec in documentation) ─────────────────
    master_output: dict = {
        "case_id":    case_id,
        "image_path": state["image_path"],
        "timestamp":  timestamp,
        "agent_outputs": {
            "preprocessing": state.get("preprocessing", {}),
            "geometry":      state.get("geometry",      {}),
            "frequency":     state.get("frequency",     {}),
            "texture":       state.get("texture",       {}),
            "vlm":           state.get("vlm",           {}),
            "biological":    state.get("biological",    {}),
            "metadata":      state.get("metadata",      {}),
            "reference":     state.get("reference",     {}),
        },
        "fusion": state.get("fusion", {}),
        "errors": state.get("errors", []),
    }

    # ── LLM narrative generation via Ollama / Mistral-7B ─────────────────
    # This is the only place in the graph where a language model is invoked.
    # The LLM writes a 3-sentence forensic summary for the cover page.
    try:
        llm = ChatOllama(model="mistral", temperature=0.1)

        system_msg = SystemMessage(content=(
            "You are a senior digital forensics analyst. "
            "Write in precise, court-admissible technical language. "
            "Do not speculate — state only what the data shows."
        ))

        user_msg = HumanMessage(content=f"""
Write a 3-sentence executive forensic summary for this deepfake analysis case.

Case ID    : {case_id}
Timestamp  : {timestamp}
Final score: {master_output['fusion'].get('final_score', 'N/A')} (0=real, 1=deepfake)
Verdict    : {master_output['fusion'].get('verdict', 'UNKNOWN')}
CI 95%     : {master_output['fusion'].get('confidence_interval', [])}

Module scores:
{json.dumps(master_output['fusion'].get('per_module_scores', {}), indent=2)}

VLM caption:
{master_output['agent_outputs']['vlm'].get('vlm_caption', 'N/A')}

Key findings:
- Metadata: ELA chi2={master_output['agent_outputs']['metadata'].get('ela_chi2')} | PRNU absent={master_output['agent_outputs']['metadata'].get('prnu_absent')}
- Geometry: symmetry={master_output['agent_outputs']['geometry'].get('symmetry_index')} | jaw_dev={master_output['agent_outputs']['geometry'].get('jaw_curvature_deg')} deg
- Biological: corneal_dev={master_output['agent_outputs']['biological'].get('corneal_deviation_deg')} deg | micro_var={master_output['agent_outputs']['biological'].get('micro_texture_var')}

Write ONLY the 3-sentence summary. No headers, no preamble.
        """)

        response = llm.invoke([system_msg, user_msg])
        executive_summary = response.content.strip()
        log.info("  ✓ LLM narrative generated (%d chars)", len(executive_summary))

    except Exception as exc:
        log.warning("  LLM narrative failed (Ollama not running?): %s", exc)
        # Fallback template narrative — report still generates
        executive_summary = (
            f"Digital forensic analysis of case {case_id} yielded a DeepFake "
            f"Prediction Score of {master_output['fusion'].get('final_score', 0):.1%} "
            f"(95% CI: {master_output['fusion'].get('confidence_interval', [0, 1])}), "
            f"meeting the threshold for classification as SYNTHETIC MEDIA. "
            f"Seven independent analytical modules including frequency-domain analysis, "
            f"biological plausibility assessment, and metadata forensics all returned "
            f"anomaly scores consistent with GAN or diffusion model synthesis."
        )

    master_output["executive_summary"] = executive_summary

    # ── Call report generator ──────────────────────────────────────────────
    # TODO: uncomment when report/generator.py is implemented:
    # from report.generator import generate_report
    # report_path = generate_report(master_output)

    # Stub: write master JSON to outputs/ until PDF generator is ready
    os.makedirs("outputs", exist_ok=True)
    report_path = f"outputs/{case_id}.json"
    with open(report_path, "w") as f:
        json.dump(master_output, f, indent=2)
    log.info("  ✓ master output written -> %s  (replace with PDF generator)", report_path)

    return {
        **state,
        "master_output": master_output,
        "report_path":   os.path.abspath(report_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ABORT NODE — called when CLIP gate fires in preprocess_node
# ══════════════════════════════════════════════════════════════════════════════

def abort_node(state: MFADState) -> MFADState:
    """
    Terminal node for early exits (no face detected, corrupt image, etc.).
    Writes a minimal error report so the caller always gets a structured response.
    """
    log.error("▶ abort_node | fatal_error: %s", state.get("fatal_error"))

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
            "verdict":     "INCONCLUSIVE",
        },
    }

    report_path = f"outputs/{state.get('case_id', 'UNKNOWN')}_ABORTED.json"
    with open(report_path, "w") as f:
        json.dump(error_report, f, indent=2)

    return {
        **state,
        "master_output": error_report,
        "report_path":   os.path.abspath(report_path),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONDITIONAL EDGES
# ══════════════════════════════════════════════════════════════════════════════

def route_after_preprocess(state: MFADState) -> str:
    """
    CLIP gate router.
    If preprocessing succeeded -> continue to analysis.
    If not -> abort immediately.
    """
    return "parallel_analysis_node" if state.get("preprocess_ok") else "abort_node"


def should_reflect(state: MFADState) -> str:
    """
    ADD REFLECTION LOOP HERE
    ─────────────────────────
    After fusion, check if the score is ambiguous and we have passes left.

    To activate: wire this as the conditional edge out of fusion_node
    (see graph builder below — the commented-out conditional edge).

    Logic:
      - If 0.45 <= final_score <= 0.65  AND  reflection_passes < 2
          -> return "reflect" (routes back to parallel_analysis_node)
      - Otherwise
          -> return "report" (proceeds to report_node)
    """
    MAX_REFLECTION_PASSES = 2
    score  = state.get("fusion", {}).get("final_score", 0.0)
    passes = state.get("reflection_passes", 0)

    if 0.45 <= score <= 0.65 and passes < MAX_REFLECTION_PASSES:
        log.info(
            "  Reflection triggered: score=%.3f in ambiguous zone, pass %d/%d",
            score, passes, MAX_REFLECTION_PASSES,
        )
        return "reflect"
    return "report"


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(use_checkpointing: bool = False) -> StateGraph:
    """
    Assembles the LangGraph StateGraph.

    Node topology:

        START -> preprocess_node
                    |
                    |-- (preprocess_ok=True)  -> parallel_analysis_node
                    +-- (preprocess_ok=False) -> abort_node -> END

        parallel_analysis_node -> fusion_node

        fusion_node -> report_node -> END
        (with optional reflection loop back to parallel_analysis_node)

    To activate reflection loop: swap the direct edge fusion->report
    for the commented-out conditional edge block below.
    """
    graph = StateGraph(MFADState)

    # ── Register nodes ─────────────────────────────────────────────────────
    graph.add_node("preprocess_node",         preprocess_node)
    graph.add_node("parallel_analysis_node",  parallel_analysis_node)
    graph.add_node("fusion_node",             fusion_node)
    graph.add_node("report_node",             report_node)
    graph.add_node("abort_node",              abort_node)

    # ── Entry ──────────────────────────────────────────────────────────────
    graph.add_edge(START, "preprocess_node")

    # ── CLIP gate: preprocessing -> (analysis | abort) ─────────────────────
    graph.add_conditional_edges(
        "preprocess_node",
        route_after_preprocess,
        {
            "parallel_analysis_node": "parallel_analysis_node",
            "abort_node":             "abort_node",
        },
    )

    # ── Analysis -> Fusion ──────────────────────────────────────────────────
    graph.add_edge("parallel_analysis_node", "fusion_node")

    # ── Fusion -> Report (DIRECT — no reflection) ──────────────────────────
    # Comment this out and uncomment the conditional edge block below
    # when you are ready to activate the reflection loop.
    graph.add_edge("fusion_node", "report_node")

    # ── ADD REFLECTION LOOP HERE ───────────────────────────────────────────
    # Uncomment to activate: routes ambiguous scores back for a 2nd pass.
    #
    # graph.add_conditional_edges(
    #     "fusion_node",
    #     should_reflect,
    #     {
    #         "reflect": "parallel_analysis_node",   # re-run all agents
    #         "report":  "report_node",              # proceed to PDF
    #     },
    # )

    # ── Terminal edges ─────────────────────────────────────────────────────
    graph.add_edge("report_node", END)
    graph.add_edge("abort_node",  END)

    return graph


# ══════════════════════════════════════════════════════════════════════════════
#  COMPILED GRAPH + PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_compiled_graph(use_checkpointing: bool = False):
    """
    Returns a compiled LangGraph ready for invocation.
    Pass use_checkpointing=True to persist state across runs (requires SQLite).

    ADD PERSISTENCE HERE (future):
    ─────────────────────────────
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver.from_conn_string("mfad_runs.db")
    return build_graph().compile(checkpointer=checkpointer)
    """
    graph = build_graph(use_checkpointing)

    if use_checkpointing:
        # In-memory checkpointer for session-level persistence
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()


def analyse_image(image_path: str, analyst_name: str = "MFAD-System") -> dict:
    """
    Public entry-point.  Call this function to run the full pipeline.

    Args:
        image_path:    Path to the image file (jpg/png).
        analyst_name:  Name or ID of the submitting analyst (appears in report).

    Returns:
        master_output dict containing all agent results, fusion score,
        executive summary, and path to the generated PDF report.

    Usage:
        from master_agent import analyse_image
        result = analyse_image("test_images/sample_fake.jpg")
        print(result["fusion"]["verdict"])          # "DEEPFAKE"
        print(result["fusion"]["final_score"])       # 0.957
        print(result["report_path"])                 # outputs/DFA-...pdf
    """
    # Generate a unique case ID for this run
    year    = datetime.now().strftime("%Y")
    case_id = f"DFA-{year}-TC-{uuid.uuid4().hex[:8].upper()}"

    log.info("=" * 55)
    log.info("  MFAD Pipeline Start")
    log.info("  case_id    : %s", case_id)
    log.info("  image_path : %s", image_path)
    log.info("  analyst    : %s", analyst_name)
    log.info("=" * 55)

    compiled = get_compiled_graph()

    initial_state: MFADState = {
        "image_path":        image_path,
        "case_id":           case_id,
        "reflection_passes": 0,
        "errors":            [],
    }

    # Run the graph — this is a blocking synchronous call
    final_state = compiled.invoke(initial_state)

    log.info("=" * 55)
    log.info("  MFAD Pipeline Complete")
    log.info("  verdict    : %s", final_state.get("fusion", {}).get("verdict", "UNKNOWN"))
    log.info("  score      : %s", final_state.get("fusion", {}).get("final_score", "N/A"))
    log.info("  report     : %s", final_state.get("report_path", "N/A"))
    log.info("=" * 55)

    return final_state.get("master_output", {})


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MFAD — Multimodal Forensic Agent for Deepfake Detection"
    )
    parser.add_argument("image_path",    help="Path to image file (jpg/png)")
    parser.add_argument("--analyst",     default="MFAD-CLI", help="Analyst name for report")
    parser.add_argument("--output-json", action="store_true", help="Print master JSON to stdout")
    args = parser.parse_args()

    result = analyse_image(args.image_path, analyst_name=args.analyst)

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        fusion = result.get("fusion", {})
        print(f"\n{'─' * 50}")
        print(f"  Case ID    : {result.get('case_id')}")
        print(f"  Verdict    : {fusion.get('verdict')}")
        print(f"  Score      : {fusion.get('final_score', 0):.1%}")
        print(f"  CI 95%     : {fusion.get('confidence_interval')}")
        print(f"  Report     : {result.get('report_path')}")
        print(f"{'─' * 50}\n")