"""
pipeline.py
===========
Connects all agents and combines their raw findings into a final verdict.

This is the ONLY place where scoring and verdicts happen.
Individual agents just measure — pipeline decides.

Flow:
    image_path
        └─► preprocessing_agent  →  outputs/preprocessing/<stem>.json
                └─► metadata_agent      →  outputs/metadata/<stem>.json
                        └─► (future agents)
                                └─► final verdict  →  outputs/final/<stem>_report.json

Scoring rules (documented explicitly):
    Each signal contributes a sub-score in [0,1].
    Final score = weighted average of all sub-scores.
    Verdict = MANIPULATED if final_score > 0.5 else AUTHENTIC.

Usage:
    python pipeline.py test_images\photo1.jpg      # single image
    python pipeline.py test_images\                # batch
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from agents.preprocessing_agent import PreprocessingAgent, run_preprocessing
from agents.metadata_agent       import MetadataAgent,       run_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("Pipeline")

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── scoring thresholds ────────────────────────────────────────────────────────
ELA_SCORE_THRESHOLD  = 0.15   # ela_score above this contributes to suspicion
ELA_CHI2_THRESHOLD   = 500.0  # ela_chi2 above this is suspicious
PRNU_VAR_THRESHOLD   = 0.05   # prnu_score below this = PRNU absent


# ═════════════════════════════════════════════════════════════════════════════
#  SCORING — all logic in one place
# ═════════════════════════════════════════════════════════════════════════════

def _score_preprocessing(pre: dict) -> tuple[float, dict]:
    """
    Score preprocessing signals.

    Signals used:
        ela_score : ELA mean pixel difference (0=clean, 1=altered)
                    Reliable on original JPEGs. Less reliable on
                    compressed/WhatsApp photos.

    Returns (score 0-1, breakdown dict)
    """
    ela  = pre["ela_score"]

    # ELA contribution — normalise to [0,1] using threshold
    ela_contrib = min(1.0, ela / ELA_SCORE_THRESHOLD*2) if ELA_SCORE_THRESHOLD > 0 else 0.0

    score = round(ela_contrib, 4)

    breakdown = {
        "ela_score":    ela,
        "ela_contrib":  ela_contrib,
        "final":        score,
    }
    return score, breakdown


def _score_metadata(meta: dict) -> tuple[float, dict]:
    """
    Score metadata signals.

    Signals and their reliability:
    ┌──────────────────────┬────────┬────────────────────────────────────────┐
    │ Signal               │ Weight │ Notes                                  │
    ├──────────────────────┼────────┼────────────────────────────────────────┤
    │ software_flagged     │  0.40  │ Very reliable — explicit editing tag   │
    │ thumbnail_mismatch   │  0.30  │ Reliable — post-capture edit evidence  │
    │ prnu_absent          │  0.20  │ Moderate — unreliable after compression│
    │ no camera + no date  │  0.10  │ Weak — real photos lose EXIF via apps  │
    └──────────────────────┴────────┴────────────────────────────────────────┘

    NOTE: ela_chi2 and camera_absent alone are NOT used as scoring signals
          because JPEG recompression (WhatsApp/email) inflates chi2 on real
          photos and strips EXIF, causing false positives.

    Returns (score 0-1, breakdown dict)
    """
    sw_flag       = meta["software_flagged"]
    thumb_miss    = meta["thumbnail_mismatch"]
    prnu_absent   = meta["prnu_absent"]
    has_camera    = meta["exif_camera_present"]
    has_datetime  = bool(meta["exif_datetime_original"])

    # no camera + no datetime = weak signal (both missing = more suspicious)
    no_provenance = (not has_camera) and (not has_datetime)

    sw_score    = 1.0 if sw_flag     else 0.0
    thumb_score = 1.0 if thumb_miss  else 0.0
    prnu_score  = 1.0 if prnu_absent else 0.0
    prov_score  = 1.0 if no_provenance else 0.0

    score = round(
        0.40 * sw_score    +
        0.30 * thumb_score +
        0.20 * prnu_score  +
        0.10 * prov_score,
        4
    )

    breakdown = {
        "software_flagged":   sw_flag,
        "thumbnail_mismatch": thumb_miss,
        "prnu_absent":        prnu_absent,
        "no_provenance":      no_provenance,
        "sw_contrib":         round(0.40 * sw_score,    4),
        "thumb_contrib":      round(0.30 * thumb_score, 4),
        "prnu_contrib":       round(0.20 * prnu_score,  4),
        "prov_contrib":       round(0.10 * prov_score,  4),
        "final":              score,
    }
    return score, breakdown


def _compute_final_score(pre_score: float,
                         meta_score: float) -> tuple[float, str]:
    """
    Combine agent scores into final score and verdict.

    Weights:
        preprocessing : 0.30  (ELA — useful but less reliable after compression)
        metadata      : 0.70  (explicit signals — more reliable)

    Add future agents:
        preprocessing : 0.15
        metadata      : 0.35
        texture       : 0.25
        gan_detector  : 0.25

    Verdict threshold: > 0.5 = MANIPULATED
    """
    final = round(
        0.30 * pre_score +
        0.70 * meta_score,
        4
    )
    verdict = "MANIPULATED" if final > 0.5 else "AUTHENTIC"
    return final, verdict


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_path: str) -> dict:
    """
    Run full pipeline on one image.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    dict — complete summary with all findings + final verdict
    """
    log.info("═══ pipeline  |  %s", Path(image_path).name)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    log.info("─── [1/2] preprocessing")
    pre_json_path = run_preprocessing(image_path)
    with open(pre_json_path) as f:
        pre = json.load(f)

    # ── Step 2: Metadata ──────────────────────────────────────────────────────
    log.info("─── [2/2] metadata")
    meta_json_path = run_metadata(pre_json_path)
    with open(meta_json_path) as f:
        meta = json.load(f)

    # ── Step N: add future agents here ────────────────────────────────────────
    # texture_json = run_texture(meta_json_path)
    # with open(texture_json) as f: texture = json.load(f)

    # ── Scoring ───────────────────────────────────────────────────────────────
    pre_score,  pre_breakdown  = _score_preprocessing(pre)
    meta_score, meta_breakdown = _score_metadata(meta)
   

    # ── Build full summary ────────────────────────────────────────────────────
    # NOTE: No verdict here. Verdict is computed after ALL agents run.
    # Each agent contributes its own anomaly_score.
    # Combined anomaly_score = weighted average of all agents.
    
    summary = {
        # ── identity & chain of custody ───────────────────────────────────────
        "image_path":   pre["image_path"],
        "hash_sha256":  pre["hash_sha256"],
        "hash_md5":     pre["hash_md5"],

        # ── face detection ────────────────────────────────────────────────────
        "face_detected": pre["face_detected"],
        "face_count":    pre["face_count"],
        "face_bboxes":   pre["face_bboxes"],   # all faces [[x1,y1,x2,y2],...]
        "face_bbox":     pre["face_bbox"],      # primary face [x1,y1,x2,y2]
        "image_dims":    pre["image_dims"],     # [W, H] original

        # ── preprocessing findings ────────────────────────────────────────────
        "ela_score":              pre["ela_score"],
        "pre_anomaly_score":      pre["anomaly_score"],

        # ── metadata findings ─────────────────────────────────────────────────
        "exif_camera_present":    meta["exif_camera_present"],
        "exif_camera_make":       meta["exif_camera_make"],
        "exif_camera_model":      meta["exif_camera_model"],
        "exif_datetime_original": meta["exif_datetime_original"],
        "exif_gps_present":       meta["exif_gps_present"],
        "software_tag":           meta["software_tag"],
        "software_flagged":       meta["software_flagged"],
        "ela_chi2":               meta["ela_chi2"],
        "ela_map_path":           meta["ela_map_path"],
        "thumbnail_mismatch":     meta["thumbnail_mismatch"],
        "prnu_score":             meta["prnu_score"],
        "prnu_absent":            meta["prnu_absent"],
        "meta_anomaly_score":     meta["anomaly_score"],


        # ── intermediate JSONs (audit trail) ──────────────────────────────────
        "preprocessing_json": pre_json_path,
        "metadata_json":      meta_json_path,

        # ── errors from all agents ────────────────────────────────────────────
        "errors": pre.get("errors", []) + meta.get("errors", []),
    }

    # save final report
    Path("outputs/final").mkdir(parents=True, exist_ok=True)
    stem       = Path(image_path).stem
    final_path = f"outputs/final/{stem}_report.json"
    with open(final_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("═══ pipeline done  |  anomaly=%.4f  →  %s", final_path)
    return summary


# ═════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN AGENT EXECUTOR  (optional)
# ═════════════════════════════════════════════════════════════════════════════

def run_langchain_agent(image_path: str) -> str:
    """
    LLM-driven version.
    Requires: pip install langchain-openai  +  OPENAI_API_KEY set
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools  = [PreprocessingAgent(), MetadataAgent()]
    llm    = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a forensic image analyst. "
         "1) Run preprocessing_agent first. "
         "2) Pass its JSON path to metadata_agent. "
         "3) Report all raw findings. Do not give a verdict — "
         "just report what each signal shows."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    executor = AgentExecutor(
        agent=create_openai_tools_agent(llm, tools, prompt),
        tools=tools, verbose=True)
    return executor.invoke({"input": f"Analyse: {image_path}"})["output"]


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline.py <image_path>")
        print("  python pipeline.py <folder/>")
        sys.exit(1)

    target = sys.argv[1]

    if Path(target).is_dir():
        from batch_run import run_batch
        run_batch(target)
    else:
        r    = run_pipeline(target)
        print(f"\n── Findings ────────────────────────────────────")
        print(f"  image_path         : {r['image_path']}")
        print(f"  hash_sha256        : {r['hash_sha256'][:20]}…")
        print(f"  hash_md5           : {r['hash_md5'][:20]}…")
        print(f"  face_detected      : {r['face_detected']}")
        print(f"  face_count         : {r['face_count']}")
        print(f"  face_bboxes        : {r['face_bboxes']}")
        print(f"  image_dims         : {r['image_dims']}")
        print(f"  ela_score          : {r['ela_score']}")
        print(f"  software_tag       : {r['software_tag']}")
        print(f"  software_flagged   : {r['software_flagged']}")
        print(f"  ela_chi2           : {r['ela_chi2']}")
        print(f"  thumbnail_mismatch : {r['thumbnail_mismatch']}")
        print(f"  prnu_absent        : {r['prnu_absent']}  (score={r['prnu_score']})")
        print(f"  pre_anomaly_score  : {r['pre_anomaly_score']}")
        print(f"  meta_anomaly_score : {r['meta_anomaly_score']}")
        print(f"  anomaly_score      : {r['anomaly_score']}  (combined)")
        print(f"\nFull report → outputs/final/{Path(target).stem}_report.json")
