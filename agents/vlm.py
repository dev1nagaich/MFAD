"""
agents/vlm.py — MFAD Multi-Factor AI Deepfake Detection System
==============================================================
Agent 5 — VLM Explainability Heat-Map Analysis
Report section: §5.4

MODEL USED: LLaVA-1.5-7b (llava-hf/llava-1.5-7b-hf) ~14 GB
  - Better instruction following than BLIP-2 for forensic prompts
  - Downloaded ONCE on first run, cached at ~/.cache/huggingface/hub/
  - Subsequent runs load from cache instantly

GRAD-CAM STATUS: placeholder — returns neutral values
  - Your teammate (Agent 3 owner) is fine-tuning EfficientNet-B4
  - When ready: replace _run_gradcam_placeholder() below
  - Search for: # TEAMMATE MERGE POINT

HOW TO TEST STANDALONE (without the full pipeline):
  Run:  python tests/test_vlm_standalone.py
  This loads tests/dummy_ctx.json and runs the full agent on it.
  You do NOT need any other agent to be working.

Output keys (contracts.py VLM_KEYS — never change these):
  heatmap_path, vlm_caption, vlm_verdict, vlm_confidence,
  saliency_score, high_activation_regions, medium_activation_regions,
  low_activation_regions, zone_gan_probability, anomaly_score

Depends on from ctx (produced by Agent 1 PreprocessingAgent):
  ctx["image_path"]      — path to the original suspect image
  ctx["face_bbox"]       — [x1, y1, x2, y2] face location in pixels
  ctx["face_crop_path"]  — path to the cropped face image

FIRST RUN WARNING:
  LLaVA-1.5-7b is ~14 GB. Downloads automatically on first call.
  Ensure disk space and stable internet before running for the first time.
  After first run it is cached locally — no re-download ever needed.
"""

import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # no display needed — saves directly to PNG file
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from transformers import LlavaProcessor, LlavaForConditionalGeneration

# contracts.py lives in the parent directory (mfad/)
# This line adds the parent to Python's search path so the import works
# whether you run this file directly or import it from master_agent.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts import validate, VLM_KEYS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Grad-CAM zone thresholds from §5.4 of reference report
RED_THRESHOLD  = 0.8    # above this  → RED zone   (high_activation_regions)
BLUE_THRESHOLD = 0.5    # above this  → BLUE zone  (medium_activation_regions)
                        # below 0.5   → VIOLET zone (low_activation_regions)

# LLaVA-1.5 uses this exact chat format.
# USER: <image> tells the model where the image goes in the conversation.
# ASSISTANT: is left empty — the model fills it in.
# Do not change this format — it is specific to LLaVA-1.5.
FORENSIC_PROMPT = (
    "USER: <image>\n"
    "Is this a face or not if yes?,"
    "Look carefully at: skin texture, eye reflections, jaw edges, "
    "and the boundary between the face and neck. "
    "First state REAL or FAKE, then describe the specific suspicious "
    "features you can see.\nASSISTANT:"
)

# Nine named face regions for zone classification.
# Each value is (x_start%, y_start%, x_end%, y_end%) as fractions of image size.
# E.g. "eyes": (0.10, 0.20, 0.90, 0.45) means:
#   horizontal band from 10% to 90% of width, 20% to 45% of height.
FACE_REGIONS = {
    "eyes":       (0.10, 0.20, 0.90, 0.45),
    "nose":       (0.25, 0.35, 0.75, 0.65),
    "mouth":      (0.20, 0.60, 0.80, 0.80),
    "cheeks":     (0.00, 0.30, 1.00, 0.65),
    "brow":       (0.05, 0.10, 0.95, 0.30),
    "chin":       (0.20, 0.75, 0.80, 1.00),
    "background": (0.00, 0.00, 1.00, 0.10),
    "hair":       (0.00, 0.00, 1.00, 0.15),
    "shoulders":  (0.00, 0.90, 1.00, 1.00),
}

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE
# These start as None. _load_llava() fills them on the first call and
# every subsequent call reuses the already-loaded model.
# This ensures the 14 GB model loads only once per process, not once per image.
# ─────────────────────────────────────────────────────────────────────────────
_llava_processor = None
_llava_model     = None


def _load_llava() -> tuple:
    """
    Load LLaVA-1.5-7b from HuggingFace. Cache it after the first load.

    device_map="auto" means:
      GPU available  → model goes on GPU automatically
      Multiple GPUs  → model is distributed across them
      No GPU         → falls back to CPU (works but slow)

    torch_dtype=torch.float16 cuts memory from ~28 GB to ~14 GB.
    Required to run on a single 16-24 GB GPU.
    """
    global _llava_processor, _llava_model

    if _llava_processor is None or _llava_model is None:
        logger.info(
            "[VLMAgent] Loading LLaVA-1.5-7b (%s) "
            "— first run downloads ~14 GB ...", LLAVA_MODEL_ID
        )
        _llava_processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID)
        _llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            device_map="auto",          # never hardcode cuda or cpu
            torch_dtype=torch.float16,  # half precision = half the VRAM
        )
        _llava_model.eval()  # evaluation mode: disables dropout, deterministic output
        logger.info("[VLMAgent] LLaVA-1.5-7b loaded.")

    return _llava_processor, _llava_model


# ─────────────────────────────────────────────────────────────────────────────
# PATH A: LLaVA FORENSIC INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _run_llava(image_path: str) -> tuple:
    """
    Run LLaVA-1.5-7b on the original full image with the forensic prompt.

    Returns:
        caption    (str)   — LLaVA's full natural language response
        verdict    (str)   — "REAL" | "FAKE" | "UNCERTAIN"
        confidence (float) — [0,1] score derived from the language content
    """
    processor, model = _load_llava()

    # Open the image as RGB (LLaVA needs PIL RGB input)
    image = Image.open(image_path).convert("RGB")

    # Processor converts image + text into PyTorch tensors the model can read
    inputs = processor(
        text=FORENSIC_PROMPT,
        images=image,
        return_tensors="pt",
    )

    # Move tensors to the same device as the model's first layer.
    # With device_map="auto", layers may be on GPU0, GPU1, or CPU — we
    # cannot hardcode "cuda:0". next(model.parameters()).device is always correct.
    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    # Generate the response.
    # torch.no_grad() disables gradient tracking — only needed during training.
    # Disabling it saves memory and speeds up inference.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,   # enough for a detailed forensic paragraph
            do_sample=False,      # greedy = deterministic, same image → same answer
        )

    # Decode token IDs back to English text
    full_text = processor.batch_decode(
        output_ids, skip_special_tokens=True
    )[0].strip()

    # LLaVA includes the full prompt in the output — extract only the answer
    caption = (
        full_text.split("ASSISTANT:")[-1].strip()
        if "ASSISTANT:" in full_text
        else full_text
    )

    logger.info("[VLMAgent] LLaVA response: %s", caption)

    verdict, confidence = _parse_verdict(caption)
    return caption, verdict, confidence


def _parse_verdict(caption: str) -> tuple:
    """
    Convert LLaVA's English paragraph into a structured verdict + confidence.

    Logic:
      1. Scan for explicit verdict words: "fake", "real", "authentic" etc.
      2. Count all supporting keywords on each side.
      3. Confidence = 0.60 base + 0.05 per extra keyword, capped at 0.98.
      4. Return UNCERTAIN when neither side is clearly dominant.

    We never return confidence=1.0 from text parsing alone — language
    is inherently ambiguous and should not claim absolute certainty.
    """
    text = caption.lower()

    fake_keywords = [
        "fake", "ai-generated", "artificial", "generated", "deepfake",
        "synthetic", "manipulated", "suspicious", "inconsistent",
        "unnatural", "anomalous", "artefact", "artifact",
        "blending", "composite", "unrealistic", "distorted",
    ]
    real_keywords = [
        "real", "authentic", "genuine", "natural", "photograph",
        "normal", "consistent", "no anomal", "no suspicious",
    ]

    fake_count = sum(1 for kw in fake_keywords if kw in text)
    real_count = sum(1 for kw in real_keywords if kw in text)

    explicit_fake = any(kw in text for kw in ["fake", "ai-generated", "deepfake"])
    explicit_real = any(kw in text for kw in ["real", "authentic", "genuine"])

    if explicit_fake and not explicit_real:
        return "FAKE", round(min(0.60 + fake_count * 0.05, 0.98), 3)

    if explicit_real and not explicit_fake:
        return "REAL", round(min(0.60 + real_count * 0.05, 0.98), 3)

    if fake_count > real_count + 1:
        return "FAKE", round(min(0.50 + (fake_count - real_count) * 0.04, 0.85), 3)

    if real_count > fake_count + 1:
        return "REAL", round(min(0.50 + (real_count - fake_count) * 0.04, 0.85), 3)

    return "UNCERTAIN", 0.50


# ─────────────────────────────────────────────────────────────────────────────
# PATH B: GRAD-CAM PLACEHOLDER
# TEAMMATE MERGE POINT
#
# Replace this entire function when Agent 3 teammate delivers EfficientNet-B4.
# Keep the return signature identical:
#   (cam_map, saliency_score, gan_probability)
#   cam_map:         np.ndarray shape (H, W) float32, values [0, 1]
#   saliency_score:  float [0, 1]
#   gan_probability: float [0, 1]
#
# The rest of the file — zone classification, heatmap save, anomaly score —
# needs NO changes. Only this function gets replaced.
# ─────────────────────────────────────────────────────────────────────────────

def _run_gradcam_placeholder(face_crop_path: str) -> tuple:
    """
    Placeholder for Grad-CAM. Returns neutral values.
    Does NOT affect LLaVA results — both paths are independent.

    When Grad-CAM is a placeholder, the anomaly_score formula weights
    LLaVA at 30% and gan_probability at 50% (neutral 0.5 = 0.25 contribution).
    LLaVA still provides meaningful signal via the remaining 30%.
    """
    logger.info(
        "[VLMAgent] Grad-CAM placeholder active. "
        "Awaiting EfficientNet-B4 from Agent 3 teammate."
    )
    neutral = np.full((224, 224), 0.5, dtype=np.float32)
    return neutral, 0.5, 0.5


# ─────────────────────────────────────────────────────────────────────────────
# ZONE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _classify_regions(cam_map: np.ndarray) -> tuple:
    """
    Classify the nine face regions into RED / BLUE / VIOLET activation zones.

    For each region:
      1. Convert fractional coordinates to pixel coordinates on the cam_map
      2. Slice out the rectangle: cam_map[y0:y1, x0:x1]
      3. Compute mean activation in that rectangle
      4. Compare against RED_THRESHOLD (0.8) and BLUE_THRESHOLD (0.5)

    zone_gan_probability = mean activation over the central face zone
    (eyes + nose + mouth) — the most forensically significant area.
    """
    h, w = cam_map.shape[:2]
    high_regions, medium_regions, low_regions = [], [], []
    central_activations = []
    central_zone = {"eyes", "nose", "mouth"}

    for region_name, (x0f, y0f, x1f, y1f) in FACE_REGIONS.items():
        x0, y0 = max(0, int(x0f * w)), max(0, int(y0f * h))
        x1, y1 = min(w, int(x1f * w)), min(h, int(y1f * h))

        if x1 <= x0 or y1 <= y0:
            low_regions.append(region_name)
            continue

        activation = float(np.mean(cam_map[y0:y1, x0:x1]))

        if activation >= RED_THRESHOLD:
            high_regions.append(region_name)
        elif activation >= BLUE_THRESHOLD:
            medium_regions.append(region_name)
        else:
            low_regions.append(region_name)

        if region_name in central_zone:
            central_activations.append(activation)

    zone_gan_prob = float(np.mean(central_activations)) if central_activations else 0.5
    return high_regions, medium_regions, low_regions, zone_gan_prob


# ─────────────────────────────────────────────────────────────────────────────
# HEATMAP SAVE
# ─────────────────────────────────────────────────────────────────────────────

def _save_heatmap(cam_map: np.ndarray, face_crop_path: str, image_path: str) -> str:
    """
    Blend the Grad-CAM heatmap onto the face crop and save as PNG.
    Returns the absolute path to the saved file.
    """
    face_bgr = cv2.imread(face_crop_path)
    if face_bgr is None:
        logger.warning("[VLMAgent] Cannot read face crop — skipping heatmap.")
        return str(TEMP_DIR / "heatmap_unavailable.png")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    crop_h, crop_w = face_rgb.shape[:2]

    # Resize cam_map to face crop dimensions (Grad-CAM output may be smaller)
    cam_resized = cv2.resize(cam_map, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    # Apply jet colormap: 0=blue (VIOLET), 0.5=yellow (BLUE), 1.0=red (RED)
    heatmap_rgb = (cm.get_cmap("jet")(cam_resized)[:, :, :3] * 255).astype(np.uint8)

    # 60% original + 40% colour overlay
    overlay = cv2.addWeighted(face_rgb, 0.60, heatmap_rgb, 0.40, 0)

    output_path = TEMP_DIR / f"heatmap_{int(time.time())}.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(face_rgb);  axes[0].set_title("Face crop");            axes[0].axis("off")
    axes[1].imshow(overlay);   axes[1].set_title("Grad-CAM overlay §5.4"); axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("0=VIOLET  0.5=BLUE  0.8+=RED", fontsize=8)
    cbar.set_ticks([0.0, 0.5, 0.8, 1.0])

    plt.suptitle(f"VLM Heatmap — {Path(image_path).name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("[VLMAgent] Heatmap saved → %s", output_path.resolve())
    return str(output_path.resolve())


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_anomaly_score(
    vlm_verdict: str, vlm_confidence: float,
    saliency_score: float, gan_probability: float, high_regions: list,
) -> float:
    """
    Weighted combination of all VLM signals into a single [0,1] score.

    Weights:
      50% gan_probability (Grad-CAM model — most reliable when real)
      30% LLaVA confidence (semantic forensic judgment)
      10% saliency_score (overall face attention)
      10% high_region_ratio (proportion of face in RED zone)

    With Grad-CAM placeholder (all 0.5):
      gan_probability=0.5 contributes a neutral 0.25 to the total.
      LLaVA still provides meaningful signal through its 30% weight.
    """
    if vlm_verdict == "FAKE":
        llava_c = vlm_confidence
    elif vlm_verdict == "REAL":
        llava_c = 1.0 - vlm_confidence
    else:
        llava_c = 0.50

    high_ratio = len(high_regions) / len(FACE_REGIONS) if FACE_REGIONS else 0.0

    score = (
        0.50 * gan_probability +
        0.30 * llava_c         +
        0.10 * saliency_score  +
        0.10 * high_ratio
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class VLMAgent:
    """
    Agent 5 — §5.4 VLM Explainability Heat-Map Analysis.
    Stateless — safe to call multiple times with different ctx dicts.
    """

    def run(self, ctx: dict) -> dict:
        """
        Main pipeline entry point.

        Args:
            ctx (dict): pipeline context. Must contain:
                        image_path, face_bbox, face_crop_path

        Returns:
            dict: all 10 VLM_KEYS populated, validated against contracts.py
        """
        image_path     = ctx.get("image_path", "")
        face_bbox      = ctx.get("face_bbox", [0, 0, 100, 100])
        face_crop_path = ctx.get("face_crop_path", "")

        if not image_path or not Path(image_path).exists():
            return self._fallback("image_path missing or not found")

        if not face_crop_path or not Path(face_crop_path).exists():
            return self._fallback("face_crop_path missing or not found")

        # ── Path A: LLaVA ─────────────────────────────────────────────────
        logger.info("[VLMAgent] Running LLaVA-1.5-7b ...")
        try:
            vlm_caption, vlm_verdict, vlm_confidence = _run_llava(image_path)
        except Exception as exc:
            logger.error("[VLMAgent] LLaVA error: %s", exc, exc_info=True)
            vlm_caption, vlm_verdict, vlm_confidence = f"[ERROR] {exc}", "UNCERTAIN", 0.50

        # ── Path B: Grad-CAM placeholder ──────────────────────────────────
        logger.info("[VLMAgent] Running Grad-CAM (placeholder) ...")
        try:
            cam_map, saliency_score, gan_probability = _run_gradcam_placeholder(
                face_crop_path
            )
        except Exception as exc:
            logger.error("[VLMAgent] Grad-CAM error: %s", exc)
            cam_map, saliency_score, gan_probability = (
                np.full((224, 224), 0.5, dtype=np.float32), 0.50, 0.50
            )

        # ── Zone classification ───────────────────────────────────────────
        try:
            high_r, mid_r, low_r, zone_gan = _classify_regions(cam_map)
        except Exception as exc:
            logger.error("[VLMAgent] Zone classification error: %s", exc)
            high_r, mid_r, low_r, zone_gan = [], [], list(FACE_REGIONS.keys()), 0.50

        # ── Heatmap PNG ───────────────────────────────────────────────────
        try:
            heatmap_path = _save_heatmap(cam_map, face_crop_path, image_path)
        except Exception as exc:
            logger.error("[VLMAgent] Heatmap error: %s", exc)
            heatmap_path = str(TEMP_DIR / "heatmap_unavailable.png")

        # ── Anomaly score ─────────────────────────────────────────────────
        anomaly_score = _compute_anomaly_score(
            vlm_verdict, vlm_confidence, saliency_score, gan_probability, high_r
        )

        # ── Build and validate output ─────────────────────────────────────
        output = {
            "heatmap_path":              heatmap_path,
            "vlm_caption":               vlm_caption,
            "vlm_verdict":               vlm_verdict,
            "vlm_confidence":            round(vlm_confidence, 4),
            "saliency_score":            round(saliency_score, 4),
            "high_activation_regions":   high_r,
            "medium_activation_regions": mid_r,
            "low_activation_regions":    low_r,
            "zone_gan_probability":      round(zone_gan, 4),
            "anomaly_score":             anomaly_score,
        }
        validate(output, VLM_KEYS, "VLMAgent")

        logger.info(
            "[VLMAgent] Done — verdict=%s confidence=%.3f anomaly=%.4f",
            vlm_verdict, vlm_confidence, anomaly_score,
        )
        return output

    def _fallback(self, reason: str = "unknown") -> dict:
        """Return neutral values when a fatal error prevents analysis."""
        logger.warning("[VLMAgent] Fallback triggered: %s", reason)
        output = {
            "heatmap_path":              str(TEMP_DIR / "heatmap_fallback.png"),
            "vlm_caption":               f"[VLM FALLBACK] {reason}",
            "vlm_verdict":               "UNCERTAIN",
            "vlm_confidence":            0.50,
            "saliency_score":            0.50,
            "high_activation_regions":   [],
            "medium_activation_regions": [],
            "low_activation_regions":    list(FACE_REGIONS.keys()),
            "zone_gan_probability":      0.50,
            "anomaly_score":             0.50,
        }
        validate(output, VLM_KEYS, "VLMAgent")
        return output