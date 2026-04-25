# agents/frequency_agent.py — MFAD Multi-Factor AI Deepfake Detection System
# =============================================================================
# FreqNet-based Frequency-Domain Deepfake Detection Agent
#
# Replaces the legacy FFT+SVM pipeline (Frank 2020) with the FreqNet deep-
# learning model (frequency-domain ResNet variant).
#
# CONTRACT (§5.2 GAN Artefact & Frequency-Domain Analysis):
#   Input:
#       input["path"]           — str: absolute path to original image
#       input["face_crop_path"] — str (optional): face-crop from preprocessing
#                                 agent. If non-empty, used instead of raw image.
#                                 Handles FF++ and similar cropped datasets.
#   Output:
#       freqnet_fake_probability — float [0, 1]: P(fake) from FreqNet sigmoid.
#                                  1.0 = almost certainly deepfake.
#       anomaly_score            — float [0, 1]: same value; used by Bayesian
#                                  fusion (§6 weight: 0.25).
#
# Model:  FreqNet (frequency-domain ResNet) — 4-class trained, binary used here
# Weights: models/4-classes-freqnet-v2.pth
#
# Preprocessing (matches training pipeline from FreqNet-DeepfakeDetection repo):
#   Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)
#
# Device: auto-selects CUDA if available, otherwise CPU.
# =============================================================================

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_AGENT_DIR   = Path(__file__).parent
_PROJECT_DIR = _AGENT_DIR.parent
_MODEL_PATH  = _PROJECT_DIR / "models" / "4-classes-freqnet-v2.pth"
_FREQNET_PKG = _PROJECT_DIR / "models" / "freqnet_networks"

# Add freqnet_networks to path for import
if str(_FREQNET_PKG) not in sys.path:
    sys.path.insert(0, str(_FREQNET_PKG))


# ── Preprocessing transform (identical to FreqNet training pipeline) ──────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ── Lazy singleton model cache ─────────────────────────────────────────────────
_model  = None
_device = None


def _get_model():
    """Load FreqNet model once, cache for subsequent calls."""
    global _model, _device

    if _model is not None:
        return _model, _device

    from freqnet import freqnet  # imported from models/freqnet_networks/

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[FrequencyAgent] Loading FreqNet on device: %s", _device)

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"[FrequencyAgent] Model weights not found: {_MODEL_PATH}\n"
            f"  Expected: MFAD/models/4-classes-freqnet-v2.pth"
        )

    model = freqnet()
    state_dict = torch.load(str(_MODEL_PATH), map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(_device)
    model.eval()

    _model = model
    log.info("[FrequencyAgent] FreqNet loaded successfully.")
    return _model, _device


# ── Null output: returned on load / inference failure ────────────────────────
_NULL_OUTPUT = {
    "freqnet_fake_probability": 0.5,
    "anomaly_score":            0.5,
    "error":                    "",
}


# ── Main entry point ──────────────────────────────────────────────────────────

def run(input: dict) -> dict:
    """
    Run FreqNet inference on a single image.

    Args:
        input: dict with keys:
            "path"           — str: path to original image (always required)
            "face_crop_path" — str: path to face crop from preprocessing agent.
                               If non-empty, this is used as the model input
                               (important for FF++ and similar cropped datasets).
                               If empty or absent, the raw image at "path" is used.

    Returns:
        dict with:
            freqnet_fake_probability — float [0, 1]: P(fake)
            anomaly_score            — float [0, 1]: same (for Bayesian fusion)
    """
    # ── Select image path ─────────────────────────────────────────────────────
    face_crop_path = input.get("face_crop_path", "")
    raw_path       = input.get("path", "")

    if face_crop_path and os.path.exists(face_crop_path):
        image_path = face_crop_path
        log.info("[FrequencyAgent] Using face crop: %s", image_path)
    elif raw_path and os.path.exists(raw_path):
        image_path = raw_path
        log.info("[FrequencyAgent] Using raw image: %s", image_path)
    else:
        log.warning("[FrequencyAgent] No valid image path provided.")
        return {**_NULL_OUTPUT, "error": "no_valid_image_path"}

    # ── Load and preprocess image ─────────────────────────────────────────────
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)   # [1, 3, 224, 224]
    except Exception as exc:
        log.error("[FrequencyAgent] Image load failed: %s", exc)
        return {**_NULL_OUTPUT, "error": f"image_load_failed: {exc}"}

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        model, device = _get_model()
        tensor = tensor.to(device)

        with torch.no_grad():
            logit = model(tensor)                  # [1, 1]
            prob  = torch.sigmoid(logit).item()    # scalar ∈ [0, 1]

    except Exception as exc:
        log.error("[FrequencyAgent] Inference failed: %s", exc)
        return {**_NULL_OUTPUT, "error": f"inference_failed: {exc}"}

    prob = float(prob)
    log.info("[FrequencyAgent] freqnet_fake_probability=%.4f", prob)

    return {
        "freqnet_fake_probability": prob,
        "anomaly_score":            prob,
    }


# ── Output validator (used by batch_run.py and contracts.py) ──────────────────

def validate_output(output: dict) -> bool:
    """
    Returns True if output satisfies the FREQUENCY_KEYS contract.
    Checks presence and value range of required keys.
    """
    required = ["freqnet_fake_probability", "anomaly_score"]
    for key in required:
        if key not in output:
            log.warning("[FrequencyAgent] validate_output: missing key '%s'", key)
            return False
        val = output[key]
        if not isinstance(val, (int, float)):
            log.warning("[FrequencyAgent] validate_output: key '%s' must be numeric", key)
            return False
        if not (0.0 <= float(val) <= 1.0):
            log.warning("[FrequencyAgent] validate_output: key '%s'=%s out of [0,1]", key, val)
            return False
    return True


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="FreqNet frequency agent test")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--face-crop", default="", help="Optional face crop path")
    args = parser.parse_args()

    result = run({"path": args.image, "face_crop_path": args.face_crop})
    print(result)
    print("VALID:", validate_output(result))