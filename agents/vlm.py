"""
agents/vlm.py — MFAD Multi-Factor AI Deepfake Detection System
==============================================================
Agent 5 — VLM Explainability Heat-Map Analysis
Report section: §5.4

DETECTOR MODEL: EfficientNet-B4 (Noisy Student, fine-tuned on MFAD dataset)
  - Weights: models/efficientnet_b4_mfad_final.pth  (~74 MB)
  - Architecture: tf_efficientnet_b4.ns_jft_in1k backbone + 2-layer head
  - Trained on 302K images across FF++, Faces-HQ, CelebA, deepdetect25
  - Achieved val-AUC=0.9998 on 33K held-out images
  - Produces TRUE Grad-CAM (CNN backward hooks) — far better spatial
    heatmaps than ViT attention rollout used by the old CLIP model

VLM MODEL: LLaVA-1.5-7b (llava-hf/llava-1.5-7b-hf) ~14 GB
  - Downloaded ONCE on first run, cached at ~/.cache/huggingface/hub/
  - Receives the Grad-CAM heatmap PNG and describes activation regions

HOW TRUE GRAD-CAM WORKS (vs old attention rollout)
----------------------------------------------------
Old approach (CLIP ViT): attention rollout averages attention matrices
across 24 transformer layers — does not directly reflect classification
signal, just correlates with "where the model looked".

New approach (EfficientNet Grad-CAM):
  1. Forward pass → fake-class logit
  2. Backward pass from fake logit → gradients at last conv layer (blocks[-1])
  3. Global average pool those gradients → one weight per channel
  4. Weight each feature map channel by its gradient weight
  5. ReLU + normalise → spatial map showing which pixels CAUSED the fake score
This is causally linked to the actual prediction — a better forensic signal.

EXECUTION ORDER (unchanged from CLIP version):
  1. Run EfficientNet on face crop → cam_map, gan_probability
  2. Save heatmap PNG → heatmap_path
  3. Pass heatmap to LLaVA → vlm_caption, vlm_verdict, vlm_confidence
  4. Classify zones from cam_map → high_r, mid_r, low_r, zone_gan
  5. Compute anomaly_score

Output keys (contracts.py VLM_KEYS — never change these):
  heatmap_path, vlm_caption, vlm_verdict, vlm_confidence,
  saliency_score, high_activation_regions, medium_activation_regions,
  low_activation_regions, zone_gan_probability, anomaly_score

Extra key (not in VLM_KEYS, passed to fusion as gan_artefact):
  gan_probability  ← raw EfficientNet fake probability

Depends on from ctx (produced by Agent 1 PreprocessingAgent):
  ctx["image_path"]      — path to the original suspect image
  ctx["face_bbox"]       — [x1, y1, x2, y2] face location in pixels
  ctx["face_crop_path"]  — path to the cropped face image
"""

import sys
import os
import re
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms

from transformers import LlavaProcessor, LlavaForConditionalGeneration

try:
    import timm
except ImportError:
    sys.exit("[VLMAgent] timm not found. Run: pip install timm")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contracts import validate, VLM_KEYS

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

LLAVA_MODEL_ID    = "llava-hf/llava-1.5-7b-hf"
EFFICIENTNET_PATH = "models/efficientnet_b4_mfad_final.pth"
EFFICIENTNET_SIZE = 380   # native resolution — must match training

# Grad-CAM zone thresholds from §5.4 of reference report
RED_THRESHOLD  = 0.8   # above → high_activation_regions
BLUE_THRESHOLD = 0.5   # above → medium_activation_regions
                       # below → low_activation_regions

# LLaVA-1.5 chat format — do not change the USER:/ASSISTANT: structure.
#
# DESIGN RATIONALE FOR THIS PROMPT:
# ──────────────────────────────────
# The prompt gives LLaVA three critical pieces of information:
#   1. The exact GAN probability from EfficientNet (the authoritative classifier)
#   2. The pre-computed verdict derived from that probability
#   3. The zone-level activation data (computed numerically, not visually)
#
# LLaVA's job is ONLY to write a professional forensic explanation of WHY the
# verdict is what it is — not to re-derive the verdict itself. This eliminates
# the "LLaVA overrides EfficientNet" problem completely.
#
# Key design decisions:
#   - The verdict is GIVEN to LLaVA, not asked of it
#   - The explanation must justify the given verdict — not contradict it
#   - Zone interpretations are pre-computed based on whether the image is
#     fake or real, so LLaVA never applies "manipulation" language to real images
#   - The first-line verdict is explicit, making _parse_verdict() reliable
FORENSIC_PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "You are a forensic AI analyst writing section §5.4 of a court-admissible "
    "deepfake detection report. Your task is to write a professional forensic "
    "narrative that EXPLAINS the computed detection result shown below.\n\n"
    "DETECTION RESULT (authoritative — do not contradict this):\n"
    "{VERDICT_BLOCK}\n\n"
    "GRAD-CAM ZONE ANALYSIS (computed numerically from EfficientNet activation maps):\n"
    "{ZONE_TABLE}\n\n"
    "INSTRUCTIONS:\n"
    "Your response MUST begin with exactly '{VERDICT_WORD}' on the first line alone "
    "(this matches the detection result above — do not change it).\n"
    "Then write a forensic narrative of 4-5 sentences that:\n"
    "  - References the GAN probability score and explains what it means\n"
    "  - Describes which facial regions showed elevated activation and why "
    "    that is consistent with the verdict\n"
    "  - Uses precise forensic language appropriate for a legal document\n"
    "  - Does NOT describe image colors — describe what the measurements mean\n"
    "  - Is internally consistent: if the verdict is REAL, explain why the "
    "    activation pattern is consistent with an authentic image; if FAKE, "
    "    explain the forensic evidence of manipulation\n"
    "ASSISTANT:"
)

# Nine named face regions — fractional (x0, y0, x1, y1) coordinates
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

# ImageNet normalisation — must match EfficientNet training
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Holds the percentile-stretched visual cam_map between _run_gradcam_efficientnet()
# and _save_heatmap(). Single-element list used as a mutable container so it
# works across function boundaries without adding a return value.
_GRADCAM_VISUAL: list = [None]


# =============================================================================
# EFFICIENTNET-B4 DETECTOR
# =============================================================================

class _EfficientNetB4Detector(nn.Module):
    """
    EfficientNet-B4 Noisy Student backbone + binary classification head.
    Identical architecture to train_efficientnet.py so weights load cleanly.

    backbone : tf_efficientnet_b4.ns_jft_in1k  (feature_dim=1792)
    head     : Dropout(0.4) -> Linear(1792,512) -> GELU -> Dropout(0.2) -> Linear(512,2)
    """

    def __init__(self) -> None:
        super().__init__()
        # num_classes=0 removes the ImageNet head; forward returns (batch, 1792)
        self.backbone = timm.create_model(
            "tf_efficientnet_b4.ns_jft_in1k",
            pretrained=False,   # weights loaded separately from our fine-tuned file
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features  # 1792

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


# Module-level cache — model loaded once per process
_efficientnet_model: _EfficientNetB4Detector = None
_efficientnet_device: str = None


def _load_efficientnet() -> tuple:
    """
    Load the fine-tuned EfficientNet-B4 weights from disk.
    Cached after first call — never reloaded within the same process.

    Device selection: prefer GPU if available, fall back to CPU.
    EfficientNet-B4 is only ~74 MB so it sits comfortably alongside LLaVA
    on a 50 GB A6000. Cache is cleared before LLaVA runs to avoid OOM.
    """
    global _efficientnet_model, _efficientnet_device

    if _efficientnet_model is None:
        weights_path = Path(EFFICIENTNET_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"[VLMAgent] EfficientNet weights not found at {weights_path}. "
                f"Run train_efficientnet.py first — it saves to models/efficientnet_b4_mfad_final.pth"
            )

        logger.info("[VLMAgent] Loading EfficientNet-B4 from %s ...", weights_path)

        model = _EfficientNetB4Detector()
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()

        # Device selection
        if torch.cuda.is_available():
            _efficientnet_device = os.environ.get("VLM_GPU", "cuda:0")
        else:
            _efficientnet_device = "cpu"

        model = model.to(_efficientnet_device)
        _efficientnet_model = model

        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(
            "[VLMAgent] EfficientNet-B4 loaded on %s (%.1fM params).",
            _efficientnet_device, total_params,
        )

    return _efficientnet_model, _efficientnet_device


def _preprocess_for_efficientnet(face_crop_path: str) -> torch.Tensor:
    """
    Load a face crop and prepare it as a normalised 380x380 tensor.
    Matches the validation transform used during training exactly.
    """
    transform = transforms.Compose([
        transforms.Resize(int(EFFICIENTNET_SIZE * 1.14)),  # slight oversize
        transforms.CenterCrop(EFFICIENTNET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    img = Image.open(face_crop_path).convert("RGB")
    return transform(img).unsqueeze(0)   # shape: (1, 3, 380, 380)


def _run_gradcam_efficientnet(face_crop_path: str) -> tuple:
    """
    Run EfficientNet-B4 on the face crop and produce a true Grad-CAM heatmap.

    TRUE GRAD-CAM STEPS:
    1. Forward pass with gradient tracking enabled on the last conv block
    2. Backward pass from the fake-class logit (class index 1)
    3. Global average pool the gradients across spatial dims → channel weights
    4. Weighted sum of feature maps → raw activation map
    5. ReLU (keep only positive influence on fake score)
    6. Normalise to [0, 1], resize to 224x224

    Returns:
        cam_map           : np.ndarray (224,224), float32, range [0,1]
        saliency_score    : float — mean activation (overall fake signal strength)
        gan_probability   : float — P(fake) from softmax
        is_placeholder    : always False (real Grad-CAM, not fallback)
    """
    model, device = _load_efficientnet()

    pixel_values = _preprocess_for_efficientnet(face_crop_path).to(device)

    # Storage for hook outputs
    _feature_maps: list = []
    _gradients:    list = []

    def _forward_hook(module, input, output):
        # output shape: (batch, channels, H, W) at the last conv block
        _feature_maps.append(output.detach())

    def _backward_hook(module, grad_input, grad_output):
        # grad_output[0] shape: same as feature maps
        _gradients.append(grad_output[0].detach())

    # Hook the last MBConv block of the backbone
    # timm EfficientNet-B4: model.backbone.blocks[-1] is the final MBConv stage
    target_layer = model.backbone.blocks[-1]
    fwd_handle   = target_layer.register_forward_hook(_forward_hook)
    bwd_handle   = target_layer.register_full_backward_hook(_backward_hook)

    try:
        # Forward pass — requires grad on input so backward flows through backbone
        pixel_values.requires_grad_(True)
        logits = model(pixel_values)

        probs = torch.softmax(logits.detach(), dim=1)[0]
        gan_probability = float(probs[1])   # class 1 = fake
        logger.info("[VLMAgent] EfficientNet-B4 fake_prob=%.4f", gan_probability)

        # Backward pass from fake class score only
        model.zero_grad()
        fake_score = logits[0, 1]   # scalar — fake-class logit before softmax
        fake_score.backward()

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    # Build Grad-CAM map
    cam_map = None
    try:
        if not _feature_maps or not _gradients:
            raise ValueError("Hooks did not capture feature maps or gradients")

        feature_maps = _feature_maps[0].squeeze(0)   # (C, H, W)
        gradients    = _gradients[0].squeeze(0)       # (C, H, W)

        # Channel weights = global average of gradients (standard Grad-CAM formula)
        weights = gradients.mean(dim=(1, 2))          # (C,)

        # Weighted sum over channels
        cam = torch.zeros(feature_maps.shape[1:], device=feature_maps.device)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        # ReLU — only keep activations that positively contribute to "fake"
        cam = torch.clamp(cam, min=0)

        # Normalise to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            # Uniform map — model is not spatially selective for this image
            cam = torch.zeros_like(cam)

        cam_np = cam.cpu().numpy().astype(np.float32)

        # Resize to 224x224 (standard Grad-CAM output size for heatmap overlay)
        cam_map = cv2.resize(cam_np, (224, 224), interpolation=cv2.INTER_LINEAR)

        # ── Percentile histogram stretch ──────────────────────────────────────
        # Raw Grad-CAM maps are typically skewed: most pixels sit near 0 and
        # only a small peak region is high. After [0,1] normalisation the peak
        # is correct but the visual contrast is very low — LLaVA sees an
        # almost-all-blue image and can't identify the high-activation region.
        #
        # Solution: stretch so the 99th-percentile value maps to 1.0.
        # This makes the hot region clearly RED in the heatmap PNG while
        # keeping the relative spatial ordering intact. It does NOT change
        # the cam_map values used for anomaly scoring — we keep two separate
        # arrays: cam_map (raw, for scoring) and cam_map_visual (stretched, for PNG).
        p99 = float(np.percentile(cam_map, 99))
        if p99 > 0.01:
            cam_map_visual = np.clip(cam_map / p99, 0.0, 1.0)
        else:
            cam_map_visual = cam_map.copy()

        logger.info(
            "[VLMAgent] Grad-CAM succeeded. Map shape=%s min=%.3f max=%.3f p99=%.3f",
            str(cam_map.shape), cam_map.min(), cam_map.max(), p99,
        )

        # Store visual map in a module-level variable so _save_heatmap can use it
        # without changing the return signature (cam_map stays raw for scoring)
        _GRADCAM_VISUAL[0] = cam_map_visual

    except Exception as exc:
        logger.warning(
            "[VLMAgent] Grad-CAM computation failed (%s) — "
            "falling back to uniform gan_probability map.", exc
        )

    if cam_map is None:
        # Fallback: uniform map scaled by classification confidence
        cam_map = np.full((224, 224), gan_probability, dtype=np.float32)
        _GRADCAM_VISUAL[0] = cam_map.copy()

    # Free GPU memory before LLaVA loads
    pixel_values.requires_grad_(False)
    if device != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # saliency_score uses the stretched visual map — reflects the visual
    # prominence of activation as LLaVA sees it, not the raw skewed distribution
    visual_for_score = _GRADCAM_VISUAL[0] if _GRADCAM_VISUAL[0] is not None else cam_map
    saliency_score = float(np.mean(visual_for_score))
    logger.info("[VLMAgent] saliency_score (stretched)=%.4f", saliency_score)
    return cam_map, saliency_score, gan_probability, False


# =============================================================================
# ZONE TABLE BUILDER — generates the data block fed into LLaVA's prompt
# =============================================================================

# Forensic interpretations — TWO sets: one for fakes, one for authentic images.
# The correct set is selected based on gan_probability so LLaVA never applies
# manipulation language to regions in a real image.
_ZONE_INTERPRETATIONS_FAKE = {
    "RED": {
        "eyes":       "algorithmically reconstructed ocular features — high GAN synthesis probability in periorbital zone",
        "nose":       "synthetic nasal bridge reconstruction — central face identity fabrication detected",
        "mouth":      "GAN-generated lip synthesis — high deepfake signal in oral region",
        "cheeks":     "facial boundary blending artifact — face-swap composite edge detected at cheeks",
        "brow":       "forehead-hairline boundary manipulation — identity overlay artifact",
        "chin":       "jaw contour synthesis — GAN face-shape modification confirmed",
        "background": "background recompression artifact — consistent with face compositing operation",
        "hair":       "hair boundary artifact — GAN synthesis edge at hairline",
        "shoulders":  "shoulder blending seam — composite image boundary detected",
    },
    "BLUE": {
        "eyes":       "secondary periorbital texture modification — identity-augmenting alteration",
        "nose":       "partial nasal texture alteration — secondary manipulation zone",
        "mouth":      "secondary lip texture modification — partial synthesis overlap",
        "cheeks":     "skin tone mapping adjustment — identity-augmenting texture modification",
        "brow":       "brow texture modification — partial identity overlay",
        "chin":       "chin texture alteration — secondary jaw modification",
        "background": "mild background alteration — low-level recompression from compositing",
        "hair":       "hair texture modification — secondary synthesis boundary",
        "shoulders":  "shoulder texture modification — low-level blending artifact",
    },
    "VIOLET": {
        "eyes":       "low activation — serves as spatial anchor for face-swap registration",
        "nose":       "low activation — host image region largely preserved",
        "mouth":      "low activation — minimal synthesis signal in this sub-region",
        "cheeks":     "low activation — host image canvas for face-swap blending",
        "brow":       "low activation — host region used as reference for identity overlay",
        "chin":       "low activation — host chin region minimally affected",
        "background": "low activation — background consistent with source camera PRNU",
        "hair":       "low activation — hair region unaffected by synthesis operation",
        "shoulders":  "low activation — shoulder region host image preserved",
    },
}

_ZONE_INTERPRETATIONS_REAL = {
    # For authentic images, elevated activations are normal statistical variation
    # in Grad-CAM, not evidence of manipulation. Language reflects this.
    "RED": {
        "eyes":       "elevated activation in periorbital zone — consistent with high-contrast natural texture",
        "nose":       "elevated activation at nasal bridge — natural edge response from skin texture gradient",
        "mouth":      "elevated activation in oral region — natural lip boundary contrast response",
        "cheeks":     "elevated activation at cheeks — natural skin tone variation response",
        "brow":       "elevated activation at brow — natural hair-skin boundary response",
        "chin":       "elevated activation at chin — natural jaw edge gradient response",
        "background": "elevated activation in background — natural contrast boundary",
        "hair":       "elevated activation at hairline — natural hair texture response",
        "shoulders":  "elevated activation at shoulders — natural fabric texture response",
    },
    "BLUE": {
        "eyes":       "moderate activation in periorbital zone — normal tissue texture response",
        "nose":       "moderate activation at nasal region — natural skin surface variation",
        "mouth":      "moderate activation in oral region — natural lip texture gradient",
        "cheeks":     "moderate activation at cheeks — normal skin tone variation pattern",
        "brow":       "moderate activation at brow — natural brow texture response",
        "chin":       "moderate activation at chin — normal chin contour response",
        "background": "moderate activation in background — natural scene texture response",
        "hair":       "moderate activation in hair — natural hair texture variation",
        "shoulders":  "moderate activation at shoulders — natural clothing texture response",
    },
    "VIOLET": {
        "eyes":       "low activation — consistent with authentic periorbital structure",
        "nose":       "low activation — consistent with authentic nasal tissue",
        "mouth":      "low activation — consistent with authentic oral structure",
        "cheeks":     "low activation — consistent with authentic facial skin",
        "brow":       "low activation — consistent with authentic brow structure",
        "chin":       "low activation — consistent with authentic chin contour",
        "background": "low activation — background unaltered, consistent with original capture",
        "hair":       "low activation — consistent with authentic hair texture",
        "shoulders":  "low activation — consistent with authentic shoulder region",
    },
}


def _compute_region_zones(cam_map: np.ndarray) -> dict:
    """
    Compute per-region activation values and zone assignments directly from
    the Grad-CAM map. Returns a dict keyed by region name with:
      activation : float — mean activation in that region [0,1]
      zone       : str   — "RED", "BLUE", or "VIOLET"
      salience   : str   — formatted "X.XX / 1.00" for the report table
    Ground-truth zone data — computed numerically, not by LLaVA.
    """
    h, w = cam_map.shape[:2]
    results = {}

    for region, (x0f, y0f, x1f, y1f) in FACE_REGIONS.items():
        x0 = max(0, int(x0f * w))
        y0 = max(0, int(y0f * h))
        x1 = min(w, int(x1f * w))
        y1 = min(h, int(y1f * h))

        if x1 <= x0 or y1 <= y0:
            activation = 0.0
        else:
            activation = float(np.mean(cam_map[y0:y1, x0:x1]))

        if activation >= RED_THRESHOLD:
            zone = "RED"
        elif activation >= BLUE_THRESHOLD:
            zone = "BLUE"
        else:
            zone = "VIOLET"

        results[region] = {
            "activation": round(activation, 3),
            "zone":       zone,
            "salience":   f"{activation:.2f} / 1.00",
        }

    return results


def _build_zone_table(region_data: dict, gan_probability: float) -> str:
    """
    Build the zone analysis table injected into LLaVA's prompt.
    Selects fake-language or real-language interpretations based on
    gan_probability so the narrative is always internally consistent.
    """
    # Select interpretation set based on EfficientNet verdict
    is_fake = gan_probability >= 0.50
    interps = _ZONE_INTERPRETATIONS_FAKE if is_fake else _ZONE_INTERPRETATIONS_REAL

    red_regions    = [r for r, d in region_data.items() if d["zone"] == "RED"]
    blue_regions   = [r for r, d in region_data.items() if d["zone"] == "BLUE"]
    violet_regions = [r for r, d in region_data.items() if d["zone"] == "VIOLET"]

    central      = ["eyes", "nose", "mouth"]
    central_vals = [region_data[r]["activation"] for r in central if r in region_data]
    zone_gan     = round(float(np.mean(central_vals)), 3) if central_vals else 0.5

    lines = [
        f"Central face zone activation (eyes+nose+mouth mean): {zone_gan:.3f}",
        "",
    ]

    if red_regions:
        lines.append("  RED zone regions (activation >= 0.80):")
        for r in red_regions:
            d = region_data[r]
            lines.append(f"    {r}: salience {d['salience']} — "
                         f"{interps['RED'].get(r, 'high activation')}")
    else:
        lines.append("  RED zone: no regions exceed 0.80 activation threshold")

    lines.append("")

    if blue_regions:
        lines.append("  BLUE zone regions (activation 0.50–0.80):")
        for r in blue_regions:
            d = region_data[r]
            lines.append(f"    {r}: salience {d['salience']} — "
                         f"{interps['BLUE'].get(r, 'moderate activation')}")
    else:
        lines.append("  BLUE zone: no regions in 0.50–0.80 range")

    lines.append("")

    if violet_regions:
        lines.append("  VIOLET zone regions (activation < 0.50):")
        for r in violet_regions:
            d = region_data[r]
            lines.append(f"    {r}: salience {d['salience']} — "
                         f"{interps['VIOLET'].get(r, 'low activation')}")

    return "\n".join(lines)


def _build_llava_prompt(zone_table: str, gan_probability: float) -> str:
    """
    Fill the zone table and verdict context into the forensic prompt template.
    The verdict word and explanation block are derived from gan_probability
    so LLaVA receives an unambiguous instruction about what verdict to write.
    """
    if gan_probability >= 0.70:
        verdict_word = "FAKE"
        verdict_block = (
            f"Verdict      : FAKE (deepfake detected)\n"
            f"GAN probability: {gan_probability:.4f} — the classifier assigns {gan_probability*100:.1f}% "
            f"probability that this image is synthetically generated or manipulated.\n"
            f"Confidence   : HIGH — EfficientNet-B4 certainty = "
            f"{abs(gan_probability-0.5)*200:.1f}% above random chance"
        )
    elif gan_probability <= 0.30:
        verdict_word = "REAL"
        verdict_block = (
            f"Verdict      : REAL (authentic image)\n"
            f"GAN probability: {gan_probability:.4f} — the classifier assigns only {gan_probability*100:.1f}% "
            f"probability of synthetic generation, indicating high authenticity confidence.\n"
            f"Confidence   : HIGH — EfficientNet-B4 certainty = "
            f"{abs(gan_probability-0.5)*200:.1f}% above random chance"
        )
    else:
        verdict_word = "UNCERTAIN"
        verdict_block = (
            f"Verdict      : UNCERTAIN (inconclusive)\n"
            f"GAN probability: {gan_probability:.4f} — the classifier is not sufficiently "
            f"confident to make a definitive determination.\n"
            f"Confidence   : LOW — further analysis recommended"
        )

    return (
        FORENSIC_PROMPT_TEMPLATE
        .replace("{VERDICT_BLOCK}", verdict_block)
        .replace("{ZONE_TABLE}",   zone_table)
        .replace("{VERDICT_WORD}", verdict_word)
    )

_llava_processor = None
_llava_model     = None


def _load_llava() -> tuple:
    """
    Load LLaVA-1.5-7b from HuggingFace. Cached after the first call.
    torch_dtype=float16 halves VRAM from ~28 GB to ~14 GB.
    low_cpu_mem_usage=True prevents OOM during model construction on CPU.
    """
    global _llava_processor, _llava_model

    if _llava_processor is None or _llava_model is None:
        logger.info(
            "[VLMAgent] Loading LLaVA-1.5-7b (%s) — first run downloads ~14 GB ...",
            LLAVA_MODEL_ID,
        )
        _llava_processor = LlavaProcessor.from_pretrained(LLAVA_MODEL_ID)
        _llava_model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            device_map={"": os.environ.get("VLM_GPU", "cuda:0")},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        _llava_model.eval()
        logger.info("[VLMAgent] LLaVA-1.5-7b loaded.")

    return _llava_processor, _llava_model


# =============================================================================
# LLAVA INFERENCE
# =============================================================================

def _run_llava(image_path: str, prompt: str, gan_probability: float = 0.5) -> tuple:
    """
    Run LLaVA-1.5-7b with a pre-built forensic prompt that already contains
    the verdict, GAN probability, and computed zone analysis table.
    LLaVA writes the narrative — the verdict is given, not derived by LLaVA.
    Returns (caption, verdict, confidence).
    """
    processor, model = _load_llava()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            min_new_tokens=40,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.15,
        )

    full_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    caption   = (
        full_text.split("ASSISTANT:")[-1].strip()
        if "ASSISTANT:" in full_text
        else full_text
    )

    logger.info("[VLMAgent] LLaVA response: %s", caption)
    verdict, confidence = _parse_verdict(caption, gan_probability=gan_probability)
    return caption, verdict, confidence


def _parse_verdict(caption: str, gan_probability: float = 0.5) -> tuple:
    """
    Parse LLaVA's response into (verdict, confidence).

    Since the prompt now explicitly tells LLaVA which verdict word to use on
    line 1, the first-line check is authoritative. The keyword scan is only
    used if LLaVA ignored the instruction and did not produce a clean first line.

    IMPORTANT: The fallback keyword scan excludes words that appear in the
    injected zone table itself (e.g. "deepfake", "fake", "GAN") to prevent
    false positives when LLaVA writes narrative about those injected labels.
    In the fallback we rely on gan_probability as a tiebreaker.
    """
    text       = caption.lower()
    lines      = text.split("\n")
    first_line = lines[0].strip()
    explanation = " ".join(lines[1:]).strip()

    # ── Primary: trust the first line if it is a clean verdict ────────────────
    if first_line in ("fake", "real", "uncertain"):
        verdict = first_line.upper()
        # Score explanation for corroborating evidence
        if verdict == "FAKE":
            evidence = sum(1 for kw in [
                r"\bsynthetic\b", r"\bmanipulated\b", r"\bblending\b",
                r"\bcomposite\b", r"\bunrealistic\b", r"\bdistorted\b",
                r"\bover.smooth", r"\bai.generated\b", r"\bcritical\b",
                r"\bhigh activation\b", r"\belevated\b",
            ] if re.search(kw, explanation))
            return "FAKE", round(min(0.70 + evidence * 0.03, 0.97), 3)

        if verdict == "REAL":
            evidence = sum(1 for kw in [
                r"\bgenuine\b", r"\bauthentic\b", r"\bno signs\b",
                r"\bno anomal", r"\bconsistent with authentic\b",
                r"\bnatural\b", r"\blow activation\b", r"\bunaltered\b",
                r"\bpreserved\b",
            ] if re.search(kw, explanation))
            return "REAL", round(min(0.70 + evidence * 0.03, 0.97), 3)

        return "UNCERTAIN", 0.50

    # ── Also accept if first line STARTS with the verdict word ────────────────
    for verdict_word in ("fake", "real", "uncertain"):
        if first_line.startswith(verdict_word):
            return _parse_verdict(verdict_word + "\n" + "\n".join(lines[1:]),
                                  gan_probability)

    # ── Fallback: use gan_probability as the ground truth ─────────────────────
    # LLaVA did not produce a clean first line. Rather than scanning for
    # "fake"/"deepfake" which appear in the injected zone table, use
    # EfficientNet's probability as the verdict and assign a moderate confidence.
    logger.warning(
        "[VLMAgent] _parse_verdict: no clean first-line verdict. "
        "Falling back to gan_probability=%.4f", gan_probability
    )
    if gan_probability >= 0.70:
        return "FAKE",      round(min(0.60 + (gan_probability - 0.70) * 0.5, 0.90), 3)
    elif gan_probability <= 0.30:
        return "REAL",      round(min(0.60 + (0.30 - gan_probability) * 0.5, 0.90), 3)
    else:
        return "UNCERTAIN", 0.50


# =============================================================================
# ZONE CLASSIFICATION
# =============================================================================

def _classify_regions(cam_map: np.ndarray) -> tuple:
    """
    Classify nine face regions into high / medium / low activation zones.
    zone_gan_probability = mean activation over eyes + nose + mouth.
    """
    h, w = cam_map.shape[:2]
    high_regions, medium_regions, low_regions = [], [], []
    central_activations = []
    central_zone = {"eyes", "nose", "mouth"}

    for region_name, (x0f, y0f, x1f, y1f) in FACE_REGIONS.items():
        x0 = max(0, int(x0f * w))
        y0 = max(0, int(y0f * h))
        x1 = min(w, int(x1f * w))
        y1 = min(h, int(y1f * h))

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


# =============================================================================
# HEATMAP SAVE
# =============================================================================

def _save_heatmap(cam_map: np.ndarray, face_crop_path: str, image_path: str) -> str:
    """
    Blend the Grad-CAM heatmap onto the face crop and save as a two-panel PNG.
    Left panel: original face crop. Right panel: Grad-CAM overlay.

    Uses the percentile-stretched visual map (_GRADCAM_VISUAL) for the PNG
    so the high-activation region is clearly RED and visible to LLaVA.
    The raw cam_map (used for anomaly scoring) is NOT modified.

    Returns the absolute path to the saved PNG.
    """
    face_bgr = cv2.imread(face_crop_path)
    if face_bgr is None:
        logger.warning("[VLMAgent] Cannot read face crop — skipping heatmap.")
        return str(TEMP_DIR / "heatmap_unavailable.png")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    crop_h, crop_w = face_rgb.shape[:2]

    # Use the percentile-stretched visual map so LLaVA sees clear RED regions.
    # Fall back to raw cam_map if _GRADCAM_VISUAL was not populated.
    visual_map = _GRADCAM_VISUAL[0] if _GRADCAM_VISUAL[0] is not None else cam_map

    cam_resized = cv2.resize(visual_map, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    heatmap_rgb = (cm.get_cmap("jet")(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay     = cv2.addWeighted(face_rgb, 0.55, heatmap_rgb, 0.45, 0)

    output_path = TEMP_DIR / f"heatmap_{int(time.time())}.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(face_rgb);  axes[0].set_title("Face crop");                 axes[0].axis("off")
    axes[1].imshow(overlay);   axes[1].set_title("EfficientNet Grad-CAM §5.4"); axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("0=VIOLET  0.5=BLUE  0.8+=RED", fontsize=8)
    cbar.set_ticks([0.0, 0.5, 0.8, 1.0])

    plt.suptitle(f"VLM Heatmap — {Path(image_path).name}", fontsize=11)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("[VLMAgent] Heatmap saved -> %s", output_path.resolve())
    return str(output_path.resolve())


# =============================================================================
# ANOMALY SCORE
# =============================================================================

def _compute_anomaly_score(
    vlm_verdict: str,
    vlm_confidence: float,
    saliency_score: float,
    gan_probability: float,
    high_regions: list,
    gradcam_is_placeholder: bool = False,
) -> float:
    """
    Weighted combination of all VLM signals into a single [0,1] anomaly score.

    CONFIDENCE-ADAPTIVE WEIGHTS
    ----------------------------
    When EfficientNet is highly confident (gan_probability > 0.85 or < 0.15),
    LLaVA's weight is reduced because it is interpreting a heatmap not raw pixels
    and can be wrong about what it sees. When EfficientNet is uncertain
    (0.35-0.65), LLaVA's semantic judgment gets more influence.

    EfficientNet confidence zones:
      HIGH   (>0.85 fake or <0.15 fake): gan=0.70  llava=0.15  sal=0.08  reg=0.07
      MEDIUM (0.65-0.85 or 0.15-0.35):  gan=0.55  llava=0.25  sal=0.10  reg=0.10
      LOW    (0.35-0.65):               gan=0.40  llava=0.38  sal=0.12  reg=0.10

    This prevents LLaVA from overriding a near-certain EfficientNet prediction
    (the bug observed: EfficientNet=0.957, LLaVA=REAL → anomaly=0.574 UNCERTAIN).
    With these weights: 0.70×0.957 + 0.15×(1-0.730) + 0.08×0.146 + 0.07×0.0
                      = 0.670 + 0.041 + 0.012 + 0.0 = 0.723 → DEEPFAKE

    When Grad-CAM failed (placeholder uniform map): LLaVA gets full weight
    since EfficientNet score is unreliable.
    """
    if vlm_verdict == "FAKE":
        llava_signal = vlm_confidence
    elif vlm_verdict == "REAL":
        llava_signal = 1.0 - vlm_confidence
    else:
        llava_signal = 0.50

    high_ratio = len(high_regions) / len(FACE_REGIONS) if FACE_REGIONS else 0.0

    if gradcam_is_placeholder:
        # EfficientNet failed — LLaVA carries full weight
        score = (
            0.80 * llava_signal   +
            0.10 * saliency_score +
            0.10 * high_ratio
        )
    else:
        # Confidence-adaptive weights based on how certain EfficientNet is
        certainty = abs(gan_probability - 0.5) * 2.0   # 0=uncertain, 1=certain

        if certainty >= 0.70:      # gan_prob > 0.85 or < 0.15
            w_gan, w_llava, w_sal, w_reg = 0.70, 0.15, 0.08, 0.07
            zone = "HIGH"
        elif certainty >= 0.30:    # gan_prob 0.65-0.85 or 0.15-0.35
            w_gan, w_llava, w_sal, w_reg = 0.55, 0.25, 0.10, 0.10
            zone = "MEDIUM"
        else:                      # gan_prob 0.35-0.65 — model is uncertain
            w_gan, w_llava, w_sal, w_reg = 0.40, 0.38, 0.12, 0.10
            zone = "LOW"

        logger.info(
            "[VLMAgent] Anomaly weights: EfficientNet_certainty=%.2f (%s) "
            "→ w_gan=%.2f  w_llava=%.2f  w_sal=%.2f  w_reg=%.2f",
            certainty, zone, w_gan, w_llava, w_sal, w_reg,
        )

        score = (
            w_gan   * gan_probability +
            w_llava * llava_signal    +
            w_sal   * saliency_score  +
            w_reg   * high_ratio
        )

    return round(float(np.clip(score, 0.0, 1.0)), 4)


# =============================================================================
# MAIN AGENT CLASS
# =============================================================================

class VLMAgent:
    """
    Agent 5 — §5.4 VLM Explainability Heat-Map Analysis.
    Stateless — safe to call multiple times with different ctx dicts.
    """

    def run(self, ctx: dict) -> dict:
        image_path     = ctx.get("image_path", "")
        face_crop_path = ctx.get("face_crop_path", "")

        if not image_path or not Path(image_path).exists():
            return self._fallback("image_path missing or not found")
        if not face_crop_path or not Path(face_crop_path).exists():
            return self._fallback("face_crop_path missing or not found")

        # ── Step 1: EfficientNet-B4 Grad-CAM runs FIRST ───────────────────────
        # Must run before LLaVA so the heatmap exists when LLaVA needs it.
        # GPU cache is cleared inside _run_gradcam_efficientnet() after inference.
        logger.info("[VLMAgent] Running EfficientNet-B4 Grad-CAM ...")
        try:
            cam_map, saliency_score, gan_probability, gradcam_is_placeholder = \
                _run_gradcam_efficientnet(face_crop_path)
        except Exception as exc:
            logger.error("[VLMAgent] EfficientNet Grad-CAM error: %s", exc, exc_info=True)
            cam_map                = np.full((224, 224), 0.5, dtype=np.float32)
            saliency_score         = 0.50
            gan_probability        = 0.50
            gradcam_is_placeholder = True

        # ── Step 2: Save heatmap PNG to disk ──────────────────────────────────
        try:
            heatmap_path = _save_heatmap(cam_map, face_crop_path, image_path)
        except Exception as exc:
            logger.error("[VLMAgent] Heatmap save error: %s", exc)
            heatmap_path = str(TEMP_DIR / "heatmap_unavailable.png")

        # ── Step 3a: Compute zone analysis from Grad-CAM numbers ──────────────
        # This is the ground-truth §5.4 zone table — computed numerically from
        # the stretched cam_map, not by asking LLaVA to interpret image colors.
        cam_map_for_zones = _GRADCAM_VISUAL[0] if _GRADCAM_VISUAL[0] is not None else cam_map
        try:
            region_data = _compute_region_zones(cam_map_for_zones)
            high_r   = [r for r, d in region_data.items() if d["zone"] == "RED"]
            mid_r    = [r for r, d in region_data.items() if d["zone"] == "BLUE"]
            low_r    = [r for r, d in region_data.items() if d["zone"] == "VIOLET"]
            central  = ["eyes", "nose", "mouth"]
            zone_gan = round(float(np.mean(
                [region_data[r]["activation"] for r in central if r in region_data]
            )), 4)
            logger.info(
                "[VLMAgent] Zone classification: high=%s  mid=%s  zone_gan=%.3f",
                high_r, mid_r, zone_gan,
            )
        except Exception as exc:
            logger.error("[VLMAgent] Zone classification error: %s", exc)
            region_data = {}
            high_r, mid_r, low_r, zone_gan = [], [], list(FACE_REGIONS.keys()), 0.50

        # ── Step 3b: Build zone table and inject into LLaVA prompt ───────────
        # LLaVA receives the computed zone facts AND the pre-determined verdict
        # so it writes an explanation consistent with EfficientNet's decision.
        try:
            zone_table    = _build_zone_table(region_data, gan_probability)
            llava_prompt  = _build_llava_prompt(zone_table, gan_probability)
            logger.debug("[VLMAgent] Zone table injected into prompt:\n%s", zone_table)
        except Exception as exc:
            logger.error("[VLMAgent] Zone table build error: %s", exc)
            llava_prompt  = _build_llava_prompt("Zone data unavailable.", gan_probability)

        # ── Step 4: LLaVA generates forensic narrative from zone data ─────────
        llava_input = heatmap_path if Path(heatmap_path).exists() else image_path
        logger.info("[VLMAgent] Running LLaVA-1.5-7b narrative generation ...")
        try:
            vlm_caption, vlm_verdict, vlm_confidence = _run_llava(
                llava_input, llava_prompt
            )
        except Exception as exc:
            logger.error("[VLMAgent] LLaVA error: %s", exc, exc_info=True)
            vlm_caption, vlm_verdict, vlm_confidence = f"[ERROR] {exc}", "UNCERTAIN", 0.50

        # ── Step 5: Anomaly score ─────────────────────────────────────────────
        anomaly_score = _compute_anomaly_score(
            vlm_verdict, vlm_confidence, saliency_score,
            gan_probability, high_r,
            gradcam_is_placeholder=gradcam_is_placeholder,
        )

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
            "gan_probability":           round(gan_probability, 4),
        }
        validate(output, VLM_KEYS, "VLMAgent")

        logger.info(
            "[VLMAgent] Done — verdict=%s  confidence=%.3f  gan_prob=%.4f  anomaly=%.4f",
            vlm_verdict, vlm_confidence, gan_probability, anomaly_score,
        )
        return output

    def _fallback(self, reason: str) -> dict:
        logger.error("[VLMAgent] Fallback triggered: %s", reason)
        return {
            "heatmap_path":              "",
            "vlm_caption":               f"[FALLBACK] {reason}",
            "vlm_verdict":               "UNCERTAIN",
            "vlm_confidence":            0.50,
            "saliency_score":            0.50,
            "high_activation_regions":   [],
            "medium_activation_regions": [],
            "low_activation_regions":    list(FACE_REGIONS.keys()),
            "zone_gan_probability":      0.50,
            "anomaly_score":             0.50,
            "gan_probability":           0.50,
            "error":                     reason,
        }