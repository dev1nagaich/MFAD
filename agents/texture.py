#!/usr/bin/env python3
"""
agents/texture.py — Texture & Skin Consistency Agent
=====================================================
MFAD — Multimodal Forensic Agent for Deepfake Detection

Detects blending seams in deepfake face images by comparing skin texture
statistics across 5 facial zones using Local Binary Patterns, Gabor filters,
and Earth Mover's Distance (Wasserstein distance).

No model downloads. No GPU needed. Pure math only.

Owner: Manya Gupta
Module: agents/texture.py
"""

import warnings
from pydantic import BaseModel, Field
from typing import Dict, Optional
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from scipy.stats import wasserstein_distance

try:
    from langchain.tools import tool as langchain_tool
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    def langchain_tool(fn):          # no-op shim when LangChain not installed
        return fn


class TextureOutput(BaseModel):
    """
    Pydantic output schema for texture consistency measurements.

    Maps to §5.3 Texture Consistency & Skin-Tone Mapping in forensic report.
    Fed into Bayesian fusion with weight 0.20 for final deepfake probability.
    """
    jaw_emd: float = Field(
        ...,
        description="Wasserstein distance between cheek and jaw, detecting boundary seams"
    )
    neck_emd: float = Field(
        ...,
        description="Wasserstein distance between jaw and neck region, primary seam indicator"
    )
    cheek_emd: float = Field(
        ...,
        description="Wasserstein distance between left and right cheeks, symmetry check"
    )
    lbp_uniformity: float = Field(
        ...,
        description="Mean LBP uniformity across zones, 0–1 range. Higher = more natural skin"
    )
    seam_detected: bool = Field(
        ...,
        description="True if any EMD > 0.45, indicating blending artefacts"
    )
    zone_scores: Dict[str, float] = Field(
        ...,
        description="Per-zone anomaly scores. Keys: forehead, cheek_L, cheek_R, jaw, nose"
    )
    anomaly_score: float = Field(
        ...,
        description="Final fused texture anomaly score 0.0–1.0 for Bayesian fusion"
    )


# ---------------------------------------------------------------------------
# Stub
# ---------------------------------------------------------------------------

def run_texture_agent_stub(image_path: str, face_bbox: list) -> TextureOutput:
    """
    Stub implementation — returns hardcoded dummy values.

    Unblocks pipeline for end-to-end testing before full implementation.
    All returned values are realistic but synthetic.

    Args:
        image_path: absolute path to image (e.g. "test_images/sample_fake.jpg")
        face_bbox:  [x1, y1, x2, y2] pixel coordinates

    Returns:
        TextureOutput with realistic deepfake indicators
    """
    return TextureOutput(
        jaw_emd=0.61,
        neck_emd=0.48,
        cheek_emd=0.22,
        lbp_uniformity=0.31,
        seam_detected=True,
        zone_scores={
            "forehead": 0.20,
            "cheek_L":  0.30,
            "cheek_R":  0.28,
            "jaw":      0.80,
            "nose":     0.15,
        },
        anomaly_score=0.895,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _lbp_histogram(zone_img: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Compute Local Binary Pattern histogram for a zone.

    LBP captures local texture patterns (edges, corners, flat regions).
    Higher max-bin probability indicates more uniform (natural) skin texture.

    FIX vs original: histogram is normalised via count / total, NOT density=True.
    density=True returns probability *density* (can exceed 1.0 for narrow bins),
    which breaks lbp_uniformity when it is used as a [0,1] score.

    Args:
        zone_img: RGB or grayscale region of face
        n_bins:   histogram bin count (default 10 for efficiency)

    Returns:
        Normalised histogram array where every value is in [0, 1] and sum == 1.
        Shape: (n_bins,)
    """
    gray = cv2.cvtColor(zone_img, cv2.COLOR_RGB2GRAY) if zone_img.ndim == 3 else zone_img.astype(np.uint8)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

    # BUG 1 FIX: use plain counts then divide — guarantees values in [0, 1]
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    total = hist.sum()
    if total > 0:
        hist = hist.astype(float) / total
    else:
        hist = np.ones(n_bins, dtype=float) / n_bins

    return hist


def _gabor_energy(zone_img: np.ndarray) -> float:
    """
    Compute mean Gabor filter energy across multiple orientations and frequencies.

    Gabor filters detect directional texture patterns at multiple scales.
    Real skin has characteristic energy distribution; GAN skin often shows
    unnatural peaks or valleys in this spectrum.

    Libraries used: skimage.filters.gabor  (NOT scipy — fixes docstring confusion)

    Args:
        zone_img: RGB or grayscale region of face

    Returns:
        Mean energy across all Gabor scales / orientations (float, always >= 0)
    """
    gray = cv2.cvtColor(zone_img, cv2.COLOR_RGB2GRAY) if zone_img.ndim == 3 else zone_img.astype(np.uint8)
    gray = gray.astype(float) / 255.0

    energies = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for freq in [0.1, 0.2, 0.4]:
            real, _ = gabor(gray, frequency=freq, theta=theta)
            energies.append(float(np.mean(real ** 2)))

    return float(np.mean(energies))


# ---------------------------------------------------------------------------
# Full implementation
# ---------------------------------------------------------------------------

def run_texture_agent(image_path: str, face_bbox: list) -> TextureOutput:
    """
    Full texture consistency agent implementation.

    Pipeline:
      1. Load image and crop face region
      2. Define 5 facial zones (proportional subdivisions)
      3. Extract LBP histogram + Gabor energy per zone
      4. Compute Earth Mover's Distance (Wasserstein) between adjacent zones
      5. Calculate per-zone anomaly scores  ← BUG 3 FIX: relative deviation
      6. Fuse into final anomaly_score      ← includes cheek_emd now
      7. Detect seams (any EMD > 0.45)

    Args:
        image_path: absolute path to image file
        face_bbox:  [x1, y1, x2, y2] bounding box in pixel coordinates

    Returns:
        TextureOutput with all measurements

    Raises:
        FileNotFoundError: if image_path does not exist
        ValueError:        if face_bbox is invalid or face region is too small
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Load image and crop face region
    # ─────────────────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x1, y1, x2, y2 = face_bbox
    face = img[y1:y2, x1:x2]
    h, w = face.shape[:2]

    if h < 50 or w < 50:
        raise ValueError(f"Face region too small: {w}x{h} px (minimum 50x50)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Define 5 facial zones (proportional to face bounding box)
    # ─────────────────────────────────────────────────────────────────────────
    zones = {
        "forehead": face[0          : int(h * 0.25), int(w * 0.2) : int(w * 0.8)],
        "nose":     face[int(h*0.35): int(h * 0.65), int(w * 0.35): int(w * 0.65)],
        "cheek_L":  face[int(h*0.25): int(h * 0.70), 0             : int(w * 0.35)],
        "cheek_R":  face[int(h*0.25): int(h * 0.70), int(w * 0.65) : w            ],
        "jaw":      face[int(h*0.70): h,              int(w * 0.1)  : int(w * 0.9)],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: LBP histograms per zone
    # ─────────────────────────────────────────────────────────────────────────
    hists: Dict[str, np.ndarray] = {}
    for name, zone in zones.items():
        if zone.size == 0:
            warnings.warn(f"Zone '{name}' is empty — using uniform histogram fallback")
            hists[name] = np.ones(10, dtype=float) / 10.0
        else:
            hists[name] = _lbp_histogram(zone)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Gabor energy per zone
    # ─────────────────────────────────────────────────────────────────────────
    gabor_energies: Dict[str, float] = {}
    for name, zone in zones.items():
        if zone.size == 0:
            gabor_energies[name] = 0.0
        else:
            gabor_energies[name] = _gabor_energy(zone)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Earth Mover's Distance (Wasserstein) between adjacent zones
    # ─────────────────────────────────────────────────────────────────────────

    # Jaw-cheek boundaries (main seam detection)
    jaw_emd = float(np.mean([
        wasserstein_distance(hists["cheek_L"], hists["jaw"]),
        wasserstein_distance(hists["cheek_R"], hists["jaw"]),
    ]))

    # Cheek symmetry — left vs right should be similar on a real face
    cheek_emd = float(wasserstein_distance(hists["cheek_L"], hists["cheek_R"]))

    # BUG 4 FIX: compute neck_emd fully before using jaw_emd as fallback.
    # The fallback is now clearly labelled and placed after jaw_emd is final.
    neck_strip = img[y2 : min(y2 + int(h * 0.15), img.shape[0]), x1:x2]

    if neck_strip.shape[0] > 5 and neck_strip.size > 0:
        neck_hist = _lbp_histogram(neck_strip)
        neck_emd  = float(wasserstein_distance(hists["jaw"], neck_hist))
    else:
        # Neck not visible (tight crop) — conservative estimate from jaw stats
        neck_emd = jaw_emd * 0.9   # explicit post-jaw fallback

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: LBP uniformity score
    # Each histogram value is now in [0, 1] (BUG 1 fix), so max is also in [0, 1]
    # ─────────────────────────────────────────────────────────────────────────
    lbp_uniformity = float(np.mean([np.max(h) for h in hists.values()]))
    # Defensive clamp in case of floating-point edge cases
    lbp_uniformity = float(np.clip(lbp_uniformity, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Per-zone anomaly scores
    #
    # BUG 3 FIX: replaced hardcoded AUTHENTIC_GABOR_MEAN = 0.004 baseline with
    # a relative (intra-image) deviation metric.  The old approach produced
    # gabor_dev ≈ 4.0 for any normal image (authentic mean was 50× too small),
    # causing every zone_score to clip to 1.0 on every image.
    #
    # New approach:
    #   - Gabor deviation = how much this zone differs from the image's own mean
    #     (captures local inconsistency without a calibrated global baseline)
    #   - LBP deviation   = how far max-bin probability is from 1.0
    #     (1.0 = single dominant pattern = very uniform = natural skin)
    # ─────────────────────────────────────────────────────────────────────────
    mean_gabor = float(np.mean(list(gabor_energies.values()))) or 1e-8

    zone_scores: Dict[str, float] = {}
    for name in zones:
        # How different is this zone from the face average? (relative)
        gabor_dev = abs(gabor_energies[name] - mean_gabor) / (mean_gabor + 1e-8)

        # How far from a perfectly uniform LBP histogram?
        lbp_max = float(np.max(hists[name]))
        lbp_dev = 1.0 - lbp_max          # 0 = very uniform, 1 = very mixed

        zone_scores[name] = float(np.clip((gabor_dev + lbp_dev) / 2.0, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Final anomaly_score + seam flag
    #
    # FIX: cheek_emd is now included in the fusion score (previously computed
    # but silently ignored).  It is weighted at 0.8 because the jaw/neck
    # boundary is the primary deepfake indicator; cheek asymmetry is secondary.
    # ─────────────────────────────────────────────────────────────────────────
    EMD_THRESHOLD = 0.45

    seam_detected = any(emd > EMD_THRESHOLD for emd in [jaw_emd, neck_emd, cheek_emd])

    # Use the worst (max) boundary score as the primary signal
    worst_emd    = max(jaw_emd, neck_emd, cheek_emd * 0.8)
    raw_score    = worst_emd / EMD_THRESHOLD
    anomaly_score = float(np.clip(raw_score, 0.0, 1.0))

    # Confirmed seam → enforce minimum score
    if seam_detected:
        anomaly_score = max(anomaly_score, 0.70)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 9: Return Pydantic object
    # ─────────────────────────────────────────────────────────────────────────
    return TextureOutput(
        jaw_emd        = jaw_emd,
        neck_emd       = neck_emd,
        cheek_emd      = cheek_emd,
        lbp_uniformity = lbp_uniformity,
        seam_detected  = seam_detected,
        zone_scores    = zone_scores,
        anomaly_score  = anomaly_score,
    )


# ---------------------------------------------------------------------------
# LangChain entry point for master_agent.py
# ---------------------------------------------------------------------------

@langchain_tool
def texture_agent(image_path: str, face_bbox: list) -> dict:
    """
    Detects face-swap blending seams via LBP texture analysis and Earth
    Mover's Distance across facial zones.

    Entry point for master_agent.py — returns a plain dict so it is
    serialisable by LangChain's AgentExecutor and JSON logging.

    Args:
        image_path: absolute path to image
        face_bbox:  [x1, y1, x2, y2] pixel bounding box from preprocessing

    Returns:
        dict with all TextureOutput fields
    """
    result = run_texture_agent(image_path, face_bbox)
    return result.model_dump()