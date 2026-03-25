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

from pydantic import BaseModel, Field
from typing import Dict
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from scipy.stats import wasserstein_distance


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


def run_texture_agent_stub(image_path: str, face_bbox: list) -> TextureOutput:
    """
    Stub implementation — returns hardcoded dummy values.
    
    Unblocks pipeline for end-to-end testing before full implementation.
    All returned values are realistic but synthetic.
    
    Args:
        image_path: absolute path to image (e.g., "test_images/sample_fake.jpg")
        face_bbox: [x1, y1, x2, y2] pixel coordinates
        
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
            "cheek_L": 0.30,
            "cheek_R": 0.28,
            "jaw": 0.80,
            "nose": 0.15,
        },
        anomaly_score=0.895,
    )


def _lbp_histogram(zone_img: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Compute Local Binary Pattern histogram for a zone.
    
    LBP captures local texture patterns (edges, corners, flat regions).
    Higher max-bin probability indicates more uniform (natural) skin texture.
    
    Args:
        zone_img: RGB or grayscale region of face
        n_bins: histogram bin count (default 10 for efficiency)
        
    Returns:
        Normalized histogram array (sums to 1), shape (n_bins,)
    """
    # Convert to grayscale if needed
    if zone_img.ndim == 3:
        gray = cv2.cvtColor(zone_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = zone_img.astype(np.uint8)
    
    # Compute LBP with uniform patterns (8 neighbors, radius 1)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    
    # Histogram with density normalization (sums to ~1)
    hist, _ = np.histogram(
        lbp.ravel(), 
        bins=n_bins, 
        range=(0, n_bins), 
        density=True
    )
    
    return hist


def _gabor_energy(zone_img: np.ndarray) -> float:
    """
    Compute mean Gabor filter energy across multiple orientations and frequencies.
    
    Gabor filters detect directional texture patterns at multiple scales.
    Real skin has characteristic energy distribution; GAN-generated skin often shows
    unnatural peaks or valleys in this spectrum.
    
    Args:
        zone_img: RGB or grayscale region of face
        
    Returns:
        Mean energy across all Gabor scales/orientations (float)
    """
    # Convert to grayscale
    if zone_img.ndim == 3:
        gray = cv2.cvtColor(zone_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = zone_img.astype(np.uint8)
    
    gray = gray.astype(float) / 255.0
    
    energies = []
    
    # 4 orientations (0°, 45°, 90°, 135°)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        # 3 frequencies (coarse to fine texture)
        for freq in [0.1, 0.2, 0.4]:
            real, _ = gabor(gray, frequency=freq, theta=theta)
            # Energy = mean of squared response
            energy = np.mean(real ** 2)
            energies.append(energy)
    
    return float(np.mean(energies))


def run_texture_agent(image_path: str, face_bbox: list) -> TextureOutput:
    """
    Full texture consistency agent implementation.
    
    Pipeline:
      1. Load image and crop face region
      2. Define 5 facial zones (proportional subdivisions)
      3. Extract LBP histogram + Gabor energy per zone
      4. Compute Earth Mover's Distance (Wasserstein) between adjacent zones
      5. Calculate per-zone anomaly scores
      6. Fuse into final anomaly_score
      7. Detect seams (any EMD > 0.45)
    
    Args:
        image_path: absolute path to image file
        face_bbox: [x1, y1, x2, y2] bounding box in pixel coordinates
        
    Returns:
        TextureOutput with all measurements
        
    Raises:
        FileNotFoundError: if image_path does not exist
        ValueError: if face_bbox is invalid or region is too small
    """
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Load image and crop face region
    # ──────────────────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    x1, y1, x2, y2 = face_bbox
    face = img[y1:y2, x1:x2]
    h, w = face.shape[:2]
    
    if h < 50 or w < 50:
        raise ValueError(f"Face region too small: {w}x{h}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Define 5 facial zones (proportional to face bounding box)
    # ──────────────────────────────────────────────────────────────────────────
    zones = {
        "forehead": face[0 : int(h * 0.25), int(w * 0.2) : int(w * 0.8)],
        "nose": face[int(h * 0.35) : int(h * 0.65), int(w * 0.35) : int(w * 0.65)],
        "cheek_L": face[int(h * 0.25) : int(h * 0.70), 0 : int(w * 0.35)],
        "cheek_R": face[int(h * 0.25) : int(h * 0.70), int(w * 0.65) : w],
        "jaw": face[int(h * 0.70) : h, int(w * 0.1) : int(w * 0.9)],
    }
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Compute LBP histograms per zone
    # ──────────────────────────────────────────────────────────────────────────
    hists = {}
    for name, zone in zones.items():
        if zone.size == 0:
            # Fallback: use uniform histogram if zone is empty
            hists[name] = np.ones(10) / 10.0
        else:
            hists[name] = _lbp_histogram(zone)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Compute Gabor energy per zone
    # ──────────────────────────────────────────────────────────────────────────
    gabor_energies = {}
    for name, zone in zones.items():
        if zone.size == 0:
            gabor_energies[name] = 0.004  # Fallback to authentic baseline
        else:
            gabor_energies[name] = _gabor_energy(zone)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Earth Mover's Distance (Wasserstein) between adjacent zones
    # ──────────────────────────────────────────────────────────────────────────
    
    # Jaw-cheek boundaries (main seam detection)
    jaw_emd = float(
        np.mean([
            wasserstein_distance(hists["cheek_L"], hists["jaw"]),
            wasserstein_distance(hists["cheek_R"], hists["jaw"]),
        ])
    )
    
    # Cheek symmetry
    cheek_emd = float(wasserstein_distance(hists["cheek_L"], hists["cheek_R"]))
    
    # Neck region: strip just below face_bbox
    neck_strip = img[y2 : min(y2 + int(h * 0.15), img.shape[0]), x1:x2]
    
    if neck_strip.shape[0] > 5 and neck_strip.size > 0:
        neck_hist = _lbp_histogram(neck_strip)
        neck_emd = float(wasserstein_distance(hists["jaw"], neck_hist))
    else:
        # Fallback: estimate neck EMD from jaw stats
        neck_emd = jaw_emd * 0.9
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: LBP uniformity score (higher = more natural)
    # ──────────────────────────────────────────────────────────────────────────
    # Uniformity = max bin probability; mean across all zones
    lbp_uniformity = float(np.mean([np.max(h) for h in hists.values()]))
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: Per-zone anomaly scores (deviation from authentic baselines)
    # ──────────────────────────────────────────────────────────────────────────
    
    # Empirical baselines for authentic (real) skin
    AUTHENTIC_GABOR_MEAN = 0.004
    AUTHENTIC_LBP_UNIF = 0.85
    
    zone_scores = {}
    for name in zones:
        # Gabor deviation: relative distance from authentic mean
        if AUTHENTIC_GABOR_MEAN > 0:
            gabor_dev = abs(gabor_energies[name] - AUTHENTIC_GABOR_MEAN) / AUTHENTIC_GABOR_MEAN
        else:
            gabor_dev = 0.0
        
        # LBP deviation: relative distance from authentic uniformity
        lbp_max = np.max(hists[name])
        if AUTHENTIC_LBP_UNIF > 0:
            lbp_dev = abs(lbp_max - AUTHENTIC_LBP_UNIF) / AUTHENTIC_LBP_UNIF
        else:
            lbp_dev = 0.0
        
        # Combine and clip to [0, 1]
        zone_scores[name] = float(min(1.0, (gabor_dev + lbp_dev) / 2.0))
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: Final anomaly_score + seam flag
    # ──────────────────────────────────────────────────────────────────────────
    
    EMD_THRESHOLD = 0.45
    
    seam_detected = any(emd > EMD_THRESHOLD for emd in [jaw_emd, neck_emd, cheek_emd])
    
    # Raw score based on max EMD relative to threshold
    raw_score = max(jaw_emd, neck_emd) / EMD_THRESHOLD if EMD_THRESHOLD > 0 else 0.0
    anomaly_score = float(min(1.0, raw_score))
    
    # If seam is confirmed, enforce minimum anomaly score
    if seam_detected:
        anomaly_score = max(anomaly_score, 0.70)
    
    # ──────────────────────────────────────────────────────────────────────────
    # Step 9: Return Pydantic object
    # ──────────────────────────────────────────────────────────────────────────
    
    return TextureOutput(
        jaw_emd=jaw_emd,
        neck_emd=neck_emd,
        cheek_emd=cheek_emd,
        lbp_uniformity=lbp_uniformity,
        seam_detected=seam_detected,
        zone_scores=zone_scores,
        anomaly_score=anomaly_score,
    )


def texture_agent(image_path: str, face_bbox: list) -> dict:
    """
    Entry point for master_agent.py integration.
    
    Calls run_texture_agent and returns dict representation for compatibility
    with LangChain @tool decorator and pipeline serialization.
    
    Args:
        image_path: absolute path to image
        face_bbox: [x1, y1, x2, y2] bounding box
        
    Returns:
        dict with all TextureOutput fields (serializable)
    """
    result = run_texture_agent(image_path, face_bbox)
    return result.model_dump()
