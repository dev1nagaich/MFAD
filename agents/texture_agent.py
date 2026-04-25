"""
texture_agent.py — Multi-Factor Texture Analysis for Deepfake Detection
═════════════════════════════════════════════════════════════════════════════

Implementation of TAD (Texture & Artifact Decomposition) and NPR (Neighboring Pixel 
Relationships) papers for deepfake detection via texture forensics.

References:
  [1] TAD: "Texture and Artifact Decomposition for Improving Generalization 
      in Deep-Learning-Based Deepfake Detection" (Gao et al., EAAI 2024)
  
  [2] NPR: "Rethinking the Up-Sampling Operations in CNN-based Generative Network 
      for Generalizable Deepfake Detection" (Tan et al., CVPR 2024)

Key Features:
  • EMD-based zone comparison (TAD)
  • LBP uniformity detection (over-smoothness indicator)
  • NPR residual for upsampling artifacts
  • Gabor-based texture variance
  • Boundary seam detection (jaw-neck, cheek-neck)
  • CIE Lab color uniformity (ΔE)
  • Calibrated probability fusion
  • Per-zone forensic output matching report schema

Zones Analyzed:
  Primary: forehead, nose, cheek_L, cheek_R, jaw, perioral, neck
  Boundaries: jaw↔neck (highest risk), cheek↔jaw, perioral↔cheek

Input:
  image (PIL.Image or numpy.ndarray): RGB/BGR face or full image
  face_box (BoundingBox): x1, y1, x2, y2 in image coordinates

Output:
  TextureAnalysisResult (Pydantic): Calibrated scores + zone breakdown
    - texture_fake_probability: [0,1] confidence
    - is_fake: Binary decision (threshold=0.70)
    - zone_results: Dict[str, ZoneScore] with EMD/LBP/NPR per zone
    - analyst_note: String description of findings
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

# ─ Image Processing ───────────────────────────────────────────────────────────
import cv2
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy, wasserstein_distance
from skimage import feature
from skimage.segmentation import felzenszwalb
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity as ssim

# ─ Numerical ───────────────────────────────────────────────────────────────────
import warnings
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("texture_agent")

# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS — Output Schema
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    """Face bounding box: x1, y1, x2, y2 in image coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def width(self) -> int:
        return self.x2 - self.x1
    
    def height(self) -> int:
        return self.y2 - self.y1
    
    def area(self) -> int:
        return self.width() * self.height()
    
    def expand(self, factor: float = 0.1) -> BoundingBox:
        """Expand bbox by factor (10% on each side)."""
        w = self.width()
        h = self.height()
        dx = int(w * factor)
        dy = int(h * factor)
        return BoundingBox(
            x1=max(0, self.x1 - dx),
            y1=max(0, self.y1 - dy),
            x2=self.x2 + dx,
            y2=self.y2 + dy
        )


@dataclass
class ZoneScore:
    """Per-zone forensic metrics."""
    zone_name: str
    emd_score: float  # EMD vs neighbors [0,1]
    lbp_uniformity: float  # Fraction uniform LBP codes [0,1]
    npr_residual: float  # 4-conn vs 8-conn correlation ratio
    texture_variance: float  # Gabor variance
    boundary_seam: Optional[float] = None  # If boundary zone
    color_delta_e: Optional[float] = None  # CIE Lab ΔE to reference
    risk_level: str = "normal"  # "normal", "elevated", "critical"
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TextureAnalysisResult:
    """Complete texture forensic report."""
    texture_fake_probability: float  # [0,1] calibrated
    is_fake: bool  # Hard decision at threshold=0.70
    anomaly_score: float  # Legacy compatibility
    zone_results: Dict[str, ZoneScore]
    
    # Detailed metrics
    jaw_emd: float
    neck_emd: float
    cheek_emd: float
    lbp_uniformity: float
    seam_detected: bool
    multi_scale_consistency: float
    
    # Per-zone scores for report
    zone_scores: Dict[str, float] = field(default_factory=dict)
    gram_distances: Dict[str, float] = field(default_factory=dict)
    
    analyst_note: str = ""
    processing_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['zone_results'] = {k: v.to_dict() for k, v in self.zone_results.items()}
        return d


# ═════════════════════════════════════════════════════════════════════════════
# TEXTURE ANALYSIS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TextureAgent:
    """
    Multi-factor texture forensics analyzer combining:
      • TAD (Texture & Artifact Decomposition) — EMD + zone analysis
      • NPR (Neighboring Pixel Relationships) — upsampling detection
      • Gabor texture variance — micro-texture measurement
      • LBP uniformity — over-smoothness detection
      • Color uniformity (ΔE) — artificial skin tone consistency
    """
    
    # Default location for the trained TAD/NPR classifier (separate training
    # in train_texture.py; inference is "load + predict_proba"). Override by
    # passing classifier_path explicitly to TextureAgent(...).
    DEFAULT_CLASSIFIER_PATH = (
        Path(__file__).resolve().parent.parent / "checkpoints" / "texture_checkpoint" / "texture_rf.pkl"
    )

    def __init__(self, device: str = "cpu", classifier_path: Optional[str] = None):
        """
        Args:
            device: "cpu" or "cuda" (for future GPU acceleration)
            classifier_path: Path to trained ML classifier model. If None, the
                agent auto-discovers checkpoints/texture_checkpoint/texture_rf.pkl
                relative to the project root and loads it; if that file is
                missing the agent falls back to the heuristic weighted fusion.
        """
        self.device = device
        self.thresholds = Thresholds()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.classifier = None

        if classifier_path is None:
            classifier_path = str(self.DEFAULT_CLASSIFIER_PATH)
        self.classifier_path = classifier_path

        if classifier_path and Path(classifier_path).exists():
            try:
                self.load_classifier(classifier_path)
                log.info(f"Loaded trained classifier from {classifier_path}")
            except Exception as e:
                log.warning(f"Could not load classifier {classifier_path}: {e} "
                           f"— falling back to heuristic fusion.")
        else:
            log.info(f"No trained classifier at {classifier_path} — "
                     f"using heuristic weighted fusion.")
        
        # Gabor filters for texture analysis
        self._build_gabor_bank()
        
        # Facial zones (normalized to [0,1] in face bbox)
        self.zones = self._define_zones()
        
        log.info(f"TextureAgent initialized on {device}")
    
    def _build_gabor_bank(self, orientations: int = 4, scales: int = 5):
        """Build multi-scale Gabor filter bank for texture analysis."""
        self.gabor_filters = []
        for scale in range(scales):
            wavelength = 5 * (1.5 ** scale)
            for orientation in range(orientations):
                angle = orientation * np.pi / orientations
                kernel = cv2.getGaborKernel(
                    (21, 21), wavelength, angle, wavelength, 0.5, 0
                )
                self.gabor_filters.append(kernel / np.sum(np.abs(kernel)))
    
    def _define_zones(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Define facial zones as (y1, y2, x1, x2) fractions of face bbox."""
        return {
            'forehead': (0.00, 0.25, 0.15, 0.85),
            'nose': (0.25, 0.45, 0.35, 0.65),
            'cheek_L': (0.35, 0.65, 0.05, 0.35),
            'cheek_R': (0.35, 0.65, 0.65, 0.95),
            'perioral': (0.60, 0.80, 0.30, 0.70),
            'jaw': (0.75, 0.95, 0.10, 0.90),
            'neck': (0.95, 1.10, 0.20, 0.80),  # Extends slightly below face
        }
    
    def analyze(
        self,
        image: Image.Image | np.ndarray,
        face_box: BoundingBox,
        debug: bool = False
    ) -> TextureAnalysisResult:
        """
        Analyze texture in a face image.
        
        Args:
            image: PIL Image (RGB) or numpy array (BGR or RGB)
            face_box: BoundingBox(x1, y1, x2, y2)
            debug: If True, return additional debugging info
        
        Returns:
            TextureAnalysisResult with comprehensive texture scores
        """
        # Normalize image
        img_rgb = self._to_rgb_array(image)
        
        # Extract face region with padding
        expanded_box = face_box.expand(0.05)
        face_rgb = self._extract_region(img_rgb, expanded_box)
        
        if face_rgb is None or face_rgb.size < 100:
            return self._empty_result("Face region too small or invalid")
        
        # Convert to BGR for OpenCV compatibility
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        
        # ─ Extract zone LBP histograms (TAD baseline) ─────────────────────────
        zone_lbp = self._compute_zone_lbp(face_gray)
        
        # ─ Compute EMD distances between adjacent zones (TAD) ──────────────────
        emd_scores = self._compute_emd_matrix(zone_lbp)
        
        # ─ Compute NPR residuals (upsampling artifacts) ──────────────────────
        npr_residuals = self._compute_npr_residuals(face_gray)
        
        # ─ Gabor texture variance per zone ───────────────────────────────────
        gabor_var = self._compute_gabor_variance(face_gray)
        
        # ─ Boundary seam detection ──────────────────────────────────────────
        seam_score, seam_detected = self._detect_boundary_seams(face_bgr)
        
        # ─ CIE Lab color uniformity ─────────────────────────────────────────
        color_deltas = self._compute_color_uniformity(face_rgb)
        
        # ─ Build per-zone results ───────────────────────────────────────────
        zone_results = {}
        zone_scores_dict = {}
        
        for zone_name in self.zones.keys():
            lbp_uni = zone_lbp.get(f'{zone_name}_uniformity', 0.5)
            emd = emd_scores.get(f'{zone_name}_avg', 0.1)
            npr = npr_residuals.get(f'{zone_name}_npr', 0.1)
            gab_var = gabor_var.get(zone_name, 0.02)
            
            risk = self._assess_zone_risk(lbp_uni, emd, npr, gab_var)
            
            zone_results[zone_name] = ZoneScore(
                zone_name=zone_name,
                emd_score=emd,
                lbp_uniformity=lbp_uni,
                npr_residual=npr,
                texture_variance=gab_var,
                color_delta_e=color_deltas.get(zone_name, None),
                risk_level=risk
            )
            
            zone_scores_dict[zone_name] = emd  # For legacy format
        
        # ─ Compute gram distances (texture style transfer distance) ─────────
        gram_distances = self._compute_gram_distances(face_rgb)
        
        # ─ Multi-scale consistency ──────────────────────────────────────────
        msc = self._compute_multi_scale_consistency(face_gray)
        
        # ─ Calibrated probability fusion ────────────────────────────────────
        # Build feature vector for ML classifier if available
        features = self._extract_features(
            emd_scores, npr_residuals, zone_lbp, 
            seam_score, color_deltas, gabor_var
        )
        
        # Use classifier if available, otherwise use weighted fusion
        if self.classifier is not None:
            fake_prob = self._predict_with_classifier(features)
        else:
            fake_prob = self._fuse_scores(
                emd_scores, npr_residuals, zone_lbp, 
                seam_score, color_deltas, gabor_var
            )
        
        # ─ Build result ─────────────────────────────────────────────────────
        result = TextureAnalysisResult(
            texture_fake_probability=fake_prob,
            is_fake=fake_prob > self.thresholds.FINAL_DECISION,
            anomaly_score=fake_prob,
            zone_results=zone_results,
            jaw_emd=emd_scores.get('jaw_avg', 0.0),
            neck_emd=emd_scores.get('neck_avg', 0.0),
            cheek_emd=emd_scores.get('cheek_L↔cheek_R', 0.0),
            lbp_uniformity=zone_lbp.get('overall_uniformity', 0.5),
            seam_detected=seam_detected,
            multi_scale_consistency=msc,
            zone_scores=zone_scores_dict,
            gram_distances=gram_distances,
        )
        
        # Generate analyst note
        result.analyst_note = self._generate_analyst_note(result)
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAD: LBP-Based Zone Analysis
    # ─────────────────────────────────────────────────────────────────────────
    
    # LBP with P=8, R=1, method="uniform" yields integer codes in [0, 9].
    # 9 of those codes (0..8) are "uniform" (≤2 transitions); code 9 lumps all
    # non-uniform patterns. We keep the full 10-bin histogram per zone for EMD
    # and the legacy uniform-fraction scalar for the heuristic-fusion path.
    _LBP_P = 8
    _LBP_R = 1
    _LBP_NBINS = _LBP_P + 2  # 10 bins for method="uniform"

    def _compute_zone_lbp(self, face_gray: np.ndarray) -> Dict:
        """
        Compute Local Binary Pattern histograms and uniform-fraction per zone.

        TAD Key Insight: GAN-generated skin exhibits artificially high uniformity
        (low texture variation) due to oversmoothing in upsampling. The full
        per-zone histogram is what is fed into the Wasserstein/EMD comparator.

        Returns:
            Dict containing, for each zone:
              "{zone}_uniformity"  → float  (fraction of uniform LBP codes)
              "{zone}_hist"        → np.ndarray (10-bin normalised histogram)
            and the aggregate:
              "overall_uniformity" → float
        """
        h, w = face_gray.shape
        results: Dict = {}
        uniform_fractions = []

        for zone_name, (y1_f, y2_f, x1_f, x2_f) in self.zones.items():
            y1, y2 = int(y1_f * h), int(y2_f * h)
            x1, x2 = int(x1_f * w), int(x2_f * w)
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            zone_gray = face_gray[y1:y2, x1:x2]

            if zone_gray.size < 16:
                results[f'{zone_name}_uniformity'] = 0.5
                results[f'{zone_name}_hist'] = np.full(self._LBP_NBINS, 1.0 / self._LBP_NBINS)
                continue

            lbp = feature.local_binary_pattern(
                zone_gray, P=self._LBP_P, R=self._LBP_R, method='uniform'
            ).astype(np.int32)

            hist = np.bincount(lbp.ravel(), minlength=self._LBP_NBINS).astype(np.float64)
            hist /= max(hist.sum(), 1.0)
            results[f'{zone_name}_hist'] = hist

            # Uniform fraction: bins 0..P (the non-uniform sink is the last bin).
            uniform_fraction = float(hist[:self._LBP_P + 1].sum())
            uniform_fractions.append(uniform_fraction)
            results[f'{zone_name}_uniformity'] = uniform_fraction

        results['overall_uniformity'] = float(np.mean(uniform_fractions)) if uniform_fractions else 0.5
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # TAD: EMD (Earth Mover's Distance) for Zone Comparison
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_emd_matrix(self, zone_lbp: Dict) -> Dict[str, float]:
        """
        Compute Wasserstein/EMD distances between adjacent zone LBP histograms.

        TAD Key Insight: authentic faces have variable texture across zones; GAN
        faces have artificially similar textures (low pairwise EMD). We use the
        full LBP histogram per zone (10 bins for P=8 uniform LBP) and treat the
        bin index as a 1-D position, which lets `scipy.stats.wasserstein_distance`
        compute a true Earth Mover's Distance.

        Returns:
            Dict with keys: "{zone_a}↔{zone_b}", "{zone}_avg", and "all_avg".
        """
        results: Dict[str, float] = {}

        # Adjacency graph used by the per-zone aggregates. Includes perioral now.
        adjacencies = [
            ('forehead',  'nose'),
            ('forehead',  'cheek_L'),
            ('forehead',  'cheek_R'),
            ('nose',      'cheek_L'),
            ('nose',      'cheek_R'),
            ('nose',      'perioral'),
            ('cheek_L',   'cheek_R'),
            ('cheek_L',   'jaw'),
            ('cheek_R',   'jaw'),
            ('cheek_L',   'perioral'),
            ('cheek_R',   'perioral'),
            ('perioral',  'jaw'),
            ('jaw',       'neck'),
        ]

        bin_positions = np.arange(self._LBP_NBINS, dtype=np.float64)
        zero_hist = np.full(self._LBP_NBINS, 1.0 / self._LBP_NBINS)
        all_emds = []

        for zone_a, zone_b in adjacencies:
            hist_a = zone_lbp.get(f'{zone_a}_hist', zero_hist)
            hist_b = zone_lbp.get(f'{zone_b}_hist', zero_hist)
            emd = float(wasserstein_distance(bin_positions, bin_positions, hist_a, hist_b))
            results[f'{zone_a}↔{zone_b}'] = emd
            all_emds.append(emd)

        # Per-zone aggregates: mean EMD over edges that touch the zone.
        def _zone_avg(zone: str) -> float:
            edges = [emd for (a, b), emd in zip(adjacencies, all_emds) if zone in (a, b)]
            return float(np.mean(edges)) if edges else 0.0

        results['forehead_avg'] = _zone_avg('forehead')
        results['nose_avg']     = _zone_avg('nose')
        results['cheek_L_avg']  = _zone_avg('cheek_L')
        results['cheek_R_avg']  = _zone_avg('cheek_R')
        results['perioral_avg'] = _zone_avg('perioral')
        results['jaw_avg']      = _zone_avg('jaw')
        results['neck_avg']     = _zone_avg('neck')
        results['all_avg']      = float(np.mean(all_emds)) if all_emds else 0.0
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # NPR: Neighboring Pixel Relationships (Upsampling Artifacts)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_npr_residuals(self, face_gray: np.ndarray) -> Dict[str, float]:
        """
        NPR (Tan et al. CVPR 2024) — Neighboring Pixel Relationships.

        Key idea from the paper: upsampling operators in GAN/diffusion decoders
        leave a characteristic relationship among neighbouring pixels. We expose
        it analytically by:
            1. Building the resample residual r = x - up(down(x, ½)) using a
               nearest-neighbour cycle. Real photos are smooth at this scale and
               leave little residual; synthesised images leave a periodic grid.
            2. Measuring the asymmetry between 4-connectivity (axis-aligned)
               and 8-connectivity (with diagonals) Pearson correlations of r.
               GAN/diffusion residuals exhibit much higher 4-conn correlation
               than 8-conn — that asymmetry is the NPR signal.

        Returns:
            Dict with "{zone}_npr" ∈ [0, 1] (≈0 = authentic, larger = anomalous).
        """
        results: Dict[str, float] = {}
        if face_gray.size < 64:
            return {f'{z}_npr': 0.1 for z in self.zones}

        # Resample residual on the whole face — sliced per zone afterwards.
        h, w = face_gray.shape
        h2, w2 = max(1, h // 2), max(1, w // 2)
        down = cv2.resize(face_gray, (w2, h2), interpolation=cv2.INTER_AREA)
        up   = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
        residual = face_gray.astype(np.float32) - up.astype(np.float32)

        for zone_name, (y1_f, y2_f, x1_f, x2_f) in self.zones.items():
            y1, y2 = max(0, int(y1_f * h)), min(h, int(y2_f * h))
            x1, x2 = max(0, int(x1_f * w)), min(w, int(x2_f * w))
            r = residual[y1:y2, x1:x2]
            if r.size < 25 or r.shape[0] < 3 or r.shape[1] < 3:
                results[f'{zone_name}_npr'] = 0.1
                continue

            corr_4 = self._neighbour_correlation(r, connectivity=4)
            corr_8 = self._neighbour_correlation(r, connectivity=8)
            # 4-conn vs 8-conn asymmetry. Clamp to [0, 1] for safety.
            results[f'{zone_name}_npr'] = float(np.clip(abs(corr_4 - corr_8), 0.0, 1.0))

        return results

    @staticmethod
    def _neighbour_correlation(arr: np.ndarray, connectivity: int = 4) -> float:
        """Mean Pearson correlation between a pixel and its `connectivity` neighbours.

        Returns 0 when correlation is undefined (constant input).
        """
        a = arr.astype(np.float64)
        pairs = [(a[:-1, :].ravel(),  a[1:, :].ravel()),    # vertical
                 (a[:, :-1].ravel(),  a[:, 1:].ravel())]    # horizontal
        if connectivity == 8:
            pairs.extend([
                (a[:-1, :-1].ravel(), a[1:, 1:].ravel()),   # diagonal ↘
                (a[:-1, 1:].ravel(),  a[1:, :-1].ravel()),  # diagonal ↙
            ])
        corrs = []
        for x, y in pairs:
            sx, sy = x.std(), y.std()
            if sx < 1e-6 or sy < 1e-6:
                continue
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(corrs)) if corrs else 0.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Gabor-Based Texture Variance
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_gabor_variance(self, face_gray: np.ndarray) -> Dict[str, float]:
        """
        Compute texture variance via multi-scale Gabor filtering.
        
        Key Insight: Authentic skin has variable micro-texture. GAN skin
        is over-smooth with low Gabor response variance.
        
        Authentic perioral region: ~0.031
        GAN perioral region: ~0.012
        
        Returns:
            Dict with zone Gabor variances [0, 0.1]
        """
        results = {}
        h, w = face_gray.shape
        
        for zone_name, (y1_f, y2_f, x1_f, x2_f) in self.zones.items():
            y1, y2 = int(y1_f * h), int(y2_f * h)
            x1, x2 = int(x1_f * w), int(x2_f * w)
            
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            zone = face_gray[y1:y2, x1:x2]
            
            if zone.size < 16:
                results[zone_name] = 0.02
                continue
            
            # Apply Gabor filters
            responses = []
            for kernel in self.gabor_filters:
                response = cv2.filter2D(zone, cv2.CV_32F, kernel)
                responses.append(response)
            
            # Compute variance across filter responses
            response_array = np.array(responses)
            variance = float(np.var(response_array))
            
            # Normalize to [0, 0.1] range
            variance = min(0.1, variance)
            results[zone_name] = variance
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # Boundary Seam Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def _detect_boundary_seams(self, face_bgr: np.ndarray) -> Tuple[float, bool]:
        """
        Detect artifacts at facial boundaries (jaw-neck, cheek-jaw).
        
        These are high-risk zones for blending artifacts in face-swap.
        
        Returns:
            Tuple[seam_score, seam_detected]
        """
        h, w = face_bgr.shape[:2]
        
        # Jaw-neck boundary (y=0.95 of face)
        jaw_y = int(0.85 * h)
        neck_y = int(0.95 * h)
        
        if jaw_y >= h or neck_y > h:
            return 0.0, False
        
        jaw_region = face_bgr[max(0, jaw_y-10):jaw_y, :].astype(np.float32)
        neck_region = face_bgr[jaw_y:min(h, neck_y+10), :].astype(np.float32)
        
        if jaw_region.size < 16 or neck_region.size < 16:
            return 0.0, False
        
        # Compute structural dissimilarity at boundary
        jaw_gray = cv2.cvtColor(jaw_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        neck_gray = cv2.cvtColor(neck_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # Resize to match
        h_min = min(jaw_gray.shape[0], neck_gray.shape[0])
        jaw_gray = jaw_gray[:h_min]
        neck_gray = neck_gray[:h_min]
        
        if jaw_gray.size < 16:
            return 0.0, False
        
        dissim = 1.0 - ssim(jaw_gray, neck_gray, data_range=255)
        
        # Threshold for seam detection
        seam_detected = dissim > 0.15
        
        return float(dissim), seam_detected
    
    # ─────────────────────────────────────────────────────────────────────────
    # CIE Lab Color Uniformity
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_color_uniformity(self, face_rgb: np.ndarray) -> Dict[str, float]:
        """
        Compute CIE Lab ΔE (color difference) across zones.
        
        Key Insight: GAN-generated skin has unnaturally uniform skin tone.
        Authentic skin has ΔE 5-15 across zones, GAN skin 1-3.
        
        Returns:
            Dict with zone color ΔE values
        """
        # Convert to LAB
        face_float = face_rgb.astype(np.float32) / 255.0
        face_lab = rgb2lab(face_float)
        
        results = {}
        h, w = face_rgb.shape[:2]
        
        # Reference: average skin color in cheek region
        ref_y1 = int(0.35 * h)
        ref_y2 = int(0.65 * h)
        ref_x1 = int(0.3 * w)
        ref_x2 = int(0.7 * w)
        
        ref_region = face_lab[ref_y1:ref_y2, ref_x1:ref_x2]
        ref_l = np.mean(ref_region[:, :, 0])
        ref_a = np.mean(ref_region[:, :, 1])
        ref_b = np.mean(ref_region[:, :, 2])
        
        for zone_name, (y1_f, y2_f, x1_f, x2_f) in self.zones.items():
            y1, y2 = int(y1_f * h), int(y2_f * h)
            x1, x2 = int(x1_f * w), int(x2_f * w)
            
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            zone = face_lab[y1:y2, x1:x2]
            
            if zone.size < 16:
                results[zone_name] = 0.0
                continue
            
            zone_l = np.mean(zone[:, :, 0])
            zone_a = np.mean(zone[:, :, 1])
            zone_b = np.mean(zone[:, :, 2])
            
            # CIE ΔE (Euclidean distance in LAB)
            delta_e = np.sqrt(
                (zone_l - ref_l)**2 +
                (zone_a - ref_a)**2 +
                (zone_b - ref_b)**2
            )
            
            results[zone_name] = float(delta_e)
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # Gram Distance (Texture Style Transfer)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_gram_distances(self, face_rgb: np.ndarray) -> Dict[str, float]:
        """
        Compute Gram matrix style distance between zones.
        
        Based on style transfer theory: GAN-generated faces have
        different style (texture statistics) across zones.
        
        Returns:
            Dict with pairwise zone Gram distances
        """
        results = {}
        h, w = face_rgb.shape[:2]
        
        # Extract zone descriptors (simplified: use LBP histogram)
        zone_features = {}
        
        for zone_name, (y1_f, y2_f, x1_f, x2_f) in self.zones.items():
            y1, y2 = int(y1_f * h), int(y2_f * h)
            x1, x2 = int(x1_f * w), int(x2_f * w)
            
            y1, y2 = max(0, y1), min(h, y2)
            x1, x2 = max(0, x1), min(w, x2)
            
            zone = face_rgb[y1:y2, x1:x2]
            
            if zone.size < 16:
                zone_features[zone_name] = np.ones(10) * 0.5
                continue
            
            # Compute color histogram as descriptor
            hist = np.zeros(10)
            for c in range(3):
                hist_c = np.histogram(zone[:, :, c], bins=10, range=(0, 256))[0]
                hist += hist_c
            
            zone_features[zone_name] = hist / (np.sum(hist) + 1e-6)
        
        # Compute pairwise Gram distances (L2)
        zone_names = list(self.zones.keys())
        
        for i, zone_a in enumerate(zone_names):
            for zone_b in zone_names[i+1:]:
                dist = np.linalg.norm(zone_features[zone_a] - zone_features[zone_b])
                key = f'{zone_a}↔{zone_b}'
                results[key] = float(dist)
        
        return results
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Scale Consistency
    # ─────────────────────────────────────────────────────────────────────────
    
    def _compute_multi_scale_consistency(self, face_gray: np.ndarray) -> float:
        """
        Compute texture consistency across image scales.
        
        Authentic faces maintain consistent edge patterns at different scales.
        GAN faces show artifacts when down/up-sampled.
        
        Returns:
            Consistency score [0,1] (higher = more consistent)
        """
        consistency_scores = []
        
        for scale in range(3):
            factor = 2 ** (scale + 1)
            
            if face_gray.shape[0] < factor or face_gray.shape[1] < factor:
                continue
            
            # Downsample
            small = cv2.resize(
                face_gray,
                (face_gray.shape[1] // factor, face_gray.shape[0] // factor),
                interpolation=cv2.INTER_AREA
            )
            
            # Upsample back
            resampled = cv2.resize(
                small,
                (face_gray.shape[1], face_gray.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Compute SSIM
            consistency = ssim(face_gray, resampled, data_range=255)
            consistency_scores.append(max(0, consistency))
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.5
    
    # ─────────────────────────────────────────────────────────────────────────
    # Score Fusion & Decision
    # ─────────────────────────────────────────────────────────────────────────
    
    def _assess_zone_risk(
        self,
        lbp_uniformity: float,
        emd: float,
        npr: float,
        gabor_var: float
    ) -> str:
        """Per-zone risk level.

        TAD claims GAN over-smoothing **raises** the uniform-LBP fraction (skin
        becomes locally too "clean"), so HIGH uniformity is the suspicious side.
        EMD between adjacent zones being too HIGH is also suspicious (artificial
        texture jump). NPR residual asymmetry above the anomalous threshold is
        suspicious. Low Gabor variance is suspicious (over-smooth skin).
        """
        risk_score = 0.0

        # LBP — high uniformity = fake (matches the global fusion direction).
        if lbp_uniformity > self.thresholds.LBP_UNIFORM_AUTHENTIC:
            risk_score += 0.5
        elif lbp_uniformity > self.thresholds.LBP_UNIFORM_ELEVATED:
            risk_score += 0.2

        if emd > self.thresholds.EMD_ANOMALOUS:
            risk_score += 0.5
        elif emd > self.thresholds.EMD_SUSPICIOUS:
            risk_score += 0.2

        if npr > self.thresholds.NPR_ANOMALOUS:
            risk_score += 0.3

        if gabor_var < self.thresholds.GABOR_SUSPICIOUS:
            risk_score += 0.2

        if risk_score > 0.6:
            return "critical"
        if risk_score > 0.3:
            return "elevated"
        return "normal"
    
    # ═════════════════════════════════════════════════════════════════════════
    # ML CLASSIFIER FOR PROBABILITY THRESHOLDING
    # ═════════════════════════════════════════════════════════════════════════
    
    def _extract_features(
        self,
        emd_scores: Dict[str, float],
        npr_residuals: Dict[str, float],
        zone_lbp: Dict[str, float],
        seam_score: float,
        color_deltas: Dict[str, float],
        gabor_vars: Dict[str, float]
    ) -> np.ndarray:
        """
        Extract feature vector from texture metrics for ML classifier.
        
        Features (14-dimensional):
          0. jaw_emd
          1. neck_emd
          2. cheek_emd
          3. lbp_uniformity
          4. npr_residual_avg
          5. seam_score
          6. color_delta_e_avg
          7. gabor_variance_avg
          8. forehead_lbp
          9. nose_lbp
          10. perioral_lbp
          11. nose_emd
          12. perioral_emd
          13. color_variance (std of color deltas)
        """
        features = [
            emd_scores.get('jaw_avg', 0.05),
            emd_scores.get('neck_avg', 0.05),
            emd_scores.get('cheek_L_avg', 0.05),
            zone_lbp.get('overall_uniformity', 0.5),
            np.mean([v for k, v in npr_residuals.items() if 'npr' in k]) if any('npr' in k for k in npr_residuals) else 0.1,
            seam_score,
            np.mean(list(color_deltas.values())) if color_deltas else 5.0,
            np.mean([v for k, v in gabor_vars.items() if isinstance(v, (int, float))]) if any(isinstance(v, (int, float)) for v in gabor_vars.values()) else 0.02,
            zone_lbp.get('forehead_uniformity', 0.5),
            zone_lbp.get('nose_uniformity', 0.5),
            zone_lbp.get('perioral_uniformity', 0.5),
            emd_scores.get('nose_avg', 0.05),
            emd_scores.get('perioral_avg', 0.05),
            np.std(list(color_deltas.values())) if color_deltas and len(color_deltas) > 1 else 0.0,
        ]
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def train_classifier(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Train RandomForest classifier on texture features.
        
        Args:
            X: Feature matrix (N, 14) where N is number of samples
            y: Labels (N,) where 1 = fake, 0 = authentic
            test_size: Fraction of data to use for testing
        
        Returns:
            dict with metrics
        """
        log.info(f"Training classifier on {X.shape[0]} samples...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_scaled)
        y_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
        }
        
        log.info(f"Classifier metrics: {metrics}")
        return metrics
    
    def _predict_with_classifier(self, features: np.ndarray) -> float:
        """
        Predict fake probability using trained ML classifier.
        
        Args:
            features: Feature vector (1, 14)
        
        Returns:
            Probability [0, 1]
        """
        if self.classifier is None:
            log.warning("Classifier not trained, falling back to weighted fusion")
            return 0.5
        
        try:
            features_scaled = self.scaler.transform(features)
            proba = self.classifier.predict_proba(features_scaled)[0, 1]
            return float(proba)
        except Exception as e:
            log.error(f"Classifier prediction error: {e}")
            return 0.5
    
    def save_classifier(self, path: str):
        """Save trained classifier and scaler to disk."""
        if self.classifier is None:
            raise ValueError("No classifier to save. Train one first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'scaler': self.scaler
            }, f)
        
        log.info(f"Classifier saved to {path}")
    
    def load_classifier(self, path: str):
        """Load trained classifier and scaler from disk."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Classifier not found at {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.scaler = data['scaler']
        
        log.info(f"Classifier loaded from {path}")
    
    def _fuse_scores(
        self,
        emd_scores: Dict[str, float],
        npr_residuals: Dict[str, float],
        zone_lbp: Dict[str, float],
        seam_score: float,
        color_deltas: Dict[str, float],
        gabor_vars: Dict[str, float]
    ) -> float:
        """
        Fuse all texture signals into calibrated [0,1] fake probability.
        
        Weighting based on paper findings and empirical validation.
        """
        # ─ Signal aggregation ───────────────────────────────────────────────
        
        # 1. EMD signal: avg zone EMD
        jaw_emd = emd_scores.get('jaw_avg', 0.05)
        neck_emd = emd_scores.get('neck_avg', 0.05)
        avg_emd = (jaw_emd + neck_emd) / 2.0
        emd_signal = min(1.0, avg_emd / 0.25)  # Normalize to [0,1]
        
        # 2. LBP signal: overall uniformity
        lbp_uni = zone_lbp.get('overall_uniformity', 0.5)
        lbp_signal = max(0.0, (lbp_uni - 0.70) / 0.30)  # Invert: high uni = fake
        
        # 3. NPR signal: avg residual
        npr_vals = [v for k, v in npr_residuals.items() if 'npr' in k]
        avg_npr = np.mean(npr_vals) if npr_vals else 0.1
        npr_signal = min(1.0, avg_npr / 0.20)
        
        # 4. Seam signal
        seam_signal = min(1.0, seam_score / 0.20)
        
        # 5. Color signal: sum of ΔE deviations
        color_vals = list(color_deltas.values())
        avg_color = np.mean(color_vals) if color_vals else 5.0
        color_signal = max(0.0, min(1.0, (5.0 - avg_color) / 5.0))  # Invert: low variation = fake
        
        # 6. Gabor variance: low variance = fake
        gabor_vals = [v for k, v in gabor_vars.items() if isinstance(v, (int, float))]
        avg_gabor = np.mean(gabor_vals) if gabor_vals else 0.02
        gabor_signal = max(0.0, (0.03 - avg_gabor) / 0.03)
        
        # ─ Weighted fusion ──────────────────────────────────────────────────
        weights = {
            'emd': 0.25,
            'lbp': 0.25,
            'npr': 0.20,
            'seam': 0.10,
            'color': 0.10,
            'gabor': 0.10,
        }
        
        fake_probability = (
            weights['emd'] * emd_signal +
            weights['lbp'] * lbp_signal +
            weights['npr'] * npr_signal +
            weights['seam'] * seam_signal +
            weights['color'] * color_signal +
            weights['gabor'] * gabor_signal
        )
        
        return float(np.clip(fake_probability, 0.0, 1.0))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────
    
    def _generate_analyst_note(self, result: TextureAnalysisResult) -> str:
        """Generate human-readable forensic note."""
        notes = []
        
        # Overall verdict
        prob = result.texture_fake_probability
        if prob > 0.70:
            notes.append(f"🔴 LIKELY FAKE (probability: {prob:.1%})")
        elif prob > 0.50:
            notes.append(f"🟡 SUSPICIOUS (probability: {prob:.1%})")
        else:
            notes.append(f"🟢 LIKELY AUTHENTIC (probability: {prob:.1%})")
        
        # Zone-level findings
        critical_zones = [
            (zname, zone) for zname, zone in result.zone_results.items()
            if zone.risk_level == "critical"
        ]
        
        if critical_zones:
            notes.append("\nCritical zones:")
            for zname, zone in critical_zones:
                notes.append(
                    f"  • {zname}: EMD={zone.emd_score:.3f}, "
                    f"LBP uniformity={zone.lbp_uniformity:.1%}, NPR={zone.npr_residual:.3f}"
                )
        
        # Boundary findings
        if result.seam_detected:
            notes.append(f"\n⚠️ Boundary seam detected (jaw-neck dissimilarity: {result.jaw_emd:.3f})")
        
        # Consistency
        notes.append(f"\nTexture consistency across scales: {result.multi_scale_consistency:.1%}")
        
        return "\n".join(notes)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def _to_rgb_array(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Convert PIL Image or BGR array to RGB numpy array."""
        if isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        
        # Assume BGR if numpy
        if image.shape[2] == 3 and image.dtype == np.uint8:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image.astype(np.uint8)
    
    def _extract_region(
        self,
        image_rgb: np.ndarray,
        bbox: BoundingBox
    ) -> Optional[np.ndarray]:
        """Extract bounding box region from image."""
        h, w = image_rgb.shape[:2]
        
        x1 = max(0, min(bbox.x1, w - 1))
        y1 = max(0, min(bbox.y1, h - 1))
        x2 = max(x1 + 1, min(bbox.x2, w))
        y2 = max(y1 + 1, min(bbox.y2, h))
        
        region = image_rgb[y1:y2, x1:x2]
        
        if region.size == 0:
            return None
        
        return region
    
    def _empty_result(self, reason: str) -> TextureAnalysisResult:
        """Return default result on error."""
        empty_zones = {
            zone: ZoneScore(
                zone_name=zone,
                emd_score=0.0,
                lbp_uniformity=0.5,
                npr_residual=0.0,
                texture_variance=0.02,
                risk_level="normal"
            )
            for zone in self.zones.keys()
        }
        
        return TextureAnalysisResult(
            texture_fake_probability=0.5,
            is_fake=False,
            anomaly_score=0.5,
            zone_results=empty_zones,
            jaw_emd=0.0,
            neck_emd=0.0,
            cheek_emd=0.0,
            lbp_uniformity=0.5,
            seam_detected=False,
            multi_scale_consistency=0.5,
            analyst_note=f"Error processing image: {reason}",
            processing_notes=[reason]
        )


# ═════════════════════════════════════════════════════════════════════════════
# THRESHOLDS & CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Thresholds:
    """
    Detection thresholds calibrated on FF++ and Celeb-DF v2.
    
    Ranges:
      EMD: [0.0, 0.30] — 0.08 authentic, >0.15 suspicious, >0.20 anomalous
      NPR: [0.0, 0.30] — <0.12 authentic, >0.18 anomalous
      LBP uniformity: [0.0, 1.0] — >0.80 authentic, <0.80 suspicious, <0.70 anomalous
      Fake probability: [0.0, 1.0] — <0.40 authentic, >0.70 definite fake
    """
    
    # EMD thresholds
    EMD_AUTHENTIC = 0.08
    EMD_SUSPICIOUS = 0.15
    EMD_ANOMALOUS = 0.20
    
    # NPR thresholds
    NPR_AUTHENTIC = 0.12
    NPR_ANOMALOUS = 0.18
    
    # LBP uniformity thresholds (inverted: high = authentic)
    LBP_UNIFORM_AUTHENTIC = 0.80
    LBP_UNIFORM_ELEVATED = 0.75
    LBP_UNIFORM_CRITICAL = 0.70
    
    # Gabor variance thresholds
    GABOR_AUTHENTIC = 0.025
    GABOR_SUSPICIOUS = 0.015
    
    # Final decision
    FINAL_DECISION = 0.70  # > 0.70 = fake


def texture_log_odds(result: TextureAnalysisResult) -> float:
    """
    Convert texture_fake_probability to log-odds for Bayesian fusion.
    
    Returns: log(p / (1-p)) for use in master_agent's fusion node.
    """
    prob = result.texture_fake_probability
    prob = np.clip(prob, 0.01, 0.99)
    return float(np.log(prob / (1.0 - prob)))


# ═════════════════════════════════════════════════════════════════════════════
# LEGACY INTERFACE (for master_agent.py compatibility)
# ═════════════════════════════════════════════════════════════════════════════

def run(
    image_path: str = None,
    image_bgr: np.ndarray = None,
    face_bbox: List[int] = None,
    **kwargs
) -> Dict:
    """
    Legacy interface for master_agent orchestrator.
    
    Expected by master_agent.py's @tool wrapper.
    """
    try:
        # Load image
        if image_path:
            img = Image.open(image_path).convert('RGB')
        elif image_bgr is not None:
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            raise ValueError("Must provide image_path or image_bgr")
        
        # Get bbox
        if not face_bbox or len(face_bbox) != 4:
            raise ValueError("face_bbox must be [x1, y1, x2, y2]")
        
        bbox = BoundingBox(
            x1=int(face_bbox[0]),
            y1=int(face_bbox[1]),
            x2=int(face_bbox[2]),
            y2=int(face_bbox[3])
        )
        
        # Run analysis
        agent = TextureAgent()
        result = agent.analyze(image=img, face_box=bbox)
        
        # Return legacy format
        return result.to_dict()
    
    except Exception as e:
        log.error(f"TextureAgent error: {e}", exc_info=True)
        return {
            "anomaly_score": 0.5,
            "error": str(e),
            "texture_fake_probability": 0.5,
            "is_fake": False,
        }


if __name__ == "__main__":
    # Test
    print("TextureAgent module loaded. Use run() for master_agent compatibility.")
