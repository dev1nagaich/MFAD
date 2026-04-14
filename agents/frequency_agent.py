# agents/frequency_agent.py
#
# Frequency Agent (L3) — Deep Sentinel Pipeline
#
# Two detection methods:
#
#   Method A — FFT Radial Spectrum (Durall et al. CVPR 2020)
#               "Watch Your Up-Convolution"
#               Produces: fft_mid_anomaly_db, fft_high_anomaly_db
#               Physically meaningful dB values for explainability layer.
#               Real face → ~0.0 dB | GAN face → mid +9.4 dB, high +13.3 dB
#
#   Method B — SVM on DCT/FFT features (Frank et al. ICML 2020 / arXiv 1911.00686)
#               "Leveraging Frequency Analysis for Deep Fake Image Recognition"
#               Uses the exact same preprocessing + feature extraction pipeline
#               the model was trained on:
#                 preprocess_image() → extract_features() → scaler → model
#               Produces: svm_fake_probability [0, 1]
#
# Output schema:
#   fft_mid_anomaly_db   — dB excess at mid frequencies  (explainability signal)
#   fft_high_anomaly_db  — dB excess at high frequencies (explainability signal)
#   svm_fake_probability — Frank 2020 SVM verdict [0, 1]
#   anomaly_score        — weighted fusion [0, 1]
#
# Fusion:
#   anomaly_score = 0.40 * fft_norm + 0.60 * svm_fake_probability
#   fft_norm is derived from fft_mid and fft_high dB values.
#
# Explainability note:
#   fft_mid_anomaly_db and fft_high_anomaly_db are preserved as raw dB numbers.
#   The report agent / explainability layer can render:
#   "Mid-frequency excess: +9.4 dB — consistent with StyleGAN2 upsampling signature."
#   svm_fake_probability provides the learned classifier signal.

import os
import pickle
import logging

import numpy as np
import cv2
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# ── Stub mode ──────────────────────────────────────────────────────────────────

STUB_MODE = False  # Set to True for testing without model

STUB_OUTPUT = {
    "fft_mid_anomaly_db":   0.0,
    "fft_high_anomaly_db":  0.0,
    "svm_fake_probability": 0.0,
    "anomaly_score":        0.0,
}

SCHEMA = set(STUB_OUTPUT.keys())

# ── SVM model path ─────────────────────────────────────────────────────────────
# Override via env var FREQ_SVM_MODEL_PATH or edit default below.
SVM_MODEL_PATH = os.environ.get(
    "FREQ_SVM_MODEL_PATH",
    "models/svm_freq_model_v3_facesHQ_celeba_ffpp.pkl"
)

# ── FFT band config (Durall 2020 / DFA-2025-TC-00471 §5.2) ───────────────────
# Band boundaries as fraction of max radial frequency
FFT_LOW_FRAC  = 0.15   # 0%  – 15%  → low band (1/f baseline)
FFT_MID_FRAC  = 0.50   # 15% – 50%  → mid band
FFT_HIGH_FRAC = 0.75   # 50% – 75%  → high band
                       # 75% – 100% → ultra (not scored separately)

# Expected dB drop from low_mean under natural 1/f rolloff (empirical, FF++ real set)
FFT_EXPECTED_MID_DROP  = 15.0   # mid  should be ~15 dB below low baseline
FFT_EXPECTED_HIGH_DROP = 35.0   # high should be ~35 dB below low baseline

# Normalisation denominators for converting dB → [0, 1]
# Reference: real ≈ 0 dB, StyleGAN2 → mid ≈ +9.4, high ≈ +13.3 (DFA-2025-TC-00471 §5.2)
FFT_MID_NORM_MAX  = 15.0
FFT_HIGH_NORM_MAX = 20.0

# ── Feature extractor config (must match training pipeline) ───────────────────
FEATURE_OUTPUT_SIZE = 300   # interpolated radial profile length — matches train_model.py


# =============================================================================
# Schema validation
# =============================================================================

def validate_output(output: dict) -> bool:
    if not SCHEMA.issubset(output.keys()):
        logger.error(f"Missing keys: {SCHEMA - output.keys()}")
        return False
    for key in ["svm_fake_probability", "anomaly_score"]:
        val = output.get(key, -1)
        if not (0.0 <= val <= 1.0):
            logger.error(f"{key} out of range [0,1]: {val}")
            return False
    return True


# =============================================================================
# Shared: azimuthal average
# Direct port of azimuthalAverage() from radialProfile.py
# (astrobetter.com / Durall 2020 codebase)
# Used by BOTH Method A and Method B.
# =============================================================================

def _azimuthal_average(image: np.ndarray, center=None) -> np.ndarray:
    """
    Azimuthally averaged radial profile.

    Args:
        image:  2D float array (log-magnitude spectrum, already fftshifted)
        center: [x, y] pixel — defaults to image center
    Returns:
        1D radial profile (mean value at each integer radius)
    """
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([
            (x.max() - x.min()) / 2.0,
            (y.max() - y.min()) / 2.0,
        ])

    r = np.hypot(x - center[0], y - center[1])

    ind      = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    r_int  = r_sorted.astype(int)
    deltar = r_int[1:] - r_int[:-1]
    rind   = np.where(deltar)[0]
    nr     = rind[1:] - rind[:-1]

    csim        = np.cumsum(i_sorted, dtype=float)
    tbin        = csim[rind[1:]] - csim[rind[:-1]]
    radial_prof = tbin / nr

    return radial_prof


# =============================================================================
# Shared: face preprocessing
# Mirrors preprocess_image() from preprocessing_freq.py exactly.
# Shared by both methods so both operate on the same face crop.
# =============================================================================

def _detect_and_crop_face(image_path: str):
    """
    Haar cascade face detection — mirrors preprocess_image() from
    preprocessing_freq.py exactly (same detector params, largest face selected,
    returns grayscale crop).

    Args:
        image_path: path to input image
    Returns:
        (face_gray, bbox) or (None, None) if no face detected
        bbox: (x1, y1, x2, y2)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, None

    # Largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    face = gray[y:y + h, x:x + w]
    bbox = (x, y, x + w, y + h)
    return face, bbox


# =============================================================================
# Shared: make_square
# Mirrors make_square() from feature_extractor.py exactly (mean padding).
# Both methods use this so FFT is computed on the same array shape.
# =============================================================================

def _make_square(img: np.ndarray) -> np.ndarray:
    h, w   = img.shape
    size   = max(h, w)
    padded = np.ones((size, size), dtype=np.float32) * np.mean(img)
    padded[:h, :w] = img
    return padded


# =============================================================================
# Method A — FFT Radial Spectrum (Durall CVPR 2020)
# =============================================================================

def _compute_fft_bands(face_gray: np.ndarray) -> dict:
    """
    Durall et al. CVPR 2020 — radial power spectrum analysis.

    Uses the SAME FFT formula as feature_extractor.py:
        magnitude = 20 * np.log(np.abs(fshift))
    so both methods are consistent on the same signal.

    Steps:
        1. make_square (mean padding)
        2. 2D FFT → fftshift → 20*log(|fshift|)  ← matches training pipeline
        3. Azimuthal average → 1D radial profile
        4. Split into low / mid / high bands
        5. Compute dB delta vs expected 1/f rolloff

    Args:
        face_gray: grayscale face crop (output of _detect_and_crop_face)
    Returns:
        fft_mid_anomaly_db  — positive = GAN artifact at mid frequencies
        fft_high_anomaly_db — positive = GAN artifact at high frequencies
        radial_profile      — full 1D profile list (preserved for explainability)
    """
    face = face_gray.astype(np.float32)
    face = _make_square(face)

    f         = np.fft.fft2(face)
    fshift    = np.fft.fftshift(f)
    magnitude = 20.0 * np.log(np.abs(fshift) + 1e-8)   # matches feature_extractor.py

    radial_prof = _azimuthal_average(magnitude)
    max_r       = len(radial_prof)

    low_end  = max(1, int(FFT_LOW_FRAC  * max_r))
    mid_end  = int(FFT_MID_FRAC  * max_r)
    high_end = int(FFT_HIGH_FRAC * max_r)

    low_mean  = radial_prof[:low_end].mean()            # 1/f baseline
    mid_mean  = radial_prof[low_end:mid_end].mean()
    high_mean = radial_prof[mid_end:high_end].mean()

    # dB delta vs expected 1/f rolloff
    # Positive → energy ABOVE natural baseline → GAN artifact
    # Near zero → natural rolloff              → real face
    fft_mid_anomaly_db  = float(mid_mean  - (low_mean - FFT_EXPECTED_MID_DROP))
    fft_high_anomaly_db = float(high_mean - (low_mean - FFT_EXPECTED_HIGH_DROP))

    return {
        "fft_mid_anomaly_db":  fft_mid_anomaly_db,
        "fft_high_anomaly_db": fft_high_anomaly_db,
        "radial_profile":      radial_prof.tolist(),    # preserved for explainability / viz
    }


# =============================================================================
# Method B — SVM on radial profile features (Frank ICML 2020 / arXiv 1911.00686)
# =============================================================================

_svm_cache = None  # lazy-load cache

def _load_svm():
    """Load (model, scaler) tuple from pickle — cached after first load."""
    global _svm_cache
    if _svm_cache is None:
        if not os.path.exists(SVM_MODEL_PATH):
            raise FileNotFoundError(
                f"SVM model not found: {SVM_MODEL_PATH}\n"
                f"Set env var FREQ_SVM_MODEL_PATH or copy model to that path."
            )
        with open(SVM_MODEL_PATH, "rb") as f:
            _svm_cache = pickle.load(f)
        logger.info(f"SVM model loaded: {SVM_MODEL_PATH}")
    return _svm_cache


def _extract_features(face_gray: np.ndarray, output_size: int = FEATURE_OUTPUT_SIZE):
    """
    Mirrors extract_features() from feature_extractor.py exactly:
        1. float32 cast
        2. make_square (mean padding)
        3. 2D FFT → fftshift → 20*log(|fshift|)
        4. azimuthalAverage → 1D radial profile
        5. interpolate to output_size (linear)
        6. normalize by DC component (index 0)

    Args:
        face_gray:   grayscale face crop
        output_size: feature vector length — must match training (default 300)
    Returns:
        1D np.ndarray (length=output_size) or None if radial profile too short
    """
    epsilon = 1e-8

    face = face_gray.astype(np.float32)
    face = _make_square(face)

    f         = np.fft.fft2(face)
    fshift    = np.fft.fftshift(f)
    magnitude = 20.0 * np.log(np.abs(fshift) + epsilon)

    radial_prof = _azimuthal_average(magnitude)

    if len(radial_prof) < 10:
        return None

    x_old = np.linspace(0, 1, len(radial_prof))
    x_new = np.linspace(0, 1, output_size)
    interp         = interp1d(x_old, radial_prof, kind="linear")
    radial_resized = interp(x_new)

    radial_resized = radial_resized / (radial_resized[0] + epsilon)  # normalize by DC

    return radial_resized


def _run_svm(face_gray: np.ndarray) -> float:
    """
    Frank et al. ICML 2020 — SVM inference on radial FFT profile features.

    Args:
        face_gray: grayscale face crop
    Returns:
        svm_fake_probability [0, 1]  (class 1 = fake)
    """
    model, scaler = _load_svm()

    feat = _extract_features(face_gray)
    if feat is None:
        logger.warning("Feature extraction returned None — face crop too small.")
        return 0.5  # uncertain fallback

    feat_scaled = scaler.transform(feat.reshape(1, -1))
    proba       = model.predict_proba(feat_scaled)[0]   # [p_real, p_fake]
    return float(proba[1])                               # p_fake


# =============================================================================
# Agent entrypoint
# =============================================================================

def run(input: dict) -> dict:
    """
    Frequency Agent (L3) — FFT radial spectrum + SVM.

    Method A (Durall CVPR 2020):
        Computes fft_mid_anomaly_db and fft_high_anomaly_db — physically
        interpretable dB excess values preserved for the explainability /
        report layer.

    Method B (Frank ICML 2020 / arXiv 1911.00686):
        Runs the exact same preprocessing + feature extraction pipeline the
        SVM was trained on to produce svm_fake_probability.

    Both methods share the same face crop and the same FFT formula
    (20*log(|fftshift|)) for consistency.

    Args:
        input: dict with keys:
            input_type  : "image"
            path        : path to original image (face detection runs here)
    Returns:
        dict matching SCHEMA with anomaly_score in [0, 1]
    """
    if STUB_MODE:
        return STUB_OUTPUT.copy()

    image_path = input["path"]

    # ── Shared face detection ──────────────────────────────────────────────────
    face_gray, bbox = _detect_and_crop_face(image_path)

    if face_gray is None:
        logger.warning(f"No face detected: {image_path} — returning neutral scores.")
        return {
            "fft_mid_anomaly_db":   0.0,
            "fft_high_anomaly_db":  0.0,
            "svm_fake_probability": 0.5,
            "anomaly_score":        0.5,
        }

    if min(face_gray.shape) < 64:
        logger.warning(f"Face crop too small {face_gray.shape} — returning neutral scores.")
        return {
            "fft_mid_anomaly_db":   0.0,
            "fft_high_anomaly_db":  0.0,
            "svm_fake_probability": 0.5,
            "anomaly_score":        0.5,
        }

    # ── Method A: FFT bands ────────────────────────────────────────────────────
    fft_result          = _compute_fft_bands(face_gray)
    fft_mid_anomaly_db  = fft_result["fft_mid_anomaly_db"]
    fft_high_anomaly_db = fft_result["fft_high_anomaly_db"]

    # ── Method B: SVM ─────────────────────────────────────────────────────────
    svm_fake_probability = _run_svm(face_gray)

    # ── Fusion ────────────────────────────────────────────────────────────────
    # fft_mid / fft_high are explainability outputs — not direct voters.
    # Convert dB → [0,1] norm for their contribution to anomaly_score.
    mid_norm  = float(np.clip(fft_mid_anomaly_db  / FFT_MID_NORM_MAX,  0.0, 1.0))
    high_norm = float(np.clip(fft_high_anomaly_db / FFT_HIGH_NORM_MAX, 0.0, 1.0))
    fft_norm  = 0.4 * mid_norm + 0.6 * high_norm   # inner fusion of two bands

    anomaly_score = float(np.clip(
        0.40 * fft_norm + 0.60 * svm_fake_probability,
        0.0, 1.0
    ))

    output = {
        "fft_mid_anomaly_db":   fft_mid_anomaly_db,
        "fft_high_anomaly_db":  fft_high_anomaly_db,
        "svm_fake_probability": svm_fake_probability,
        "anomaly_score":        anomaly_score,
    }

    assert validate_output(output), f"Schema validation failed: {output}"
    return output