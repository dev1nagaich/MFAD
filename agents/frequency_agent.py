# agents/frequency_agent.py
# Deep Sentinel — Frequency Agent (L3)
# Pure-math implementation: FFT radial spectrum + Block DCT
# No deep learning model. No GPU required.
#
# Techniques sourced from:
#   FFT azimuthal averaging:
#     Durall et al. CVPR 2020 — "Watch Your Up-Convolution"
#     arXiv: 2005.06803
#     (band definitions and dB thresholds are empirical, not from this paper)
#   Block DCT decomposition:
#     Frank et al. ICML 2020 — "Leveraging Frequency Analysis for Deep Fake Image Recognition"
#     arXiv: 2003.08685
#     (hf_ratio thresholds are empirical, not from this paper)

import json
import numpy as np
import cv2
from pathlib import Path
from scipy.fftpack import dct as scipy_dct

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config.json"
with open(_CONFIG_PATH) as f:
    _CFG = json.load(f)["frequency_agent"]

_FFT  = _CFG["fft_bands"]
_FTHR = _CFG["fft_thresholds"]
_DTHR = _CFG["dct_thresholds"]
_W    = _CFG["submethod_weights"]

# ── Schema ────────────────────────────────────────────────────────────────────

STUB_MODE = False

STUB_OUTPUT = {
    "fft_mid_anomaly_db":  0.0,
    "fft_high_anomaly_db": 0.0,
    "anomaly_score":       0.0,
}

SCHEMA = STUB_OUTPUT.keys()


def validate_output(output: dict) -> bool:
    """Check all 3 schema fields are present.
    fft fields: float, >= 0.
    anomaly_score: float, in [0, 1]."""
    for k in ["fft_mid_anomaly_db", "fft_high_anomaly_db"]:
        if k not in output:
            return False
        if not isinstance(output[k], float):
            return False
        if output[k] < 0.0:
            return False
    if "anomaly_score" not in output:
        return False
    if not isinstance(output["anomaly_score"], float):
        return False
    if not (0.0 <= output["anomaly_score"] <= 1.0):
        return False
    return True


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run(input: dict) -> dict:
    """
    Frequency Agent: Detects GAN upsampling artifacts in the frequency domain.

    Two pure-math methods:
      A) FFT radial spectrum (Durall et al. CVPR 2020 — technique)
         Azimuthal averaging of 2D FFT magnitude. Measures energy excess
         at mid and high frequency bands relative to a baseline offset.
         Band definitions and dB offsets are empirically calibrated.
      B) Block DCT 8x8 (Frank et al. ICML 2020 — technique)
         Measures ratio of high-frequency to low-frequency AC coefficient
         energy across 8x8 blocks. Thresholds are empirically calibrated.

    anomaly_score = clip((fft_weight * fft_normalised) + (dct_weight * dct_score), 0, 1)
    Weights: fft=0.55, dct=0.45

    Args:
        input: dict with keys: input_type (str), path (str — face_crop_path)
    Returns:
        dict with fft_mid_anomaly_db, fft_high_anomaly_db, anomaly_score.
        No exceptions raised — handled upstream by safe_run().
    """
    if STUB_MODE:
        return STUB_OUTPUT.copy()

    image_path = input["path"]

    fft_result = _run_fft_analysis(image_path)
    dct_result = _run_dct_analysis(image_path)

    # Normalise raw dB excess to [0, 1].
    # High-band weighted 0.7 within FFT because it showed signal on StyleGAN2
    # in v1 (3.34 dB) while mid-band returned 0.0.
    # normalization_factor=10.0 (not 20.0) — at 20.0, fft_high=3.34 only
    # contributes 0.064 to anomaly_score, forcing DCT to score >=0.97 alone.
    # At 10.0, FFT contributes 0.129 and DCT needs >=0.825 — achievable.
    norm_factor = _FTHR["normalization_factor"]
    fft_normalised = float(np.clip(
        (0.3 * fft_result["fft_mid_anomaly_db"] +
         0.7 * fft_result["fft_high_anomaly_db"]) / norm_factor,
        0.0, 1.0
    ))

    anomaly_score = float(np.clip(
        _W["fft"] * fft_normalised + _W["dct"] * dct_result["dct_score"],
        0.0, 1.0
    ))

    output = {
        "fft_mid_anomaly_db":  fft_result["fft_mid_anomaly_db"],
        "fft_high_anomaly_db": fft_result["fft_high_anomaly_db"],
        "anomaly_score":       anomaly_score,
    }

    assert validate_output(output), f"Schema validation failed: {output}"
    return output


# ── Method A: FFT Radial Spectrum ─────────────────────────────────────────────

def _run_fft_analysis(image_path: str) -> dict:
    """
    Azimuthal radial averaging of 2D FFT magnitude spectrum.
    Technique: Durall et al. CVPR 2020.

    GAN generators fail to reproduce natural spectral distributions.
    This produces measurable energy excess at mid and high frequency bands.

    Band boundaries and dB offset thresholds are empirically set in config.json
    and must be calibrated on FF++ data before production use.
    See scripts/calibrate_fft.py.

    Returns raw dB excess values (float >= 0). Zero means no anomaly detected
    at that band given current thresholds.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_CFG["image_size"], _CFG["image_size"])).astype(np.float32)

    f         = np.fft.fft2(img)
    fshift    = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag   = 20.0 * np.log10(magnitude + 1e-8)

    h, w   = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X   = np.ogrid[:h, :w]
    R      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r  = min(cy, cx)

    radial_mean = np.array([
        log_mag[R == r].mean() if (R == r).any() else 0.0
        for r in range(max_r)
    ])

    low_end  = int(_FFT["low_pct"]  * max_r)
    mid_end  = int(_FFT["mid_pct"]  * max_r)
    high_end = int(_FFT["high_pct"] * max_r)

    low_energy  = radial_mean[:low_end].mean()
    mid_energy  = radial_mean[low_end:mid_end].mean()
    high_energy = radial_mean[mid_end:high_end].mean()

    expected_mid  = low_energy + _FTHR["mid_expected_offset"]
    expected_high = low_energy + _FTHR["high_expected_offset"]

    return {
        "fft_mid_anomaly_db":  float(max(0.0, mid_energy  - expected_mid)),
        "fft_high_anomaly_db": float(max(0.0, high_energy - expected_high)),
    }


# ── Method B: Block DCT ───────────────────────────────────────────────────────

def _run_dct_analysis(image_path: str) -> dict:
    """
    Block 8x8 DCT coefficient distribution analysis.
    Technique: Frank et al. ICML 2020.

    GAN images show elevated energy at high-frequency AC positions (32-63)
    relative to low-frequency positions (1-15) in each 8x8 DCT block.

    hf_ratio = mean(|AC 32-63|) / mean(|AC 1-15|)

    Thresholds real_hf_ratio_min and fake_hf_ratio_max are empirically set
    in config.json and must be calibrated on FF++ data before production use.
    See scripts/calibrate_dct.py (already exists in repo).

    Returns dct_score in [0, 1] — used internally by run() only.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (_CFG["image_size"], _CFG["image_size"])).astype(np.float32)

    block_size = _DTHR["block_size"]
    h, w       = img.shape
    dct_coeffs = []

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block     = img[i:i + block_size, j:j + block_size]
            block_dct = scipy_dct(scipy_dct(block.T, norm="ortho").T, norm="ortho")
            dct_coeffs.append(block_dct.flatten())

    dct_matrix  = np.array(dct_coeffs)
    coeff_means = np.abs(dct_matrix).mean(axis=0)

    hf_energy = coeff_means[32:].mean()
    lf_energy = coeff_means[1:16].mean()
    hf_ratio  = hf_energy / (lf_energy + 1e-8)

    real_min  = _DTHR["real_hf_ratio_min"]
    fake_max  = _DTHR["fake_hf_ratio_max"]

    return {
        "dct_score": float(np.clip(
            (hf_ratio - real_min) / (fake_max - real_min + 1e-8),
            0.0, 1.0
        ))
    }
