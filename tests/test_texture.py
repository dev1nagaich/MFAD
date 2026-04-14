#!/usr/bin/env python3
"""
test_texture_texvit.py — Test Suite for Tex-ViT Texture Agent
==============================================================

Tests the texture agent on:
  1. A synthetic AUTHENTIC face (smooth, consistent skin texture)
  2. A synthetic DEEPFAKE face  (two different textures blended at jaw)
  3. A real image from disk (optional — pass path as argument)

Also shows WHERE to get real test images (datasets + download helpers).

Usage:
    python test_texture_texvit.py                    # synthetic tests only
    python test_texture_texvit.py path/to/face.jpg   # also test a real image

Dependencies:
    pip install opencv-python-headless scikit-image scipy pydantic numpy
    pip install torch torchvision   # optional but strongly recommended
"""

import sys
import os
import json
import tempfile

import cv2
import numpy as np

# ── Import the agent ─────────────────────────────────────────────────────────
# Adjust path if needed — add parent (MFAD) directory, not tests directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.texture import run_texture_agent, run_texture_agent_stub, TextureOutput


# =============================================================================
# SYNTHETIC IMAGE GENERATORS
# =============================================================================

def _perlin_like_noise(h: int, w: int, scale: float = 20.0) -> np.ndarray:
    """Generate smooth noise resembling natural skin texture."""
    x = np.linspace(0, scale, w)
    y = np.linspace(0, scale, h)
    xx, yy = np.meshgrid(x, y)
    noise = (np.sin(xx) * np.cos(yy)
             + 0.5 * np.sin(2 * xx + 1.3) * np.cos(2 * yy + 0.7)
             + 0.25 * np.sin(4 * xx + 2.1) * np.cos(4 * yy + 1.5))
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def make_authentic_face(size: int = 256) -> tuple[np.ndarray, list]:
    """
    Create a synthetic image that looks like a genuine photo:
    - Smooth, globally consistent skin tone
    - Natural texture variation (Perlin-like noise)
    - No blending seam at jaw

    Returns (image_rgb, face_bbox)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background
    img[:] = [120, 100, 90]

    # Face region: consistent warm skin tone with smooth texture
    face_x1, face_y1 = size // 6, size // 8
    face_x2, face_y2 = size * 5 // 6, size * 7 // 8
    fh = face_y2 - face_y1
    fw = face_x2 - face_x1

    # Base skin tone
    skin_base = np.array([210, 170, 140], dtype=float)

    # Consistent texture across entire face
    noise = _perlin_like_noise(fh, fw, scale=8.0)
    for c, base in enumerate(skin_base):
        channel = np.clip(base + (noise - 0.5) * 30, 0, 255).astype(np.uint8)
        img[face_y1:face_y2, face_x1:face_x2, c] = channel

    # Simulate lighting gradient (authentic)
    gradient = np.linspace(1.05, 0.92, fh)[:, np.newaxis]
    face_region = img[face_y1:face_y2, face_x1:face_x2].astype(float)
    face_region = np.clip(face_region * gradient[:, :, np.newaxis], 0, 255)
    img[face_y1:face_y2, face_x1:face_x2] = face_region.astype(np.uint8)

    # Neck (same skin tone, slightly darker)
    neck_y1, neck_y2 = face_y2, min(face_y2 + 40, size)
    neck_noise = _perlin_like_noise(neck_y2 - neck_y1, fw, scale=8.0)
    for c, base in enumerate(skin_base * 0.90):
        channel = np.clip(base + (neck_noise - 0.5) * 20, 0, 255).astype(np.uint8)
        img[neck_y1:neck_y2, face_x1:face_x2, c] = channel

    bbox = [face_x1, face_y1, face_x2, face_y2]
    return img, bbox


def make_deepfake_face(size: int = 256) -> tuple[np.ndarray, list]:
    """
    Create a synthetic image that looks like a face-swap deepfake:
    - Upper face: warm skin tone (original body)
    - Lower face / jaw: cooler, different skin tone (swapped face)
    - Sharp texture discontinuity at ~70% of face height
    - Different noise frequency simulating GAN upsampling artefacts

    Returns (image_rgb, face_bbox)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = [120, 100, 90]

    face_x1, face_y1 = size // 6, size // 8
    face_x2, face_y2 = size * 5 // 6, size * 7 // 8
    fh = face_y2 - face_y1
    fw = face_x2 - face_x1

    # ── Upper face: original body skin (warm tone, coarse texture) ─────────
    upper_h = int(fh * 0.68)
    upper = img[face_y1:face_y1 + upper_h, face_x1:face_x2]
    skin_upper = np.array([215, 172, 138], dtype=float)
    noise_upper = _perlin_like_noise(upper_h, fw, scale=7.0)
    for c, base in enumerate(skin_upper):
        upper[:, :, c] = np.clip(base + (noise_upper - 0.5) * 35, 0, 255)
    img[face_y1:face_y1 + upper_h, face_x1:face_x2] = upper

    # ── Lower face / jaw: swapped face skin (cooler, GAN-smooth texture) ───
    lower_h = fh - upper_h
    lower = img[face_y1 + upper_h:face_y2, face_x1:face_x2]
    skin_lower = np.array([190, 165, 155], dtype=float)   # cooler / different tone
    # GAN skin: higher frequency but smoother (less natural variation)
    noise_lower = _perlin_like_noise(lower_h, fw, scale=18.0)  # finer, smoother
    for c, base in enumerate(skin_lower):
        lower[:, :, c] = np.clip(base + (noise_lower - 0.5) * 12, 0, 255)  # less variance
    img[face_y1 + upper_h:face_y2, face_x1:face_x2] = lower

    # ── Neck: continuation of original body (warm, mismatches jaw) ─────────
    neck_y1, neck_y2 = face_y2, min(face_y2 + 40, size)
    neck_noise = _perlin_like_noise(neck_y2 - neck_y1, fw, scale=7.0)
    for c, base in enumerate(skin_upper * 0.90):
        img[neck_y1:neck_y2, face_x1:face_x2, c] = np.clip(
            base + (neck_noise - 0.5) * 28, 0, 255)

    bbox = [face_x1, face_y1, face_x2, face_y2]
    return img, bbox


# =============================================================================
# TEST RUNNER
# =============================================================================

def _save_temp(img_rgb: np.ndarray) -> str:
    """Save RGB image to a temp file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return tmp.name


def print_result(label: str, result: TextureOutput, expected: str):
    """Pretty-print a single test result."""
    verdict = "✅ DEEPFAKE" if result.seam_detected else "❌ AUTHENTIC"
    score_bar = "█" * int(result.anomaly_score * 20) + "░" * (20 - int(result.anomaly_score * 20))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Expected: {expected}")
    print(f"  Verdict : {verdict}")
    print(f"{'='*60}")
    print(f"  Anomaly score : {result.anomaly_score:.3f}  [{score_bar}]")
    print(f"  Jaw dist      : {result.jaw_emd:.4f}  (jaw-cheek boundary)")
    print(f"  Neck dist     : {result.neck_emd:.4f}  (neck-jaw boundary)")
    print(f"  Cheek dist    : {result.cheek_emd:.4f}  (L/R symmetry)")
    print(f"  LBP uniformity: {result.lbp_uniformity:.3f}")
    print(f"  Scale consist : {result.multi_scale_consistency:.3f}")
    print()
    print("  Zone anomaly scores:")
    for zone, score in result.zone_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"    {zone:10s}: {score:.3f}  [{bar}]")
    print()
    print("  Gram distances (all pairs):")
    for pair, dist in result.gram_distances.items():
        print(f"    {pair:28s}: {dist:.4f}")


def run_tests(real_image_path: str = None):
    print("\n" + "="*60)
    print("  TEX-VIT TEXTURE AGENT — TEST SUITE")
    print("="*60)

    # ── Test 1: Authentic face ───────────────────────────────────────────────
    print("\n[1/3] Generating synthetic AUTHENTIC face...")
    auth_img, auth_bbox = make_authentic_face(256)
    auth_path = _save_temp(auth_img)
    try:
        result_auth = run_texture_agent(auth_path, auth_bbox)
        print_result("TEST 1 — AUTHENTIC (synthetic)", result_auth,
                     "seam_detected=False, low anomaly_score")
    finally:
        os.unlink(auth_path)

    # ── Test 2: Deepfake face ────────────────────────────────────────────────
    print("\n[2/3] Generating synthetic DEEPFAKE face...")
    fake_img, fake_bbox = make_deepfake_face(256)
    fake_path = _save_temp(fake_img)
    try:
        result_fake = run_texture_agent(fake_path, fake_bbox)
        print_result("TEST 2 — DEEPFAKE (synthetic)", result_fake,
                     "seam_detected=True, high anomaly_score")
    finally:
        os.unlink(fake_path)

    # ── Test 3: Real image from disk (optional) ──────────────────────────────
    if real_image_path:
        print(f"\n[3/3] Testing REAL image: {real_image_path}")
        # Auto-detect face bbox with OpenCV Haar cascade
        img_bgr = cv2.imread(real_image_path)
        if img_bgr is None:
            print(f"  ERROR: Cannot read {real_image_path}")
        else:
            # Try to auto-detect face
            detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                bbox = [x, y, x + w, y + h]
                print(f"  Auto-detected face bbox: {bbox}")
            else:
                # Use full image as face
                ih, iw = img_bgr.shape[:2]
                bbox = [0, 0, iw, ih]
                print(f"  No face detected — using full image: {bbox}")

            result_real = run_texture_agent(real_image_path, bbox)
            print_result("TEST 3 — REAL IMAGE", result_real,
                         "unknown (check against ground truth)")
    else:
        print("\n[3/3] Skipped — no real image path provided.")
        print("       Run: python test_texture_texvit.py path/to/face.jpg")

    # ── Stub sanity check ────────────────────────────────────────────────────
    print("\n[STUB CHECK]")
    stub_result = run_texture_agent_stub("dummy.jpg", [0, 0, 200, 200])
    print(f"  Stub anomaly_score: {stub_result.anomaly_score} (expected ~0.884)")
    print(f"  Stub seam_detected: {stub_result.seam_detected} (expected True)")

    print("\n✅ All tests complete.\n")


# =============================================================================
# WHERE TO GET REAL TEST IMAGES
# =============================================================================

DATASET_INFO = """
═══════════════════════════════════════════════════════════════════════════════
  WHERE TO GET REAL DEEPFAKE TEST IMAGES
═══════════════════════════════════════════════════════════════════════════════

1. FaceForensics++ (FF++)  ← RECOMMENDED for your project
   - Paper's primary benchmark dataset
   - Contains: DeepFakes, Face2Face, FaceSwap, NeuralTextures
   - Request access: https://github.com/ondyari/FaceForensics
   - Fill out the form at: http://kaldir.vc.in.tum.de/faceforensics/
   - You'll get a download script; run it to get images/videos

2. Celeb-DF v2  ← Good for testing on celebrity faces (like Tom Cruise)
   - Link: https://github.com/yuezunli/celeb-deepfakeforensics
   - Contains 590 YouTube videos + 5639 deepfake videos of celebrities
   - Direct download available (no form needed)
   - wget command after accepting terms on their GitHub page

3. DFDC (DeepFake Detection Challenge)
   - Kaggle dataset: https://www.kaggle.com/c/deepfake-detection-challenge/data
   - Requires Kaggle account + API key
   - 470 GB total — use a small subset:
     kaggle competitions download -c deepfake-detection-challenge
     --file metadata.json  # start with just labels

4. 140k Real vs Fake Faces (Kaggle)  ← EASIEST TO GET STARTED
   - https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
   - StyleGAN-generated fake faces vs real FFHQ faces
   - Simple binary classification, easy to download:
     kaggle datasets download -d xhlulu/140k-real-and-fake-faces

5. FFHQ (real faces only — for authentic baseline)
   - https://github.com/NVlabs/ffhq-dataset
   - 70,000 high-quality real face images
   - Good for testing authentic = low anomaly_score

QUICK START for testing your agent RIGHT NOW:
─────────────────────────────────────────────
  # Option A: Celeb-DF (no sign-up needed)
  git clone https://github.com/yuezunli/celeb-deepfakeforensics
  # Download sample frames from their provided link

  # Option B: 140k Kaggle dataset
  pip install kaggle
  kaggle datasets download -d xhlulu/140k-real-and-fake-faces
  unzip 140k-real-and-fake-faces.zip -d test_faces/

  # Then test:
  python test_texture_texvit.py test_faces/fake/00000.jpg
  python test_texture_texvit.py test_faces/real/00000.jpg

  # Option C: Single image from FF++ paper examples
  # Any face image + OpenCV auto-detects bbox — works immediately

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(DATASET_INFO)
    real_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tests(real_image_path=real_path)