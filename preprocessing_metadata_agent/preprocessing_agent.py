"""
preprocessing_agent.py
======================
PURPOSE : Compute image integrity data — hashes, face detection,
          normalisation, ELA score.
          Does NOT give any verdict. Just returns raw findings.

INPUT   : image_path (str)
OUTPUT  : saves  outputs/preprocessing/<stem>.json
          returns that JSON file path (str)

JSON output:
{
    "image_path"          : "/abs/path/to/image.jpg",
    "face_bbox"           : [x1, y1, x2, y2],       <- primary (largest) face
    "face_bboxes"         : [[x1,y1,x2,y2], ...],   <- ALL detected faces
    "face_count"          : 2,
    "face_detected"       : true,
    "image_dims"          : [1920, 1080],
    "hash_sha256"         : "a3f9b2...",
    "hash_md5"            : "c4e8a1...",
    "ela_score"           : 0.34,
    "normalised_img_path" : "outputs/preprocessing/photo1_normalised.jpg",
    "exif_metadata"       : { ... },
    "errors"              : []
}

Dependencies:
    pip install Pillow piexif numpy opencv-python langchain langchain-core pydantic
    pip install retina-face mediapipe   # optional — better face detection
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Type

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# ── optional face detection backends ─────────────────────────────────────────
try:
    from retinaface import RetinaFace
    _HAS_RETINA = True
except ImportError:
    _HAS_RETINA = False

try:
    import mediapipe as mp
    _HAS_MP = True
except ImportError:
    _HAS_MP = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import piexif
    _HAS_PIEXIF = True
except ImportError:
    _HAS_PIEXIF = False
    warnings.warn("piexif not installed — EXIF metadata skipped.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("PreprocessingAgent")

OUTPUT_DIR  = Path("outputs/preprocessing")
TARGET_SIZE = (512, 512)
ELA_QUALITY = 75          # lower quality → larger diff → easier to spot edits
ELA_AMPLIFY = 10.0        # amplification for visualisation


# ═════════════════════════════════════════════════════════════════════════════
#  1. HASHING  — chain of custody
# ═════════════════════════════════════════════════════════════════════════════

def _compute_hashes(path: str) -> tuple[str, str]:
    """Compute SHA-256 and MD5 over raw file bytes."""
    sha, md = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            sha.update(chunk)
            md.update(chunk)
    return sha.hexdigest(), md.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
#  2. EXIF  — basic metadata
# ═════════════════════════════════════════════════════════════════════════════

def _extract_exif(path: str) -> dict:
    """Extract readable EXIF tags. Returns empty dict if unavailable."""
    if not _HAS_PIEXIF:
        return {}
    try:
        exif_dict = piexif.load(path)
    except Exception:
        return {}
    out = {}
    for ifd in ("0th", "Exif", "GPS"):
        for tag_id, val in exif_dict.get(ifd, {}).items():
            name = piexif.TAGS[ifd].get(tag_id, {}).get("name", str(tag_id))
            if isinstance(val, bytes):
                try:
                    val = val.decode("utf-8", errors="replace").strip("\x00")
                except Exception:
                    val = val.hex()
            out[name] = val
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  3. FACE DETECTION  — all backends, return all faces sorted largest first
# ═════════════════════════════════════════════════════════════════════════════

def _sort_by_area(bboxes: list[list[int]]) -> list[list[int]]:
    return sorted(bboxes,
                  key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                  reverse=True)


def _detect_retinaface(path: str) -> list[list[int]]:
    try:
        faces = RetinaFace.detect_faces(path)
        if not isinstance(faces, dict) or not faces:
            return []
        bboxes = []
        for face in faces.values():
            x1, y1, x2, y2 = face["facial_area"]
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        return _sort_by_area(bboxes)
    except Exception as e:
        log.warning("RetinaFace: %s", e)
        return []


def _detect_mediapipe(img: Image.Image) -> list[list[int]]:
    try:
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=1,
                                   min_detection_confidence=0.5) as det:
            arr    = np.array(img.convert("RGB"))
            result = det.process(arr)
        if not result.detections:
            return []
        h, w   = arr.shape[:2]
        bboxes = []
        for d in result.detections:
            bb = d.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width)  * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
        return _sort_by_area(bboxes)
    except Exception as e:
        log.warning("MediaPipe: %s", e)
        return []


def _detect_opencv(img: Image.Image) -> list[list[int]]:
    if not _HAS_CV2:
        return []
    try:
        casc  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray  = np.array(img.convert("L"))
        faces = casc.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(60, 60))
        if not len(faces):
            return []
        bboxes = [[int(x), int(y), int(x+w), int(y+h)]
                  for x, y, w, h in faces]
        return _sort_by_area(bboxes)
    except Exception as e:
        log.warning("OpenCV: %s", e)
        return []


def _detect_all_faces(path: str,
                      img: Image.Image) -> tuple[list[list[int]], bool]:
    """
    Try RetinaFace → MediaPipe → OpenCV.
    Returns (face_bboxes, face_detected).
    If nothing found, returns full image bbox and face_detected=False.
    """
    bboxes: list[list[int]] = []

    if _HAS_RETINA:
        bboxes = _detect_retinaface(path)
        if bboxes:
            log.info("  RetinaFace: %d face(s)", len(bboxes))

    if not bboxes and _HAS_MP:
        bboxes = _detect_mediapipe(img)
        if bboxes:
            log.info("  MediaPipe: %d face(s)", len(bboxes))

    if not bboxes and _HAS_CV2:
        bboxes = _detect_opencv(img)
        if bboxes:
            log.info("  OpenCV: %d face(s)", len(bboxes))

    if not bboxes:
        log.warning("  No face detected — using full image bbox")
        w, h = img.size
        return [[0, 0, w, h]], False

    return bboxes, True


# ═════════════════════════════════════════════════════════════════════════════
#  4. NORMALISATION
# ═════════════════════════════════════════════════════════════════════════════

def _normalise(img: Image.Image, bbox: list[int]) -> Image.Image:
    """
    Crop to face bbox with 15% padding on each side,
    convert to RGB, resize to 512×512 using Lanczos.
    """
    x1, y1, x2, y2 = bbox
    W, H = img.size
    px   = int((x2 - x1) * 0.15)
    py   = int((y2 - y1) * 0.15)
    x1   = max(0, x1 - px);  y1 = max(0, y1 - py)
    x2   = min(W, x2 + px);  y2 = min(H, y2 + py)
    return img.crop((x1, y1, x2, y2)).convert("RGB").resize(
        TARGET_SIZE, Image.LANCZOS)


# ═════════════════════════════════════════════════════════════════════════════
#  5. ELA  — Error Level Analysis
# ═════════════════════════════════════════════════════════════════════════════

def _compute_ela(img: Image.Image) -> float:
    """
    Re-save image at ELA_QUALITY and compute mean absolute pixel difference.

    Formula:
        1. Save img as JPEG at quality=75
        2. Reload the saved JPEG
        3. diff = |original - reloaded|  per pixel per channel
        4. ela_score = mean(diff) / 255 * amplify,  clamped to [0, 1]

    Interpretation:
        Unedited regions compress consistently → small diff → low score
        Edited/spliced regions have inconsistent compression → large diff → high score

    Range: 0.0 (clean) → 1.0 (heavily altered)

    NOTE: This score is most meaningful on original JPEG files.
          PNG or heavily recompressed files (WhatsApp) will show inflated scores.
    """
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    diff      = ImageChops.difference(img.convert("RGB"),
                                      Image.open(buf).convert("RGB"))
    diff_arr  = np.array(diff, dtype=np.float32)
    ela_score = float(np.mean(diff_arr) / 255.0) * ELA_AMPLIFY
    return round(min(1.0, ela_score), 6)


# ═════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def run_preprocessing(image_path: str) -> str:
    """
    Run preprocessing on one image.
    Saves JSON to outputs/preprocessing/<stem>.json.
    Returns the JSON file path.

    This agent only MEASURES — it does not score or give a verdict.
    The anomaly scoring and verdict happen in pipeline.py after
    all agents have run.
    """
    errors   = []
    abs_path = str(Path(image_path).resolve())

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Image not found: {abs_path}")

    log.info("▶ preprocessing  |  %s", Path(abs_path).name)

    # 1. hashes (on raw bytes before any processing)
    sha256, md5 = _compute_hashes(abs_path)
    log.info("  SHA-256: %s…", sha256[:16])

    # 2. open image
    img = Image.open(abs_path)
    img.load()
    image_dims = list(img.size)   # [W, H]
    log.info("  dims: %s", image_dims)

    # 3. EXIF
    exif_meta = _extract_exif(abs_path)

    # 4. face detection — ALL faces
    face_bboxes, face_detected = _detect_all_faces(abs_path, img)
    face_count = len(face_bboxes) if face_detected else 0
    face_bbox  = face_bboxes[0]   # primary = largest
    log.info("  faces=%d  primary=%s", face_count, face_bbox)

    # 5. ELA score — MUST run on original image before any resizing.
    #    Resizing interpolates pixels and destroys the compression artifacts
    #    that ELA relies on. Always use the raw opened image here.
    try:
        ela_score = _compute_ela(img)
        log.info("  ela_score: %.6f  (on original %s image)", ela_score, image_dims)
    except Exception as e:
        errors.append(f"ela: {e}")
        ela_score = 0.0

    # 6. normalise on primary face — for downstream DL models ONLY.
    #    NOT used for any forensic analysis.
    try:
        norm_img = _normalise(img, face_bbox)
    except Exception as e:
        errors.append(f"normalise: {e}")
        norm_img = img.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)

    # 7. save normalised image (downstream agents use this)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem          = Path(abs_path).stem
    norm_img_path = str(OUTPUT_DIR / f"{stem}_normalised.jpg")
    norm_img.save(norm_img_path, quality=95)

    # ── output dict — raw measurements only, no scoring ───────────────────────
    result = {
        "image_path":          abs_path,
        "face_bbox":           face_bbox,       # [x1,y1,x2,y2] primary face
        "face_bboxes":         face_bboxes,     # all faces
        "face_count":          face_count,
        "face_detected":       face_detected,
        "image_dims":          image_dims,      # [W, H] original
        "hash_sha256":         sha256,          # chain of custody
        "hash_md5":            md5,             # chain of custody
        "ela_score":           ela_score,       # 0=clean, 1=heavily altered
        "anomaly_score":       ela_score,       # preprocessing anomaly = ela_score
        "normalised_img_path": norm_img_path,   # 512×512 face crop for DL models
        "exif_metadata":       exif_meta,
        "errors":              errors,
    }

    json_path = str(OUTPUT_DIR / f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("✔ preprocessing done  |  json → %s", json_path)
    return json_path


# ═════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOL
# ═════════════════════════════════════════════════════════════════════════════

class PreprocessingInput(BaseModel):
    image_path: str = Field(
        description="Path to the image file to analyse.")


class PreprocessingAgent(BaseTool):
    """
    LangChain tool — always run this FIRST.
    Input  : image_path
    Output : path to JSON file (pass to all subsequent agents)
    """
    name: str = "preprocessing_agent"
    description: str = (
        "Always run this tool FIRST on any image. "
        "Computes SHA-256/MD5 hashes, detects all face bounding boxes, "
        "normalises image to 512x512, computes ELA score (0=clean, 1=altered). "
        "Input: image_path (str). "
        "Output: path to JSON file with keys: "
        "image_path, face_bbox, face_bboxes, face_count, face_detected, "
        "image_dims, hash_sha256, hash_md5, ela_score. "
        "Pass the returned JSON path to all other agents."
    )
    args_schema: Type[BaseModel] = PreprocessingInput

    def _run(self, image_path: str) -> str:
        return run_preprocessing(image_path)

    async def _arun(self, image_path: str) -> str:
        return self._run(image_path)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessing_agent.py <image_path>")
        sys.exit(1)
    out = run_preprocessing(sys.argv[1])
    print(f"\nJSON → {out}\n")
    with open(out) as f:
        data = json.load(f)
    # print without exif clutter
    display = {k: v for k, v in data.items() if k != "exif_metadata"}
    print(json.dumps(display, indent=2))
    print(f"\nFaces : {data['face_count']}")
    for i, b in enumerate(data["face_bboxes"]):
        tag = " ← primary" if i == 0 else ""
        print(f"  Face {i+1} : {b}  ({b[2]-b[0]}×{b[3]-b[1]} px){tag}")
