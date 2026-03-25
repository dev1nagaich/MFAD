"""
metadata_agent.py
=================
PURPOSE : Compute metadata forensic signals — EXIF, software tag,
          ELA chi-squared, thumbnail mismatch, PRNU.
          Does NOT give any verdict. Just returns raw findings.

INPUT   : preprocessing_json_path (str) — JSON from preprocessing_agent
OUTPUT  : saves  outputs/metadata/<stem>.json
          returns that JSON file path (str)

JSON output:
{
    "image_path"            : "/abs/path/to/image.jpg",
    "face_bbox"             : [x1, y1, x2, y2],
    "face_bboxes"           : [[x1,y1,x2,y2], ...],
    "exif_camera_present"   : false,
    "exif_camera_make"      : null,
    "exif_camera_model"     : null,
    "exif_datetime_original": null,
    "exif_gps_present"      : false,
    "software_tag"          : "Adobe Photoshop 24.0",
    "software_flagged"      : true,
    "ela_chi2"              : 847.3,
    "ela_map_path"          : "outputs/metadata/photo1_ela_map.png",
    "thumbnail_mismatch"    : true,
    "prnu_score"            : 0.013,    <- raw value, not a bool
    "prnu_absent"           : true,
    "exif_raw"              : { ... },
    "errors"                : []
}

Dependencies:
    pip install Pillow piexif numpy langchain langchain-core pydantic
"""

from __future__ import annotations

import io
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Type

import numpy as np
from PIL import Image, ImageChops, ImageFilter
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

try:
    import piexif
    _HAS_PIEXIF = True
except ImportError:
    _HAS_PIEXIF = False
    warnings.warn("piexif not installed — EXIF analysis limited.")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("MetadataAgent")

OUTPUT_DIR          = Path("outputs/metadata")
ELA_QUALITY         = 95    # high quality → only edited regions show difference
THUMB_MSE_THRESHOLD = 800.0 # pixel MSE above this = thumbnail mismatch
PRNU_VAR_THRESHOLD  = 0.05  # normalised variance below this = PRNU absent

# AI / editing software keywords (case-insensitive)
SUSPICIOUS_SW = [
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "dall-e", "dall·e", "stable diffusion", "midjourney", "firefly",
    "adobe", "canva", "fotor", "pixlr", "luminar",
    "gan", "diffusion", "generative", "ai image",
]


# ═════════════════════════════════════════════════════════════════════════════
#  1. EXIF PARSING
# ═════════════════════════════════════════════════════════════════════════════

def _parse_exif(path: str) -> dict:
    """
    Extract structured EXIF data.

    Returns dict with:
        camera_make, camera_model, software, datetime_original,
        gps_present, raw (all tags as flat dict)
    """
    base = {
        "camera_make": None, "camera_model": None,
        "software": None, "datetime_original": None,
        "gps_present": False, "raw": {},
    }

    if not _HAS_PIEXIF:
        # fallback to Pillow
        try:
            img = Image.open(path)
            raw = getattr(img, "_getexif", lambda: {})() or {}
            from PIL.ExifTags import TAGS
            named = {TAGS.get(k, k): v for k, v in raw.items()}
            base.update({
                "camera_make":       str(named.get("Make",  "")).strip() or None,
                "camera_model":      str(named.get("Model", "")).strip() or None,
                "software":          str(named.get("Software", "")).strip() or None,
                "datetime_original": str(named.get("DateTimeOriginal", "")).strip() or None,
                "gps_present":       "GPSInfo" in named,
            })
        except Exception:
            pass
        return base

    try:
        exif_dict = piexif.load(path)
    except Exception as e:
        log.warning("piexif load failed: %s", e)
        return base

    def _d(v):
        """Decode bytes to readable string."""
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8", errors="replace").strip("\x00 ") or None
            except Exception:
                return v.hex()
        return v if v != "" else None

    ifd0 = exif_dict.get("0th",  {})
    exif = exif_dict.get("Exif", {})
    gps  = exif_dict.get("GPS",  {})

    base["camera_make"]       = _d(ifd0.get(piexif.ImageIFD.Make))
    base["camera_model"]      = _d(ifd0.get(piexif.ImageIFD.Model))
    base["software"]          = _d(ifd0.get(piexif.ImageIFD.Software))
    base["datetime_original"] = _d(exif.get(piexif.ExifIFD.DateTimeOriginal))
    base["gps_present"]       = bool(gps)

    # flat raw dict for report
    raw = {}
    for ifd_name, data in [("0th", ifd0), ("Exif", exif)]:
        for tag_id, val in data.items():
            name      = piexif.TAGS[ifd_name].get(tag_id, {}).get("name", str(tag_id))
            raw[name] = _d(val)
    base["raw"] = raw
    return base


# ═════════════════════════════════════════════════════════════════════════════
#  2. SOFTWARE TAG CHECK
# ═════════════════════════════════════════════════════════════════════════════

def _check_software(sw: Optional[str]) -> tuple[Optional[str], bool]:
    """
    Check if software tag indicates editing or AI generation.

    Returns (software_tag, is_flagged).
    Flagged = any suspicious keyword found (case-insensitive).
    """
    if not sw:
        return None, False
    flagged = any(k in sw.lower() for k in SUSPICIOUS_SW)
    return sw.strip(), flagged


# ═════════════════════════════════════════════════════════════════════════════
#  3. ELA CHI-SQUARED
# ═════════════════════════════════════════════════════════════════════════════

def _compute_ela_chi2(img: Image.Image, stem: str) -> tuple[float, str]:
    """
    Error Level Analysis at high quality (95%) — detects editing artifacts.

    Formula:
        1. Re-save image as JPEG at quality=95
        2. diff = |original - resaved|  per pixel
        3. Compute 64-bin histogram of luminance diff
        4. Chi-squared vs uniform distribution

    Interpretation:
        Uniform/natural image → flat diff histogram → low chi2
        Edited regions → non-uniform diff → high chi2
        chi2 > 500 is suspicious

    Also saves amplified ELA map as PNG for visual inspection in report.
    """
    orig = img.convert("RGB")
    buf  = io.BytesIO()
    orig.save(buf, "JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    diff     = ImageChops.difference(orig, Image.open(buf).convert("RGB"))
    diff_arr = np.array(diff, dtype=np.float32)

    # save amplified ELA map
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ela_map_path = str(OUTPUT_DIR / f"{stem}_ela_map.png")
    Image.fromarray(
        np.clip(diff_arr * 20, 0, 255).astype(np.uint8)
    ).save(ela_map_path)

    # chi-squared over 64-bin luminance histogram
    luma        = diff_arr.mean(axis=2).flatten()
    observed, _ = np.histogram(luma, bins=64, range=(0, 255))
    observed    = observed.astype(np.float64) + 1e-9   # avoid div by zero
    expected    = observed.sum() / len(observed)        # uniform expectation
    chi2        = float(np.sum((observed - expected) ** 2 / expected))

    log.info("  ela_chi2=%.1f", chi2)
    return round(chi2, 3), ela_map_path


# ═════════════════════════════════════════════════════════════════════════════
#  4. THUMBNAIL MISMATCH
# ═════════════════════════════════════════════════════════════════════════════

def _check_thumbnail_mismatch(path: str, img: Image.Image) -> bool:
    """
    Cameras embed a thumbnail at capture time.
    If image was edited later, main image changes but thumbnail stays old.
    High MSE between thumbnail and main image = post-capture editing.

    Formula:
        MSE = mean((thumb_pixels - main_resized_pixels)^2)
        MSE > 800 → mismatch detected
    """
    if not _HAS_PIEXIF:
        return False
    try:
        thumb_bytes = piexif.load(path).get("thumbnail")
        if not thumb_bytes:
            log.info("  no embedded thumbnail")
            return False
        thumb  = Image.open(io.BytesIO(thumb_bytes)).convert("RGB")
        tw, th = thumb.size
        if tw < 10 or th < 10:
            return False
        main = img.convert("RGB").resize((tw, th), Image.LANCZOS)
        mse  = float(np.mean(
            (np.array(thumb, np.float32) - np.array(main, np.float32)) ** 2))
        log.info("  thumbnail MSE=%.1f  (threshold=%.1f)",
                 mse, THUMB_MSE_THRESHOLD)
        return mse > THUMB_MSE_THRESHOLD
    except Exception as e:
        log.warning("  thumbnail check error: %s", e)
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  5. PRNU  — Photo Response Non-Uniformity
# ═════════════════════════════════════════════════════════════════════════════

def _compute_prnu(img: Image.Image) -> tuple[float, bool]:
    """
    Real camera sensors have a fixed-pattern noise signature (PRNU).
    Synthetic/GAN images don't have this noise pattern.

    Formula:
        1. Convert to grayscale float [0,1]
        2. Smooth with Gaussian blur to estimate 'clean' signal
        3. Residual = original - smoothed  (this is the noise)
        4. prnu_score = var(residual) / var(original)
           normalised variance of the noise relative to image variance

    Interpretation:
        Real camera photo  → structured sensor noise → prnu_score > 0.05
        Synthetic image    → no sensor noise         → prnu_score < 0.05

    NOTE: prnu_score is returned as a raw float alongside prnu_absent bool.
          The pipeline uses the raw score for more nuanced decision-making.

    Returns (prnu_score, prnu_absent)
    """
    gray = np.array(img.convert("L"), np.float32) / 255.0
    h, w = gray.shape
    if h < 64 or w < 64:
        log.info("  image too small for PRNU — skipping")
        return 0.0, False

    pil_g   = Image.fromarray((gray * 255).astype(np.uint8))
    smooth  = np.array(pil_g.filter(ImageFilter.GaussianBlur(radius=2)),
                       np.float32) / 255.0
    residual = gray - smooth

    img_var  = float(np.var(gray))
    res_var  = float(np.var(residual))
    prnu_score = res_var / (img_var + 1e-8)

    prnu_absent = prnu_score < PRNU_VAR_THRESHOLD
    log.info("  prnu_score=%.5f  absent=%s  (threshold=%.3f)",
             prnu_score, prnu_absent, PRNU_VAR_THRESHOLD)
    return round(prnu_score, 6), prnu_absent


# ═════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def run_metadata(preprocessing_json_path: str) -> str:
    """
    Run metadata forensics on an image.
    Reads image_path and face_bbox from preprocessing JSON.
    Saves its own JSON to outputs/metadata/<stem>.json.

    This agent only MEASURES — it does not score or give a verdict.
    """
    errors = []

    if not os.path.isfile(preprocessing_json_path):
        raise FileNotFoundError(
            f"Preprocessing JSON not found: {preprocessing_json_path}")

    with open(preprocessing_json_path) as f:
        pre = json.load(f)

    image_path  = pre["image_path"]
    face_bbox   = pre["face_bbox"]
    face_bboxes = pre["face_bboxes"]

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Source image not found: {image_path}")

    log.info("▶ metadata  |  %s", Path(image_path).name)
    img  = Image.open(image_path)
    img.load()
    stem = Path(image_path).stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. EXIF
    exif           = _parse_exif(image_path)
    camera_present = bool(exif["camera_make"] or exif["camera_model"])
    sw_tag, sw_flag = _check_software(exif["software"])
    log.info("  camera=%s  software_tag=%s  flagged=%s",
             camera_present, sw_tag, sw_flag)

    # 2. ELA chi2
    try:
        ela_chi2, ela_map_path = _compute_ela_chi2(img, stem)
    except Exception as e:
        errors.append(f"ELA: {e}")
        ela_chi2, ela_map_path = 0.0, None
        log.error("  ELA failed: %s", e)

    # 3. Thumbnail mismatch
    try:
        thumb_mismatch = _check_thumbnail_mismatch(image_path, img)
    except Exception as e:
        errors.append(f"thumbnail: {e}")
        thumb_mismatch = False
        log.error("  thumbnail check failed: %s", e)

    # 4. PRNU
    try:
        prnu_score, prnu_absent = _compute_prnu(img)
    except Exception as e:
        errors.append(f"PRNU: {e}")
        prnu_score, prnu_absent = 0.0, False
        log.error("  PRNU failed: %s", e)

    # ── output dict — raw measurements only, no scoring ───────────────────────
    # anomaly score for metadata agent:
    # weighted combination of reliable signals only
    # software_flagged=0.40, thumbnail_mismatch=0.30, prnu_absent=0.20, no_provenance=0.10
    no_provenance = (not camera_present) and (not exif["datetime_original"])
    anomaly_score = round(min(1.0,
        0.40 * (1.0 if sw_flag        else 0.0) +
        0.30 * (1.0 if thumb_mismatch else 0.0) +
        0.20 * (1.0 if prnu_absent    else 0.0) +
        0.10 * (1.0 if no_provenance  else 0.0)
    ), 6)

    result = {
        # passthrough from preprocessing (so downstream has everything)
        "image_path":             image_path,
        "face_bbox":              face_bbox,
        "face_bboxes":            face_bboxes,
        # EXIF findings
        "exif_camera_present":    camera_present,
        "exif_camera_make":       exif["camera_make"],
        "exif_camera_model":      exif["camera_model"],
        "exif_datetime_original": exif["datetime_original"],
        "exif_gps_present":       exif["gps_present"],
        # software tag
        "software_tag":           sw_tag,
        "software_flagged":       sw_flag,
        # ELA
        "ela_chi2":               ela_chi2,       # raw chi2 value
        "ela_map_path":           ela_map_path,   # path to ELA map PNG
        # thumbnail
        "thumbnail_mismatch":     thumb_mismatch,
        # PRNU
        "prnu_score":             prnu_score,     # raw float (lower = more suspicious)
        "prnu_absent":            prnu_absent,    # True = no sensor noise detected
        # anomaly score for this agent
        "anomaly_score":          anomaly_score,  # 0=clean, 1=suspicious
        # raw exif dump for report
        "exif_raw":               exif["raw"],
        "errors":                 errors,
    }

    json_path = str(OUTPUT_DIR / f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    log.info("✔ metadata done  |  json → %s", json_path)
    return json_path


# ═════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN TOOL
# ═════════════════════════════════════════════════════════════════════════════

class MetadataInput(BaseModel):
    preprocessing_json_path: str = Field(
        description=(
            "Path to JSON file produced by preprocessing_agent. "
            "e.g. 'outputs/preprocessing/photo1.json'"
        )
    )


class MetadataAgent(BaseTool):
    """
    LangChain tool — run after preprocessing_agent.
    Input  : preprocessing_json_path
    Output : path to metadata JSON file
    """
    name: str = "metadata_agent"
    description: str = (
        "Run after preprocessing_agent. "
        "Input: preprocessing_json_path — JSON path from preprocessing_agent. "
        "Checks EXIF camera info, flags AI/editing software tags, "
        "computes ELA chi-squared, checks thumbnail mismatch, "
        "measures PRNU sensor noise. "
        "Output: path to JSON file with raw measurements — no verdict."
    )
    args_schema: Type[BaseModel] = MetadataInput

    def _run(self, preprocessing_json_path: str) -> str:
        return run_metadata(preprocessing_json_path)

    async def _arun(self, preprocessing_json_path: str) -> str:
        return self._run(preprocessing_json_path)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metadata_agent.py <preprocessing_json_path>")
        sys.exit(1)
    out = run_metadata(sys.argv[1])
    print(f"\nJSON → {out}\n")
    with open(out) as f:
        data = json.load(f)
    display = {k: v for k, v in data.items() if k != "exif_raw"}
    print(json.dumps(display, indent=2))
