"""
agents/face/geometry.py
Owner: Veedhee
Activated by: face_present=True from content_router
Face bbox lifecycle: geometry_agent runs RetinaFace → gets face_bbox →
pool_dispatch_node passes it to remaining face agents
"""

import numpy as np
import dlib
import cv2
from scipy.spatial import distance as dist


# ──────────────────────────────────────────────
# Authentic norms (mean ± 2 SD thresholds)
# ──────────────────────────────────────────────
NORMS = {
    "symmetry_index":     {"mean": 0.85, "sd": 0.07},
    "jaw_curvature_deg":  {"mean": 45.0,  "sd": 25.0},
    "ear_alignment_px":   {"mean": 5.0,  "sd": 4.0},
    "philtrum_length":    {"mean": 0.15, "sd": 0.05},
}

DLIB_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # ~100MB, not in git


# ──────────────────────────────────────────────
# RetinaFace loader (lazy import)
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Step 1 — RetinaFace detection on full image
# ──────────────────────────────────────────────
def detect_face_retinaface(image_bgr: np.ndarray) -> dict:
    """
    Uses dlib instead of RetinaFace for face detection.
    Returns same schema so the rest of the pipeline is unchanged.
    """
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dets, scores, _ = detector.run(gray, 0)

    if len(dets) == 0:
        return {"face_bbox": None, "landmark_confidence": 0.0, "found": False}

    # Pick highest scoring detection
    best_idx = int(np.argmax(scores))
    d = dets[best_idx]
    score = float(scores[best_idx])

    # Normalize score to 0-1 range (dlib scores typically 0-3)
    confidence = min(max(score / 1.0, 0.0), 1.0)


    h, w = image_bgr.shape[:2]
    face_bbox = [
        max(0, d.left()),
        max(0, d.top()),
        min(w, d.right()),
        min(h, d.bottom())
    ]
    return {
        "face_bbox": face_bbox,
        "landmark_confidence": round(confidence, 4),
        "found": True,
    }


# ──────────────────────────────────────────────
# Step 2 — Crop face region
# ──────────────────────────────────────────────
def crop_face(image_bgr: np.ndarray, face_bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = face_bbox
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image_bgr[y1:y2, x1:x2]


# ──────────────────────────────────────────────
# Step 3 — dlib 68-point landmarks on face crop
# ──────────────────────────────────────────────
def get_landmarks_dlib(face_crop_bgr: np.ndarray) -> np.ndarray | None:
    """
    Returns (68, 2) array of (x, y) landmark coordinates,
    or None if no face is detected in the crop.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 1)

    if len(dets) == 0:
        return None

    shape = predictor(gray, dets[0])
    landmarks = np.array([[shape.part(i).x, shape.part(i).y]
                           for i in range(68)], dtype=np.float64)
    return landmarks


# ──────────────────────────────────────────────
# Step 4 — Anthropometric ratio computations
# ──────────────────────────────────────────────

def compute_symmetry_index(lm: np.ndarray) -> float:
    """
    Compares left vs right mirrored landmark distances from the face midline.
    Uses paired landmarks: eyebrows (17-21 vs 22-26), eyes (36-41 vs 42-47),
    mouth corners (48, 54).
    Returns symmetry_index in [0, 1]; 1.0 = perfect symmetry.
    """
    # Midline x = midpoint of nasion (landmark 27) and chin (8)
    midline_x = (lm[27][0] + lm[8][0]) / 2.0

    # Paired landmark indices (left, right)
    pairs = [
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # eyebrows
        (36, 45), (39, 42),                                   # eye corners
        (48, 54), (49, 53), (50, 52),                         # mouth
        (0, 16), (1, 15), (2, 14), (3, 13),                   # jaw sides
    ]

    diffs = []
    for l_idx, r_idx in pairs:
        d_left  = abs(lm[l_idx][0] - midline_x)
        d_right = abs(lm[r_idx][0] - midline_x)
        max_d = max(d_left, d_right, 1e-6)
        diffs.append(abs(d_left - d_right) / max_d)

    asymmetry = float(np.mean(diffs))
    return round(1.0 - asymmetry, 4)


def compute_jaw_curvature_deg(lm: np.ndarray) -> float:
    """
    Fits a polynomial to jaw landmarks (0–16) and returns the
    mean angular deviation in degrees from a straight line.
    """
    jaw_pts = lm[0:17]  # 17 jaw landmarks
    x = jaw_pts[:, 0]
    y = jaw_pts[:, 1]

    coeffs = np.polyfit(x, y, 2)           # quadratic fit
    a = coeffs[0]
    # Curvature ≈ 2a for quadratic; convert to degrees via arctan
    curvature_rad = np.arctan(abs(2 * a * np.mean(x)))
    return round(float(np.degrees(curvature_rad)), 4)


def compute_ear_alignment_px(lm: np.ndarray) -> float:
    """
    Measures vertical misalignment between left ear (landmark 0)
    and right ear (landmark 16) in pixels.
    """
    left_ear_y  = lm[0][1]
    right_ear_y = lm[16][1]
    return round(float(abs(left_ear_y - right_ear_y)), 4)


def compute_philtrum_length(lm: np.ndarray) -> float:
    """
    Philtrum = distance from nose base (landmark 33) to upper lip (landmark 51),
    normalised by face height (chin 8 to nasion 27).
    """
    nose_base   = lm[33]
    upper_lip   = lm[51]
    chin        = lm[8]
    nasion      = lm[27]

    philtrum_px   = dist.euclidean(nose_base, upper_lip)
    face_height   = dist.euclidean(chin, nasion)

    if face_height < 1e-6:
        return 0.0
    return round(float(philtrum_px / face_height), 4)


# ──────────────────────────────────────────────
# Step 5 — Anomaly score vs authentic norms
# ──────────────────────────────────────────────
def compute_anomaly_score(metrics: dict) -> float:
    """
    Z-score each metric against norms; anomaly_score = mean of
    normalised deviations clipped to [0, 1].
    A score > 0.5 suggests deviation beyond ±2 SD from norms.
    """
    z_scores = []
    for key, norm in NORMS.items():
        value = metrics.get(key)
        if value is None:
            continue
        z = abs(value - norm["mean"]) / (2 * norm["sd"] + 1e-9)
        z_scores.append(min(z, 1.0))          # clip each z to [0, 1]

    return round(float(np.mean(z_scores)) if z_scores else 0.0, 4)


# ──────────────────────────────────────────────
# Main agent entry point
# ──────────────────────────────────────────────
def run(image_bgr: np.ndarray, face_present: bool = True) -> dict:
    """
    Geometry agent entry point.

    Args:
        image_bgr:    Full image as BGR numpy array.
        face_present: Gate flag from content_router. If False, skip processing.

    Returns:
        Agent output dict matching the defined schema.
    """
    # Gate check
    if not face_present:
        return {"agent_applicable": False, "reason": "face_present=False from content_router"}

    # ── Step 1: RetinaFace detection ──
    detection = detect_face_retinaface(image_bgr)

    face_bbox           = detection["face_bbox"]
    landmark_confidence = detection["landmark_confidence"]

    # ── Step 6: Early exit conditions ──
    if not detection["found"] or landmark_confidence < 0.1:
        return {
            "face_bbox":            face_bbox,
            "symmetry_index":       None,
            "jaw_curvature_deg":    None,
            "ear_alignment_px":     None,
            "philtrum_length":      None,
            "landmark_confidence":  landmark_confidence,
            "anomaly_score":        None,
            "agent_applicable":     False,
        }

    # ── Step 2: Crop face ──
    face_crop = crop_face(image_bgr, face_bbox)

    # ── Step 3: dlib 68-point landmarks ──
    landmarks = get_landmarks_dlib(face_crop)

    if landmarks is None:
        return {
            "face_bbox":            face_bbox,
            "symmetry_index":       None,
            "jaw_curvature_deg":    None,
            "ear_alignment_px":     None,
            "philtrum_length":      None,
            "landmark_confidence":  landmark_confidence,
            "anomaly_score":        None,
            "agent_applicable":     False,
        }

    # ── Step 4: Anthropometric ratios ──
    symmetry_index    = compute_symmetry_index(landmarks)
    jaw_curvature_deg = compute_jaw_curvature_deg(landmarks)
    ear_alignment_px  = compute_ear_alignment_px(landmarks)
    philtrum_length   = compute_philtrum_length(landmarks)

    metrics = {
        "symmetry_index":    symmetry_index,
        "jaw_curvature_deg": jaw_curvature_deg,
        "ear_alignment_px":  ear_alignment_px,
        "philtrum_length":   philtrum_length,
    }

    # ── Step 5: Anomaly score ──
    anomaly_score = compute_anomaly_score(metrics)

    return {
        "face_bbox":            face_bbox,
        "symmetry_index":       symmetry_index,
        "jaw_curvature_deg":    jaw_curvature_deg,
        "ear_alignment_px":     ear_alignment_px,
        "philtrum_length":      philtrum_length,
        "landmark_confidence":  landmark_confidence,
        "anomaly_score":        anomaly_score,
        "agent_applicable":     True,
    }