import cv2
import dlib
import numpy as np
import os
import math


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

NORMS = {
    "inter_ocular_dist": (149.637, 14.964),
    "symmetry_index": (0.833, 0.123),
    "jaw_curvature_deg": (8.117, 0.812),
    "nasolabial_fold_depth": (11.915, 1.687),
    "eye_aspect_ratio_l": (0.326, 0.064),
    "eye_aspect_ratio_r": (0.328, 0.068),
    "lip_thickness_ratio": (0.167, 0.078),
    "philtrum_length_mm": (9.657, 2.697),
    "ear_alignment_px": (2.174, 1.980),
}

class GeometryAgent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.detector  = dlib.get_frontal_face_detector()
            cls._instance.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        return cls._instance

    def _get_landmarks(self, image_bgr: np.ndarray, face_bbox: list):
        """
        Compute 68 landmarks from a pre-loaded BGR numpy array.
        No file I/O — image is passed in directly by the caller (master_agent).
        """
        img = image_bgr.copy()
        h, w = img.shape[:2]
        scale = 1.0
        if min(h, w) < 300:
            scale = 300.0 / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_LINEAR)

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

        x1 = max(0, int(face_bbox[0] * scale))
        y1 = max(0, int(face_bbox[1] * scale))
        x2 = min(w, int(face_bbox[2] * scale))
        y2 = min(h, int(face_bbox[3] * scale))

        rect  = dlib.rectangle(x1, y1, x2, y2)
        shape = self.predictor(rgb, rect)
        pts   = np.array([[shape.part(i).x / scale, shape.part(i).y / scale]
                          for i in range(68)], dtype=float)
        return pts
    # ------------------------------------------------------------------ #
    #  FIX 1 — Yaw-corrected IOD and px/mm                                #
    # ------------------------------------------------------------------ #
    def _estimate_yaw_correction(self, pts):
        """
        Estimate a yaw correction factor from landmark asymmetry.

        For a perfectly frontal face the left and right halves of the jaw
        are equal, giving ratio = 1.0.  As yaw increases one half shrinks;
        ratio approaches 0 at ~90°.  Clamped to [0.5, 1.0] so we never
        over-correct on noisy detections.

        Returns a scalar in (0.5, 1.0].
        """
        mid_x      = (pts[0, 0] + pts[16, 0]) / 2.0
        left_half  = mid_x - pts[0, 0]
        right_half = pts[16, 0] - mid_x
        if left_half < 1 or right_half < 1:
            return 1.0   # can't estimate — assume frontal
        ratio = min(left_half, right_half) / (max(left_half, right_half) + 1e-6)
        return float(np.clip(ratio, 0.5, 1.0))

    def _iod_raw(self, pts):
        """Raw pixel IOD — outer eye corners."""
        return float(np.linalg.norm(pts[45] - pts[36]))

    def _iod_normalised(self, pts, img_width):
        """
        Yaw-corrected IOD scaled to a 500px-wide reference image.
        Oblique faces are corrected before the width normalisation so
        philtrum_length_mm and nasolabial_fold_depth stay consistent.
        """
        raw_iod        = self._iod_raw(pts)
        yaw_correction = self._estimate_yaw_correction(pts)
        corrected_iod  = raw_iod / (yaw_correction + 1e-6)
        return float(corrected_iod * (500.0 / img_width))

    def _px_per_mm(self, pts, img_width):
        """
        Pixels per mm derived from yaw-corrected IOD / 65 mm
        (mean adult inter-ocular distance).
        """
        raw_iod        = self._iod_raw(pts)
        yaw_correction = self._estimate_yaw_correction(pts)
        corrected_iod  = raw_iod / (yaw_correction + 1e-6)
        return float(corrected_iod / 65.0)

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #
    def _symmetry_index(self, pts):
        LEFT  = [0,1,2,3,4,5,6,7,31,32,36,37,38,39,40,41,48,49,50,59,58,57]
        RIGHT = [16,15,14,13,12,11,10,9,35,34,45,44,43,42,47,46,54,53,52,55,56,51]
        mid_x = (pts[0, 0] + pts[16, 0]) / 2
        iod   = self._iod_raw(pts)
        diffs = [abs(abs(pts[l, 0] - mid_x) - abs(pts[r, 0] - mid_x))
                 for l, r in zip(LEFT, RIGHT)]
        return float(1.0 - np.mean(diffs) / iod)

    def _jaw_curvature(self, pts):
        """True angle (degrees) of chin deviation from the straight jaw line (pts 0→16)."""
        jaw      = pts[0:17]
        start    = jaw[0]
        end      = jaw[16]
        mid      = jaw[8]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return 0.0
        t         = np.dot(mid - start, line_vec) / (line_len ** 2)
        closest   = start + t * line_vec
        deviation = np.linalg.norm(mid - closest)
        # arctan gives the true subtended angle in degrees
        return float(math.degrees(math.atan2(deviation, line_len)))

    def _nasolabial_fold_depth(self, pts, img_width):
        """
        Approximate nasolabial fold depth using the perpendicular
        distance from nose-mouth midpoint to the cheek line, in mm.
        Now uses yaw-corrected px_per_mm so oblique faces are consistent.
        """
        px_mm = self._px_per_mm(pts, img_width)

        left_mid   = (pts[33] + pts[48]) / 2
        right_mid  = (pts[33] + pts[54]) / 2
        cheek_left  = pts[2]
        cheek_right = pts[14]
        cheek_width_px = np.linalg.norm(cheek_right - cheek_left)

        left_depth  = np.linalg.norm(left_mid  - cheek_left)  / (cheek_width_px + 1e-6)
        right_depth = np.linalg.norm(right_mid - cheek_right) / (cheek_width_px + 1e-6)

        avg_ratio = (left_depth + right_depth) / 2.0
        return float(avg_ratio * 65.0 / (px_mm + 1e-6) * 0.35)

    def _eye_aspect_ratio(self, pts, side="left"):
        p = pts[36:42] if side == "left" else pts[42:48]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return float((A + B) / (2.0 * C + 1e-6))

    # ------------------------------------------------------------------ #
    #  FIX 2 — lip metric returns None when mouth is closed               #
    # ------------------------------------------------------------------ #
    def _lip_thickness_ratio(self, pts):
        """
        Returns the lip-height-to-mouth-width ratio when the mouth is open.

        Returns None when the mouth appears closed (ratio < 0.05).
        Callers must treat None as missing data, not as a zero contribution.
        Previously this silently returned the norm mean (0.22), which masked
        the fact that the measurement was unavailable.
        """
        lip_height  = np.linalg.norm(pts[62] - pts[66])
        mouth_width = np.linalg.norm(pts[48] - pts[54])
        ratio = float(lip_height / (mouth_width + 1e-6))
        if ratio < 0.05:
            return None   # mouth closed — metric not reliable, skip entirely
        return ratio

    def _philtrum_length_mm(self, pts, img_width):
        px_mm = self._px_per_mm(pts, img_width)
        px    = np.linalg.norm(pts[33] - pts[51])
        return float(px / (px_mm + 1e-6))

    def _ear_alignment(self, pts):
        """Vertical pixel difference between jaw endpoints (proxy for ear alignment)."""
        return float(abs(pts[0, 1] - pts[16, 1]))

    def _neck_face_boundary(self, pts, img):
        chin_x = int(pts[8, 0])
        chin_y = int(pts[8, 1])
        h      = img.shape[0]
        y1 = chin_y
        y2 = min(h, chin_y + 60)
        x1 = max(0, chin_x - 20)
        x2 = min(img.shape[1], chin_x + 20)
        if y2 <= y1:
            return "smooth"
        gray_strip = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        variance   = float(cv2.Laplacian(gray_strip, cv2.CV_64F).var())
        return "smooth" if variance < 500 else "sharp_edge"

    # ------------------------------------------------------------------ #
    #  FIX 3 — soft anomaly score, ramps from z > 1σ (not hard z > 2σ)   #
    # ------------------------------------------------------------------ #
    def _anomaly_score(self, metrics):
        """
        Soft scoring: any deviation beyond 1σ starts contributing.

        Old behaviour: hard gate at z > 2 meant a face consistently 1.8σ
        off on every metric scored exactly 0 — discarding real signal.

        New behaviour: smooth ramp from z = 1.0 onward.
            z = 1.0  → contribution ≈ 0.00  (normal range, silent)
            z = 1.5  → contribution ≈ 0.15  (mild suspicion)
            z = 2.0  → contribution ≈ 0.26  (moderate anomaly)
            z = 3.0  → contribution ≈ 0.41  (strong anomaly)
            z = 5.0  → contribution ≈ 0.57  (extreme)

        None values (e.g. closed-mouth lip metric) are skipped cleanly —
        they are never imputed, so missing data does not silently bias scores.

        Suggested detection threshold after this change: 0.20–0.25
        (was 0.30 with the old hard-gate formula).
        """
        contributions = []
        for key in NORMS:
            val = metrics.get(key)
            if val is None:          # missing data — skip, do not impute
                continue
            mu, sigma = NORMS[key]
            z = abs(val - mu) / (sigma + 1e-6)
            if z > 1.0:
                contribution = 1.0 - 1.0 / (1.0 + (z - 1.0) * 0.35)
                contributions.append(contribution)

        if not contributions:
            return 0.0

        return float(min(1.0, 0.6 * max(contributions) + 0.4 * np.mean(contributions)))

    # ------------------------------------------------------------------ #
    #  Overlay image — Figure 1 in report                                 #
    # ------------------------------------------------------------------ #
    def _save_overlay(self, img, pts, out_dir: str = "."):
        """
        Draw landmark overlay and save to out_dir.
        No longer requires image_path — takes an explicit output directory.
        """
        overlay = img.copy()
        for i, (x, y) in enumerate(pts.astype(int)):
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(overlay, str(i), (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 255), 1)
        for i in range(0, 16):
            cv2.line(overlay, tuple(pts[i].astype(int)),
                     tuple(pts[i + 1].astype(int)), (255, 100, 0), 1)
        mid_x = int((pts[0, 0] + pts[16, 0]) / 2)
        cv2.line(overlay, (mid_x, int(pts[27, 1])),
                 (mid_x, int(pts[8, 1])), (0, 100, 255), 1)
        out_path = os.path.join(out_dir, "figure1_landmark_overlay.png")
        cv2.imwrite(out_path, overlay)
        return out_path

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def run(self, image_bgr: np.ndarray, face_present: bool) -> dict:
        """
        Entry point called by geometry_tool in master_agent.

        Parameters
        ----------
        image_bgr   : np.ndarray  — full image loaded by cv2.imread in master_agent
        face_present: bool        — always True (gate handled by preprocess_node)

        Returns
        -------
        dict matching the contracts.py output schema for §5.1.
        """
        if not face_present:
            return {"agent_applicable": False}

        # ── Run dlib detector internally (not passed in from outside) ──────
        img = image_bgr.copy()
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        dets = self.detector(rgb, 1)
        if len(dets) == 0:
            return {"agent_applicable": False, "error": "dlib: no face detected"}

        det = dets[0]  # use highest-confidence detection
        face_bbox = [det.left(), det.top(), det.right(), det.bottom()]

        # ── Landmarks from numpy array (no file I/O) ───────────────────────
        pts      = self._get_landmarks(img, face_bbox)
        iod_norm = self._iod_normalised(pts, w)

        # Clip bbox for output
        x1 = max(0, face_bbox[0])
        y1 = max(0, face_bbox[1])
        x2 = min(w, face_bbox[2])
        y2 = min(h, face_bbox[3])

        # lip_thickness_ratio may be None (mouth closed) — stored as-is
        lip_ratio = self._lip_thickness_ratio(pts)

        metrics = {
            "inter_ocular_dist":     iod_norm,
            "symmetry_index":        self._symmetry_index(pts),
            "jaw_curvature_deg":     self._jaw_curvature(pts),
            "nasolabial_fold_depth": self._nasolabial_fold_depth(pts, w),
            "eye_aspect_ratio_l":    self._eye_aspect_ratio(pts, "left"),
            "eye_aspect_ratio_r":    self._eye_aspect_ratio(pts, "right"),
            "lip_thickness_ratio":   lip_ratio,   # None = mouth closed
            "philtrum_length_mm":    self._philtrum_length_mm(pts, w),
            "ear_alignment_px":      self._ear_alignment(pts),
        }

        anomaly = self._anomaly_score(metrics)

        return {
            # ── required by contracts.py (§5.1) ───────────────────────────
            "face_bbox":            [x1, y1, x2, y2],
            "symmetry_index":       metrics["symmetry_index"],
            "jaw_curvature_deg":    metrics["jaw_curvature_deg"],
            "ear_alignment_px":     metrics["ear_alignment_px"],
            "philtrum_length_mm":   metrics["philtrum_length_mm"],
            "interocular_dist_px":  metrics["inter_ocular_dist"],   # spec key
            "eye_aspect_ratio_l":   metrics["eye_aspect_ratio_l"],
            "eye_aspect_ratio_r":   metrics["eye_aspect_ratio_r"],
            "lip_thickness_ratio":  lip_ratio,                       # None when mouth closed
            "neck_face_boundary":   self._neck_face_boundary(pts, img),   # "smooth" | "sharp_edge"
            "landmark_confidence":  float(min(1.0, iod_norm / 80.0)),
            "anomaly_score":        anomaly,
            "agent_applicable":     anomaly > 0.0,
            # ── extra fields for report §5.1 ──────────────────────────────
            "nasolabial_fold_depth":  metrics["nasolabial_fold_depth"],
            "lip_metric_available":   lip_ratio is not None,
            "yaw_correction_factor":  self._estimate_yaw_correction(pts),
            "overlay_image_path":     self._save_overlay(img, pts),
        }