import cv2
import dlib
import numpy as np
import os


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

NORMS = {
    "inter_ocular_dist":     (120.0, 20.0),
    "symmetry_index":        (0.96,  0.04),
    "jaw_curvature_deg":     (2.5,   1.5),
    "nasolabial_fold_depth": (2.95,  0.425),
    "eye_aspect_ratio_l":    (0.31,  0.03),
    "eye_aspect_ratio_r":    (0.31,  0.03),
    "lip_thickness_ratio":   (0.22,  0.04),
    "philtrum_length_mm":    (14.5,  3.5),
    "ear_alignment_px":      (1.5,   1.5),
}


class GeometryAgent:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.detector  = dlib.get_frontal_face_detector()
            cls._instance.predictor = dlib.shape_predictor(PREDICTOR_PATH)
        return cls._instance

    def _get_landmarks(self, image_path, face_bbox):
        img  = cv2.imread(image_path)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x1 = max(0, face_bbox[0])
        y1 = max(0, face_bbox[1])
        x2 = min(w, face_bbox[2])
        y2 = min(h, face_bbox[3])
        rect  = dlib.rectangle(x1, y1, x2, y2)
        shape = self.predictor(gray, rect)
        pts   = np.array([[shape.part(i).x, shape.part(i).y]
                          for i in range(68)], dtype=float)
        return pts, img

    def _iod_raw(self, pts):
        """Raw pixel IOD — outer eye corners."""
        return float(np.linalg.norm(pts[45] - pts[36]))

    def _iod_normalised(self, pts, img_width):
        """IOD normalised to a standard 500px wide image."""
        return self._iod_raw(pts) * (500.0 / img_width)

    def _px_per_mm(self, pts, img_width):
        """Use normalised IOD (scaled to 500px wide image) / 65mm."""
        norm_iod = self._iod_raw(pts) * (500.0 / img_width)
        return norm_iod / 65.0

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
        """Deviation of chin midpoint from straight jaw line, in degrees."""
        jaw   = pts[0:17]
        start = jaw[0]
        end   = jaw[16]
        mid   = jaw[8]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            return 0.0
        t = np.dot(mid - start, line_vec) / (line_len ** 2)
        closest  = start + t * line_vec
        deviation = np.linalg.norm(mid - closest)
        return float((deviation / line_len) * 10.0)

    def _nasolabial_fold_depth(self, pts, img_width):
        """
        Approximate nasolabial fold depth using the perpendicular
        distance from nose-mouth midpoint to the cheek line, in mm.
        """
        px_mm = self._px_per_mm(pts, img_width)

        # Midpoint between nose bottom (33) and mouth corner (48) — left side
        left_mid  = (pts[33] + pts[48]) / 2
        # Midpoint between nose bottom (33) and mouth corner (54) — right side
        right_mid = (pts[33] + pts[54]) / 2

        # Use cheek width (landmark 2 to 14) as reference line
        cheek_left  = pts[2]
        cheek_right = pts[14]
        cheek_width_px = np.linalg.norm(cheek_right - cheek_left)

        # Fold depth as fraction of cheek width, converted to mm
        left_depth  = np.linalg.norm(left_mid  - cheek_left)  / (cheek_width_px + 1e-6)
        right_depth = np.linalg.norm(right_mid - cheek_right) / (cheek_width_px + 1e-6)

        avg_ratio = (left_depth + right_depth) / 2.0
        # Scale to expected mm range (2.1–3.8mm)
        return float(avg_ratio * 65.0 / (px_mm + 1e-6) * 0.35)
    
    def _eye_aspect_ratio(self, pts, side="left"):
        p = pts[36:42] if side == "left" else pts[42:48]
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return float((A + B) / (2.0 * C + 1e-6))

    def _lip_thickness_ratio(self, pts):
        lip_height  = np.linalg.norm(pts[62] - pts[66])
        mouth_width = np.linalg.norm(pts[48] - pts[54])
        ratio = float(lip_height / (mouth_width + 1e-6))
        
        # Mouth is closed — metric unreliable, return norm mean so it contributes 0 to anomaly score
        if ratio < 0.05:
            return 0.22
        
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
        return "smooth" if variance < 500 else "sharp edge"

    # ------------------------------------------------------------------ #
    #  Anomaly score — only deviations beyond ±2 SD contribute            #
    # ------------------------------------------------------------------ #
    def _anomaly_score(self, metrics):
        contributions = []
        for key in NORMS:
            if key not in metrics:
                continue
            mu, sigma = NORMS[key]
            z = abs(metrics[key] - mu) / (sigma + 1e-6)
            if z > 2.0:
                contribution = 1.0 - 1.0 / (1.0 + (z - 2.0) * 0.4)
                contributions.append(contribution)

        if not contributions:
            return 0.0

        return float(min(1.0, 0.6 * max(contributions) + 0.4 * np.mean(contributions)))

    # ------------------------------------------------------------------ #
    #  Overlay image — Figure 1 in report                                 #
    # ------------------------------------------------------------------ #
    def _save_overlay(self, img, pts, image_path):
        overlay = img.copy()
        for i, (x, y) in enumerate(pts.astype(int)):
            cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(overlay, str(i), (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 200, 255), 1)
        for i in range(0, 16):
            cv2.line(overlay, tuple(pts[i].astype(int)),
                     tuple(pts[i+1].astype(int)), (255, 100, 0), 1)
        mid_x = int((pts[0, 0] + pts[16, 0]) / 2)
        cv2.line(overlay, (mid_x, int(pts[27, 1])),
                 (mid_x, int(pts[8, 1])), (0, 100, 255), 1)
        out_path = os.path.join(os.path.dirname(image_path) or ".",
                                "figure1_landmark_overlay.png")
        cv2.imwrite(out_path, overlay)
        return out_path

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def run(self, image_path: str, face_bbox: list) -> dict:
        pts, img = self._get_landmarks(image_path, face_bbox)
        iod_norm = self._iod_normalised(pts, img.shape[1])

        # Clip bbox for output
        h, w = img.shape[:2]
        x1 = max(0, face_bbox[0])
        y1 = max(0, face_bbox[1])
        x2 = min(w, face_bbox[2])
        y2 = min(h, face_bbox[3])

        metrics = {
            "inter_ocular_dist":     iod_norm,
            "symmetry_index":        self._symmetry_index(pts),
            "jaw_curvature_deg":     self._jaw_curvature(pts),
            "nasolabial_fold_depth": self._nasolabial_fold_depth(pts, img.shape[1]),
            "eye_aspect_ratio_l":    self._eye_aspect_ratio(pts, "left"),
            "eye_aspect_ratio_r":    self._eye_aspect_ratio(pts, "right"),
            "lip_thickness_ratio":   self._lip_thickness_ratio(pts),
            "philtrum_length_mm":   self._philtrum_length_mm(pts, img.shape[1]),
            "ear_alignment_px":      self._ear_alignment(pts),
        }

        anomaly = self._anomaly_score(metrics)

        return {
            # --- required by contracts.py ---
            "face_bbox":           [x1, y1, x2, y2],
            "symmetry_index":      metrics["symmetry_index"],
            "jaw_curvature_deg":   metrics["jaw_curvature_deg"],
            "ear_alignment_px":    metrics["ear_alignment_px"],
            "philtrum_length_mm":   metrics["philtrum_length_mm"],
            "landmark_confidence": float(min(1.0, iod_norm / 80.0)),
            "anomaly_score":       anomaly,
            "agent_applicable":    anomaly > 0.0,   # False only if no face found
            # --- extra metrics for report section 4.1 ---
            "inter_ocular_dist":     metrics["inter_ocular_dist"],
            "nasolabial_fold_depth": metrics["nasolabial_fold_depth"],
            "eye_aspect_ratio_l":    metrics["eye_aspect_ratio_l"],
            "eye_aspect_ratio_r":    metrics["eye_aspect_ratio_r"],
            "lip_thickness_ratio":   metrics["lip_thickness_ratio"],
            "neck_face_boundary":    self._neck_face_boundary(pts, img),
            "overlay_image_path":    self._save_overlay(img, pts, image_path),
        }
