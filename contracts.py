"""
contracts.py — MFAD Multi-Factor AI Deepfake Detection System
=============================================================
Single source of truth for all agent output schemas.

Every key in every contract maps directly to a measurement described in the
reference forensic report DFA-2025-TC-00471 (Deep Sentinel Forensic AI Lab).
The section reference is noted in each comment so anyone can verify alignment.

This file defines three things:
  1. Output key contracts for every agent
  2. AUTHENTIC_BASELINES — population-level norms agents use to flag anomalies
  3. STUB_TEST_CASE_DFA_2025_TC_00471 — measurements from ONE specific test image,
     used ONLY for stub pipeline-flow testing. Not thresholds. Not expected values.
"""

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — PREPROCESSING
# Maps to: §1 Chain of Custody + §5.6 Provenance & Metadata Forensics
# ─────────────────────────────────────────────────────────────────────────────
PREPROCESSING_KEYS = [
    # §1 Chain of Custody
    "image_path",           # str   — absolute path to the analysed image
    "face_bbox",            # list  — [x1, y1, x2, y2] face bounding box in pixels
    "hash_sha256",          # str   — SHA-256 hex digest of raw input file (integrity check)
    "face_crop_path",       # str   — path to cropped face region saved to temp/
    "normalized_path",      # str   — path to face normalised to 224x224 for DL models
    "landmarks_path",       # str   — path to JSON file of 468 MediaPipe landmark coords
    # §5.6 Provenance & Metadata Forensics
    "exif_camera_present",  # bool  — False if camera make/model absent (absent in >97% GAN images)
    "ela_chi2",             # float — Error Level Analysis chi squared statistic (high = re-encoded)
    "thumbnail_mismatch",   # bool  — True if EXIF thumbnail != full-resolution image
    "prnu_absent",          # bool  — True if no camera sensor PRNU noise fingerprint found
    "software_tag",         # str   — EXIF Software field value, empty string if absent
    "icc_profile",          # str   — ICC colour profile name (inconsistency flags manipulation)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — GEOMETRY
# Maps to: §5.1 Facial Geometry & Landmark Analysis
# ─────────────────────────────────────────────────────────────────────────────
GEOMETRY_KEYS = [
    "symmetry_index",       # float — facial symmetry index           (authentic: 0.92-1.00)
    "jaw_curvature_deg",    # float — jaw-line curvature in degrees    (authentic: <5 deg)
    "ear_alignment_px",     # float — ear lobe vertical misalignment   (authentic: <3 px)
    "philtrum_length_mm",   # float — philtrum length in mm            (authentic: 11-18 mm)
    "interocular_dist_px",  # float — inter-ocular distance in pixels  (authentic: 58-72 px)
    "eye_aspect_ratio_l",   # float — left eye aspect ratio            (authentic: 0.28-0.34)
    "eye_aspect_ratio_r",   # float — right eye aspect ratio           (authentic: 0.28-0.34)
    "lip_thickness_ratio",  # float — lip thickness ratio              (authentic: 0.18-0.26)
    "neck_face_boundary",   # str   — "smooth" | "sharp_edge"  (sharp_edge = manipulation indicator)
    "anomaly_score",        # float — [0,1] geometry deepfake probability  (report §6 weight: 0.15)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — FREQUENCY + GAN
# Maps to: §5.2 GAN Artefact & Frequency-Domain Analysis
# ─────────────────────────────────────────────────────────────────────────────
FREQUENCY_KEYS = [
    "fft_mid_anomaly_db",       # float — PSD excess in 16-50 cycles/px band (dB)  (p<0.001 flag)
    "fft_high_anomaly_db",      # float — PSD excess in 51-100 cycles/px band (dB) (p<0.001 flag)
    "fft_ultrahigh_anomaly_db", # float — PSD excess in >100 cycles/px band (dB)   (StyleGAN2 sig.)
    "gan_probability",          # float — EfficientNet-B4 (FF++ v3) GAN classifier [0,1]
    "upsampling_grid_detected", # bool  — True if 4x4 px DCGAN grid artifact found in FFT
    "anomaly_score",            # float — [0,1] combined frequency+GAN probability  (§6 weight: 0.25)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — TEXTURE + SKIN CONSISTENCY
# Maps to: §5.3 Texture Consistency & Skin-Tone Mapping
# The report measures per-zone EMD for 5 distinct zone pairs — not a single score.
# ─────────────────────────────────────────────────────────────────────────────
TEXTURE_KEYS = [
    "forehead_cheek_emd",    # float — EMD forehead <-> cheek (L)        (authentic: <0.08)
    "cheek_jaw_emd_l",       # float — EMD cheek <-> jaw (L)             (authentic: <0.08)
    "cheek_jaw_emd_r",       # float — EMD cheek <-> jaw (R)             (authentic: <0.08)
    "periorbital_nasal_emd", # float — EMD periorbital <-> nasal bridge  (authentic: <0.08)
    "lip_chin_emd",          # float — EMD upper lip <-> chin            (authentic: <0.08)
    "neck_face_emd",         # float — EMD neck <-> face boundary        (authentic: <0.08; >0.15 = seam)
    "lbp_uniformity",        # float — LBP uniformity ratio              (authentic: >0.85)
    "seam_detected",         # bool  — True if any boundary EMD > 0.15
    "anomaly_score",         # float — [0,1] texture deepfake probability (§6 weight: 0.20)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — BIOLOGICAL PLAUSIBILITY
# Maps to: §5.5 Biological Plausibility Assessment
# ─────────────────────────────────────────────────────────────────────────────
BIOLOGICAL_KEYS = [
    "rppg_snr",              # float — rPPG signal-to-noise ratio        (authentic: >0.45)
    "corneal_deviation_deg", # float — corneal highlight angular deviation L vs R eye (authentic: <5 deg)
    "micro_texture_var",     # float — perioral micro-texture variance    (authentic mean: ~0.031)
    "vascular_pearson_r",    # float — subcutaneous vascular Pearson r   (authentic: >0.88)
    "anomaly_score",         # float — [0,1] biological plausibility probability (§6 weight: 0.15)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 6 — VLM EXPLAINABILITY
# Maps to: §5.4 Explainability Heat-Map Analysis (VLM Attention)
# The report defines 3 colour-coded activation zones: RED (>0.8), BLUE (0.5-0.8), VIOLET (<0.5)
# ─────────────────────────────────────────────────────────────────────────────
VLM_KEYS = [
    "heatmap_path",               # str   — path to Grad-CAM heatmap overlay PNG
    "vlm_caption",                # str   — BLIP-2 structured forensic natural language finding
    "vlm_verdict",                # str   — "REAL" | "FAKE" | "UNCERTAIN" parsed from caption
    "vlm_confidence",             # float — confidence of vlm_verdict [0,1]
    "saliency_score",             # float — mean Grad-CAM activation over face bbox [0,1]
    "high_activation_regions",    # list  — regions salience > 0.8   (RED zone in report)
    "medium_activation_regions",  # list  — regions salience 0.5-0.8 (BLUE zone)
    "low_activation_regions",     # list  — regions salience < 0.5   (VIOLET zone)
    "zone_gan_probability",       # float — GAN probability over central face zone (eyes/nose/mouth)
    "anomaly_score",              # float — [0,1] VLM-based probability (§6 weight: 0.25)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 7 — METADATA PROVENANCE + REFERENCE COMPARISON
# Maps to: §5.6 Provenance & Metadata Forensics + §7 Comparative Reference Analysis
# ─────────────────────────────────────────────────────────────────────────────
METADATA_KEYS = [
    # §5.6 Provenance
    "exif_camera_present",       # bool  — camera EXIF absent = strong GAN indicator
    "ela_chi2",                  # float — ELA chi squared statistic
    "thumbnail_mismatch",        # bool  — EXIF thumbnail vs full-res mismatch
    "prnu_absent",               # bool  — no camera PRNU fingerprint
    "software_tag",              # str   — EXIF Software field
    "jpeg_quantisation_anomaly", # bool  — non-standard JPEG quantisation table detected
    # §7 Comparative Reference Analysis
    "cosine_dist_authentic",     # float — FaceNet cosine dist to authentic reference (same person: <0.40)
    "cosine_dist_fake",          # float — FaceNet cosine dist to known deepfake cluster
    "facenet_dist",              # float — FaceNet-512 embedding distance to authentic
    "arcface_dist",              # float — ArcFace identity vector distance to authentic
    "shape_3dmm_dist",           # float — 3DMM shape coefficient cosine distance
    "reference_verdict",         # str   — "HIGH_DISSIMILARITY_TO_AUTHENTIC" | "LOW_DISSIMILARITY"
    "anomaly_score",             # float — [0,1] metadata provenance probability (§6)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 8 — BAYESIAN FUSION
# Maps to: §6 Confidence Scoring & Statistical Analysis
# ─────────────────────────────────────────────────────────────────────────────
FUSION_KEYS = [
    "final_score",          # float — Bayesian ensemble DeepFake Prediction Score [0,1]
    "confidence_interval",  # list  — [lower, upper] 95% CI via bootstrap resampling
    "per_module_scores",    # dict  — anomaly_score per module (see MODULE_SCORE_KEYS)
    "decision",             # str   — "DEEPFAKE" | "UNCERTAIN" | "AUTHENTIC"
    "interpretation",       # str   — "Very High Confidence" | "High" | "Moderate" | "Low"
    "model_auc_roc",        # float — model AUC-ROC on validation set  (reference model: 0.983)
    "false_positive_rate",  # float — FPR at 90% recall operating point (reference model: <0.021)
    "calibration_ece",      # float — Expected Calibration Error        (reference model: 0.014)
    "decision_threshold",   # float — threshold used for this run (default: 0.70)
]

# ─────────────────────────────────────────────────────────────────────────────
# AGENT 9 — REPORT GENERATOR
# Maps to: §8 Legal Certification & Admissibility + §9-§10 Analyst Declaration
# ─────────────────────────────────────────────────────────────────────────────
REPORT_KEYS = [
    "report_path",          # str  — absolute path to generated PDF
    "report_id",            # str  — format: DFA-{YYYY}-TC-{6_char_hex}
    "generated_at",         # str  — ISO 8601 timestamp
    "compliance_standards", # list — ["ISO/IEC 27037:2012", "SWGDE 2023",
                            #         "NIST SP 800-101r1", "ACPO v5", "FRE 702"]
    "analyst_name",         # str  — lead forensic analyst name
    "lab_accreditation",    # str  — e.g. "ASCLD/LAB International, ISO 17025:2017"
    "hash_sha256_verified", # bool — confirms input file hash matches at report generation
]

# ─────────────────────────────────────────────────────────────────────────────
# Module score keys — keys inside per_module_scores dict
# Names match the §6 table in the reference report exactly
# ─────────────────────────────────────────────────────────────────────────────
MODULE_SCORE_KEYS = [
    "geometry",     # §5.1 Facial Geometry & Landmark Deviation
    "gan_artefact", # §5.2 GAN Artefact Detection (EfficientNet-B4)
    "frequency",    # §5.2 Frequency-Domain Spectral Anomaly
    "texture",      # §5.3 Texture / Skin-Tone Consistency
    "vlm",          # §5.4 VLM Explainability Attention Score
    "biological",   # §5.5 Biological Plausibility Failure
    "metadata",     # §5.6 Metadata & Provenance Anomalies
]

# ─────────────────────────────────────────────────────────────────────────────
# Fusion weights — from §6 of reference report
# ─────────────────────────────────────────────────────────────────────────────
FUSION_WEIGHTS = {
    "geometry":     0.15,
    "gan_artefact": 0.25,
    "frequency":    0.25,
    "texture":      0.20,
    "vlm":          0.25,
    "biological":   0.15,
    "metadata":     0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# Decision thresholds — §6 of reference report
# ─────────────────────────────────────────────────────────────────────────────
DEEPFAKE_THRESHOLD  = 0.70   # final_score >= this → DEEPFAKE
AUTHENTIC_THRESHOLD = 0.35   # final_score <= this → AUTHENTIC; between → UNCERTAIN


# ─────────────────────────────────────────────────────────────────────────────
# Validation utility
# ─────────────────────────────────────────────────────────────────────────────
def validate(output: dict, keys: list, agent_name: str) -> bool:
    """
    Validate that an agent output dict contains all required keys.
    Raises ValueError listing any missing keys.
    Returns True on success.
    """
    missing = [k for k in keys if k not in output]
    if missing:
        raise ValueError(
            f"[contracts] {agent_name} output is missing required keys: {missing}\n"
            f"Got keys: {list(output.keys())}"
        )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTIC BASELINES
# Population-level norms from §5.1-§5.5 of reference report.
# Agents compare their measurements from the actual input image against these.
# These never change — they are research-derived baselines, not case measurements.
# ─────────────────────────────────────────────────────────────────────────────
AUTHENTIC_BASELINES = {
    # §5.1 Geometry
    "symmetry_index_min":            0.92,
    "symmetry_index_max":            1.00,
    "jaw_curvature_deg_max":         5.0,
    "ear_alignment_px_max":          3.0,
    "philtrum_length_mm_min":        11.0,
    "philtrum_length_mm_max":        18.0,
    "interocular_dist_px_min":       58.0,
    "interocular_dist_px_max":       72.0,
    "eye_aspect_ratio_min":          0.28,
    "eye_aspect_ratio_max":          0.34,
    "lip_thickness_ratio_min":       0.18,
    "lip_thickness_ratio_max":       0.26,
    # §5.2 Frequency
    "fft_mid_anomaly_db_threshold":  3.0,
    "fft_high_anomaly_db_threshold": 5.0,
    # §5.3 Texture
    "emd_max":                       0.08,
    "seam_emd_threshold":            0.15,
    "lbp_uniformity_min":            0.85,
    # §5.5 Biological
    "rppg_snr_min":                  0.45,
    "corneal_deviation_deg_max":     5.0,
    "micro_texture_var_mean":        0.031,
    "vascular_pearson_r_min":        0.88,
    # §7 Reference comparison
    "cosine_dist_same_person_max":   0.40,
}


# ─────────────────────────────────────────────────────────────────────────────
# STUB TEST CASE — DFA-2025-TC-00471
# Measurements from ONE specific deepfake image (Tom Cruise, March 2026).
# Use ONLY to test stub pipeline flow end-to-end.
# Real images produce completely different values. Do NOT treat as targets.
# ─────────────────────────────────────────────────────────────────────────────
STUB_TEST_CASE_DFA_2025_TC_00471 = {
    # §6 final results
    "final_score":              0.950,
    "confidence_interval":      [0.931, 0.966],
    "model_auc_roc":            0.983,
    "false_positive_rate":      0.021,
    "calibration_ece":          0.014,
    "decision_threshold":       0.70,
    "per_module_scores": {
        "geometry":     0.884,
        "gan_artefact": 0.967,
        "frequency":    0.912,
        "texture":      0.895,
        "vlm":          0.931,
        "biological":   0.826,
        "metadata":     0.973,
    },
    # §5.1 Geometry
    "symmetry_index":           0.74,
    "jaw_curvature_deg":        11.2,
    "ear_alignment_px":         8.7,
    "philtrum_length_mm":       20.4,
    "interocular_dist_px":      64.3,
    "eye_aspect_ratio_l":       0.31,
    "eye_aspect_ratio_r":       0.29,
    "lip_thickness_ratio":      0.22,
    "neck_face_boundary":       "sharp_edge",
    # §5.2 Frequency
    "fft_mid_anomaly_db":       9.4,
    "fft_high_anomaly_db":      13.3,
    "fft_ultrahigh_anomaly_db": 15.6,
    "gan_probability":          0.967,
    # §5.3 Texture — per zone
    "forehead_cheek_emd":       0.061,
    "cheek_jaw_emd_l":          0.193,
    "cheek_jaw_emd_r":          0.211,
    "periorbital_nasal_emd":    0.072,
    "lip_chin_emd":             0.148,
    "neck_face_emd":            0.274,
    "lbp_uniformity":           0.51,
    # §5.5 Biological
    "rppg_snr":                 0.09,
    "corneal_deviation_deg":    14.3,
    "micro_texture_var":        0.012,
    "vascular_pearson_r":       0.41,
    # §7 Reference comparison
    "cosine_dist_authentic":    0.71,
    "cosine_dist_fake":         0.18,
    "facenet_dist":             0.71,
    "arcface_dist":             0.68,
    "shape_3dmm_dist":          0.58,
}
