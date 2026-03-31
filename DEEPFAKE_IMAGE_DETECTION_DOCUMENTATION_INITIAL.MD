# MFAD — Technical Documentation v2.0
### Multimodal Forensic Agent for Deepfake Detection (Image-Only)

**Target output:** A forensic report matching the structure of `DeepFake_Forensic_Report_DFA2025TC00471.pdf`
**Stack:** LangGraph · LangChain · HuggingFace · PyTorch · OpenCV · ReportLab · Ollama (Mistral-7B)
**Hardware:** RTX A6000 × 2 (96 GB total VRAM) — LLaVA-1.5-7b loads on a single GPU
**Rule:** Every agent returns a Python `dict` with an `anomaly_score` key (float 0–1). No exceptions.

> **CORRECTION from v1.0:** Several descriptions in the original documentation were inaccurate relative to the actual code. This document reflects what the code actually does, not what was originally planned. Deviations from v1.0 are marked **[CHANGED]**.

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Data Flow — End to End](#data-flow)
3. [contracts.py](#contractspy)
4. [agents/preprocessing_agent.py](#agentspreprocessing_agentpy) — Aditi
5. [agents/geometry.py](#agentsgeometrypy) — Veedhee
6. [agents/frequency_agent.py](#agentsfrequency_agentpy) — Pushp
7. [agents/texture.py](#agentsTexturepy) — Manya
8. [agents/vlm.py](#agentsvlmpy) — Aryan
9. [agents/biological_plausibility_agent.py](#agentsbiological_plausibility_agentpy) — Diksha
10. [agents/metadata_agent.py](#agentsmetadata_agentpy) — Sania / Aditi
11. [fusion/bayesian.py](#fusionbayesianpy) — Dev
12. [master_agent.py](#master_agentpy) — Dev
13. [report_agent/template.py](#report_agenttemplatepy) — Saania '27
14. [report_agent/generate.py](#report_agentgeneratepy) — Saania '27
15. [config.json](#configjson)
16. [Report Reference — Page by Page](#report-reference)
17. [How to Run](#how-to-run)

---

## Repository Structure

```
mfad/
├── contracts.py                    # shared output schema — read this first
├── config.json                     # runtime config for all agents
├── master_agent.py                 # LangGraph StateGraph orchestrator
├── agents/
│   ├── preprocessing_agent.py      # Aditi — pipeline entry, saves JSON, everyone depends on this
│   ├── geometry.py                 # Veedhee — 68-pt dlib landmarks, anthropometric ratios
│   ├── frequency_agent.py          # Pushp — FFT azimuthal + Block DCT (config-driven)
│   ├── texture.py                  # Manya — LBP, Gabor, EMD skin zone analysis
│   ├── vlm.py                      # Aryan — LLaVA-1.5-7b caption + Grad-CAM heatmap (placeholder)
│   ├── biological_plausibility_agent.py  # Diksha — pupil BIoU, corneal IoU, micro-texture
│   └── metadata_agent.py           # Sania/Aditi — EXIF, ELA chi2, PRNU, thumbnail check
├── fusion/
│   └── bayesian.py                 # log-odds Bayesian fusion of all anomaly_scores
├── report_agent/
│   ├── __init__.py
│   ├── template.py                 # ReportLab page layout — 10 sections, full cover
│   ├── generate.py                 # Mistral-7B narrative + final PDF assembly
│   └── changelog.md                # Saania's change log — update after every edit
├── test_images/
│   ├── sample_real.jpg             # verified authentic face for testing
│   └── sample_fake.jpg             # known deepfake for testing
└── outputs/                        # generated heatmaps, ELA maps, PDFs, JSONs
    ├── preprocessing/              # <stem>.json + <stem>_normalised.jpg
    └── metadata/                   # <stem>.json + <stem>_ela_map.png
```

---

## Data Flow

```
INPUT IMAGE (jpg/png) — e.g. test_images/image1.jpg
        │
        ▼
preprocess_node (CLIP gate)
  → preprocessing_agent.run_preprocessing(image_path)
  → saves outputs/preprocessing/<stem>.json
  → returns dict with: image_path, face_bbox, face_bboxes, face_crop_path,
                        normalized_path, hash_sha256, ela_score, ...
        │
        │ (face_detected = True)
        ▼
parallel_analysis_node (asyncio.gather — all 6 run concurrently)
        │
        ├── geometry_tool          image_path + face_bbox → geometry.run(image_bgr)
        ├── frequency_tool         image_path + face_bbox → crops face → frequency_agent.run()
        ├── texture_tool           image_path + face_bbox → texture.run_texture_agent()
        ├── vlm_tool               image_path + face_bbox + face_crop_path → VLMAgent().run()
        ├── biological_tool        image_path + face_bbox → analyse_image()
        └── metadata_tool          preprocessing_json_path → run_metadata()
        │
        ▼
fusion_node
  → collects anomaly_score from every agent output
  → calls bayesian_fusion(per_module_scores)
  → returns final_score, confidence_interval, decision, per_module_scores
        │
        ▼
report_node
  → merges all agent outputs into flat ctx dict
  → calls Mistral-7B via Ollama for narrative text (falls back gracefully)
  → calls ReportGenerator().generate(ctx) → build_report() → PDF
        │
        ▼
FORENSIC PDF REPORT → outputs/DFA-{YEAR}-TC-{HEX}.pdf
```

**[CHANGED from v1.0]:** The orchestrator is **LangGraph StateGraph**, not `create_tool_calling_agent`.
Mistral-7B is only used for narrative text in report_node, not for agent orchestration.

---

## contracts.py

**Owner:** Everyone reads this. No one modifies output keys without team agreement.
**Purpose:** Single source of truth for every agent's output schema AND population-level baselines.

Key sections:
- `PREPROCESSING_KEYS`, `GEOMETRY_KEYS`, `FREQUENCY_KEYS`, `TEXTURE_KEYS`, `VLM_KEYS`, `BIOLOGICAL_KEYS`, `METADATA_KEYS`, `FUSION_KEYS`, `REPORT_KEYS`
- `MODULE_SCORE_KEYS` — keys inside `per_module_scores` dict
- `FUSION_WEIGHTS` — reliability multipliers per module (do NOT sum to 1.0 — this is intentional)
- `DEEPFAKE_THRESHOLD = 0.70`, `AUTHENTIC_THRESHOLD = 0.35`
- `AUTHENTIC_BASELINES` — population norms agents compare against
- `STUB_TEST_CASE_DFA_2025_TC_00471` — measurements from the reference test image (Tom Cruise deepfake). Use ONLY for stub pipeline testing. These are not thresholds.

**[CHANGED from v1.0]:** The `REFERENCE_OUTPUT` schema described in v1.0 does not exist.
Reference comparison keys are merged into `METADATA_KEYS`. The `validate()` utility
function raises `ValueError` with a list of missing keys if any are absent.

---

## agents/preprocessing_agent.py

**Owner:** Aditi
**Priority:** HIGHEST — all other agents are blocked until this completes
**Libraries:** `Pillow`, `piexif`, `retinaface` (preferred) → `mediapipe` → `cv2` (fallback chain)

### What it does
1. Computes SHA-256 + MD5 hashes of raw file bytes (chain of custody)
2. Detects all faces via RetinaFace → MediaPipe → OpenCV (in order of preference)
3. Selects largest face as primary bbox; returns all faces in `face_bboxes`
4. Crops and normalises primary face to 512×512 RGB using Lanczos with 15% padding
5. Computes ELA score (Error Level Analysis) on the ORIGINAL image before any resizing
6. Saves everything to `outputs/preprocessing/<stem>.json`
7. Returns the JSON file path (not a dict)

**Important:** master_agent's `preprocessing_tool` opens the JSON and inlines the dict,
then adds `"preprocessing_json_path"` to it. All downstream agents receive the full dict.

### Input
```python
image_path: str  # e.g. "test_images/sample.jpg"
```

### Output (saved to JSON, returned as dict by preprocessing_tool)
```python
{
    "image_path":          "/abs/path/to/image.jpg",
    "face_bbox":           [x1, y1, x2, y2],        # primary (largest) face
    "face_bboxes":         [[x1,y1,x2,y2], ...],    # all detected faces
    "face_count":          1,
    "face_detected":       True,
    "image_dims":          [1920, 1080],             # original [W, H]
    "hash_sha256":         "a3f9b2...",
    "hash_md5":            "c4e8a1...",
    "ela_score":           0.34,                     # 0=clean, 1=heavily altered
    "anomaly_score":       0.34,                     # = ela_score for this agent
    "normalised_img_path": "outputs/preprocessing/photo1_normalised.jpg",
    "normalized_path":     "outputs/preprocessing/photo1_normalised.jpg",  # alias
    "face_crop_path":      "outputs/preprocessing/photo1_normalised.jpg",  # for VLM
    "landmarks_path":      "",    # filled by geometry agent
    "icc_profile":         "",    # not yet extracted
    "exif_metadata":       { ... },
    "errors":              [],
    "preprocessing_json_path": "outputs/preprocessing/photo1.json"  # added by tool wrapper
}
```

### Report connection
- `hash_sha256` → cover page and Chain of Custody table (§1)
- `face_bbox` → all downstream agents for face cropping
- `ela_score` → preliminary tampering indicator in §1 table
- `face_crop_path` → VLM agent for heatmap generation

---

## agents/geometry.py

**Owner:** Veedhee
**Libraries:** `dlib`, `opencv-python`, `numpy`, `scipy`

### What it does
1. Calls dlib's `get_frontal_face_detector()` to confirm face in crop
2. Loads `shape_predictor_68_face_landmarks.dat` for 68-point landmarks
3. Computes 8 anthropometric ratios vs authentic population baselines:
   - `symmetry_index` — lateral landmark distances vs midline
   - `jaw_curvature_deg` — polynomial fit curvature of jaw landmarks 0–16
   - `ear_alignment_px` — vertical offset between landmark 0 and 16
   - `philtrum_length_mm` — nose-to-lip / face height ratio
   - `interocular_dist_px` — distance between left and right eye centres
   - `eye_aspect_ratio_l/r` — standard 6-point EAR formula per eye
   - `lip_thickness_ratio` — lip height / face height
   - `neck_face_boundary` — "smooth" or "sharp_edge" via variance jump
4. Z-scores each metric against `NORMS`; anomaly_score = mean of clipped z-scores

**[CHANGED from v1.0]:** Uses **dlib** for face detection (not RetinaFace which is handled
by preprocessing). The entry point `run(image_bgr, face_present)` takes a **numpy array**,
not a path — master_agent's `geometry_tool` reads the image with cv2 before calling it.

### Setup note
```bash
# Download once (~100 MB), place in repo root:
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```
Add `shape_predictor_68_face_landmarks.dat` to `.gitignore`. Share via shared folder.

### Input (as called by geometry_tool in master_agent)
```python
image_bgr: np.ndarray   # full image, loaded by cv2.imread
face_present: bool      # True always (gate handled by preprocess_node)
```

### Output
```python
{
    "face_bbox":            [x1, y1, x2, y2],
    "symmetry_index":       0.74,
    "jaw_curvature_deg":    11.2,
    "ear_alignment_px":     8.7,
    "philtrum_length_mm":   0.21,    # renamed from philtrum_length in v1.0
    "interocular_dist_px":  64.3,
    "eye_aspect_ratio_l":   0.31,
    "eye_aspect_ratio_r":   0.29,
    "lip_thickness_ratio":  0.22,
    "neck_face_boundary":   "sharp_edge",
    "landmark_confidence":  0.91,
    "anomaly_score":        0.884,
    "agent_applicable":     True,
}
```

### Report connection
Populates **§5.1 — Geometric & Structural Analysis** table. Every value is compared
against `AUTHENTIC_BASELINES` from contracts.py. Template auto-highlights values
outside baseline in crimson.

---

## agents/frequency_agent.py

**Owner:** Pushp
**Libraries:** `numpy`, `scipy`, `opencv-python`
**No model download needed.** Pure math only.

### What it does
**Method A — FFT Azimuthal Averaging** (Durall et al. CVPR 2020 technique)
1. Reads face crop path from `input["path"]` (not the full image — master_agent crops and saves temp)
2. 2D FFT → shift → magnitude → log scale → radial averaging
3. Computes energy excess vs expected roll-off in 3 bands: mid (16–50 cyc/px), high (51–100 cyc/px), ultra-high (>100 cyc/px)
4. Thresholds and expected offsets are in `config.json` — must be calibrated on FF++ data

**Method B — Block DCT** (Frank et al. ICML 2020 technique)
1. 8×8 block DCT across entire face crop
2. hf_ratio = mean(|AC 32–63|) / mean(|AC 1–15|)
3. Scaled to [0,1] using thresholds from config.json

**[CHANGED from v1.0]:** This agent does NOT load EfficientNet-B4. The `gan_probability`
key is currently a stub (`float(anomaly_score)`). The real EfficientNet model integration
is pending. The FFT + DCT math IS real and produces meaningful scores.

### Input (as called by frequency_tool in master_agent)
```python
# master_agent crops the face, saves a temp file, then passes:
input = {"input_type": "face_crop", "path": "temp/freq_crop_XXXXXXXX.jpg"}
```

### Output
```python
{
    "fft_mid_anomaly_db":       9.4,
    "fft_high_anomaly_db":      3.2,
    "fft_ultrahigh_anomaly_db": 0.0,    # pending ultra-band computation
    "gan_probability":          0.967,  # stub = anomaly_score until EfficientNet wired
    "upsampling_grid_detected": False,  # stub
    "anomaly_score":            0.856,
}
```

### config.json section used
```json
"frequency_agent": {
    "image_size": 224,
    "fft_bands": { "low_pct": 0.15, "mid_pct": 0.50, "high_pct": 0.75, "ultra_pct": 0.90 },
    "fft_thresholds": { "mid_expected_offset": -24.1, "high_expected_offset": -36.6,
                        "ultra_expected_offset": -55.0, "normalization_factor": 10.0 },
    "dct_thresholds": { "real_hf_ratio_min": 0.08, "fake_hf_ratio_max": 0.35, "block_size": 8 },
    "submethod_weights": { "fft": 0.55, "dct": 0.45 }
}
```

### Report connection
FFT spectrum figure and GAN probability bar in **§5.2 — GAN Artefact & Frequency-Domain Analysis**.

---

## agents/texture.py

**Owner:** Manya
**Libraries:** `scikit-image`, `scipy`, `numpy`, `opencv-python`
**No model download needed.** Pure math only.

### What it does
1. Crops face to 5 zones: forehead, nose, cheek_L, cheek_R, jaw (proportional to bbox)
2. LBP histogram (P=8, R=1, uniform) per zone — captures micro-texture
3. Gabor filter bank (4 orientations × 3 frequencies) per zone — captures directional texture
4. Earth Mover's Distance between adjacent zone histogram pairs:
   - `jaw_emd` — mean of cheek_L-to-jaw and cheek_R-to-jaw
   - `cheek_emd` — cheek_L vs cheek_R (symmetry check)
   - `neck_emd` — jaw vs neck strip below face bbox (falls back to jaw_emd × 0.9 if neck not visible)
5. Seam detected if any EMD > 0.45
6. Per-zone anomaly score = (relative Gabor deviation + LBP uniformity deviation) / 2
7. Final anomaly_score = clip(worst_emd / 0.45, 0, 1); minimum 0.70 if seam detected

### Input
```python
image_path: str   # absolute path
face_bbox:  list  # [x1, y1, x2, y2]
```

### Output (Pydantic TextureOutput, returned as dict via .model_dump())
```python
{
    "jaw_emd":         0.61,
    "neck_emd":        0.48,
    "cheek_emd":       0.22,
    "lbp_uniformity":  0.31,
    "seam_detected":   True,
    "zone_scores":     {"forehead": 0.20, "cheek_L": 0.30, "cheek_R": 0.28, "jaw": 0.80, "nose": 0.15},
    "anomaly_score":   0.895,
}
```

### Mapping to contracts.py TEXTURE_KEYS (done in master_agent report_node)
```
jaw_emd  → cheek_jaw_emd_l, cheek_jaw_emd_r  (both get same value — improve later)
neck_emd → neck_face_emd
cheek_emd → periorbital_nasal_emd, lip_chin_emd
```

### Report connection
Zone heatmap and EMD table in **§5.3 — Texture Consistency & Skin-Tone Mapping**.
`seam_detected = True` triggers finding "Blending artefact detected at jaw-neck boundary."

---

## agents/vlm.py

**Owner:** Aryan
**Libraries:** `transformers`, `torch`, `opencv-python`, `Pillow`, `matplotlib`
**Model:** `llava-hf/llava-1.5-7b-hf` (~14 GB, downloads on first run to `~/.cache/huggingface/hub/`)

**[CHANGED from v1.0]:** Model is **LLaVA-1.5-7b**, NOT BLIP-2. BLIP-2 was the original plan
but LLaVA gives better instruction-following for forensic prompts.

### What it does
**Path A — LLaVA-1.5-7b Forensic Caption**
1. Loads LLaVA-1.5-7b with `device_map="auto"` and `torch_dtype=torch.float16`
2. Sends forensic prompt: "Is this a face or not if yes? Look carefully at: skin texture, eye reflections, jaw edges, and the boundary between the face and neck. First state REAL or FAKE, then describe specific suspicious features."
3. Decodes response → extracts ASSISTANT section
4. `_parse_verdict()` scans for explicit fake/real keywords → verdict + confidence

**Path B — Grad-CAM (PLACEHOLDER)**
Currently returns `np.full((224,224), 0.5)` neutral map.
**Teammate merge point:** Replace `_run_gradcam_placeholder()` with real EfficientNet-B4 Grad-CAM.
Keep return signature: `(cam_map, saliency_score, gan_probability)`.

**Path C — Zone Classification**
9 face regions classified by mean activation: HIGH (>0.8), MID (0.5–0.8), LOW (<0.5).

**Anomaly score formula:**
```
0.50 × gan_probability + 0.30 × llava_confidence + 0.10 × saliency + 0.10 × high_ratio
```
With Grad-CAM placeholder: gan_probability=0.5 contributes neutral 0.25. LLaVA still meaningful.

### Input (ctx dict, as called by vlm_tool)
```python
ctx = {
    "image_path":     "/abs/path/to/image.jpg",
    "face_bbox":      [x1, y1, x2, y2],
    "face_crop_path": "outputs/preprocessing/photo1_normalised.jpg",  # must exist on disk
}
```

### Output
```python
{
    "heatmap_path":              "temp/heatmap_XXXXXXXXXX.png",
    "vlm_caption":               "The central facial region shows...",
    "vlm_verdict":               "FAKE",      # or "REAL" or "UNCERTAIN"
    "vlm_confidence":            0.83,
    "saliency_score":            0.50,        # neutral until Grad-CAM implemented
    "high_activation_regions":   [],          # empty until Grad-CAM implemented
    "medium_activation_regions": [],
    "low_activation_regions":    ["eyes", "nose", "mouth", ...],
    "zone_gan_probability":      0.50,        # neutral until Grad-CAM implemented
    "anomaly_score":             0.50,        # weighted: LLaVA dominates with placeholder
}
```

### Hardware note (RTX A6000 × 2)
`device_map="auto"` places LLaVA on GPU 0 (48 GB available). Model needs ~14 GB in float16.
Model loads once per process, cached in `_llava_processor` / `_llava_model` globals.
First-run download: ~14 GB. Subsequent runs: instant (local cache).

### Report connection
Full-width Grad-CAM heatmap figure + VLM caption table in **§5.4 — Explainability Heat-Map Analysis**.

---

## agents/biological_plausibility_agent.py

**Owner:** Diksha
**Libraries:** `opencv-python`, `mediapipe`, `numpy`, `scikit-learn`, `matplotlib`

**[CHANGED from v1.0]:** This agent does NOT compute rPPG from a single image
(that requires video). Instead it computes **pupil shape analysis** and **corneal
reflection IoU** — different signals that detect the same underlying GAN artifacts.

### What it does
1. **Pupil BIoU (Boundary IoU):** Otsu-threshold pupil segmentation → fit ellipse →
   measure how well the detected boundary matches the fitted ellipse.
   Real pupils are circular/elliptical; GAN pupils are often irregular.
2. **Corneal Reflection IoU:** Detects Hough circle (limbus) → bright spot mask in each eye.
   IoU between left and right eye reflection patterns. Real faces: consistent light source
   = high IoU. Deepfakes: inconsistent.
3. **Shape features:** Solidity (area/convex hull area), convexity, aspect ratio, Hu moment 1.
   GAN-synthesized pupils show lower solidity and abnormal aspect ratios.

**Note:** This is a standalone evaluation script adapted for pipeline use.
`master_agent`'s `biological_tool` calls `analyse_image()` and maps the output
to contracts.py BIOLOGICAL_KEYS via proxy mapping. No changes to this file needed.

### Input (called by biological_tool via analyse_image())
```python
img_bgr:    np.ndarray   # full image as BGR numpy array
face_mesh:  mp.FaceMesh  # MediaPipe FaceMesh instance
cfg:        dict         # CFG dict from agent file (biou_threshold, iou_threshold, etc.)
```

### Raw output (before proxy mapping by biological_tool)
```python
{
    "biou_left":          0.43,
    "biou_right":         0.51,
    "avg_biou":           0.47,
    "iou_reflect":        0.12,
    "solidity":           0.88,
    "convexity":          0.94,
    "aspect":             0.79,
    "hu1":               -3.2,
    "reflection_count":   14,
    "landmarks_found":    True,
    "prediction":         "fake",
}
```

### Proxy mapping in biological_tool (master_agent.py)
```python
"rppg_snr"             = avg_biou          (proxy: BIoU as physiological regularity proxy)
"corneal_deviation_deg"= (1-iou_reflect)*20 (proxy: IoU mismatch → angular deviation)
"micro_texture_var"    = solidity * 0.031  (proxy: low solidity → over-smoothed)
"vascular_pearson_r"   = None              (not available from single-image pupil analysis)
"anomaly_score"        = 0.6*(1-avg_biou) + 0.4*(1-min(1.0, iou_reflect*2))
```

### Report connection
Proxy values populate **§5.5 — Biological Plausibility Assessment** table.
Template auto-flags rppg_snr < 0.45 and corneal_deviation_deg > 5°.

---

## agents/metadata_agent.py

**Owner:** Sania / Aditi
**Libraries:** `piexif`, `Pillow`, `numpy`
**Note:** Runs on the original image (not the face crop).

### What it does
1. **EXIF parsing:** Camera make/model, software tag, GPS, datetime via piexif → Pillow fallback
2. **Software flag:** Checks for Photoshop, DALL-E, Stable Diffusion, Midjourney, etc. in Software tag
3. **ELA chi-squared:** Re-save at quality 95 → diff → 64-bin histogram → chi-squared vs uniform.
   Natural images → flat diff → low chi2. Edited regions → high chi2. Suspicious: chi2 > 500.
4. **Thumbnail mismatch:** Extract EXIF thumbnail → resize main image → MSE comparison. MSE > 800 = mismatch.
5. **PRNU score:** `prnu_score = var(image - GaussianBlur(image)) / var(image)`. Below 0.05 = no sensor noise.
6. **Anomaly score:** `0.40×software_flagged + 0.30×thumbnail_mismatch + 0.20×prnu_absent + 0.10×no_provenance`

**Input:** `preprocessing_json_path` (str) — path to preprocessing JSON file
**Output:** Saves `outputs/metadata/<stem>.json`, returns path

### Output (after required additions)
```python
{
    "image_path":                ...,
    "exif_camera_present":       False,
    "exif_camera_make":          None,
    "software_tag":              "Adobe Photoshop 24.0",
    "software_flagged":          True,
    "ela_chi2":                  847.3,
    "ela_map_path":              "outputs/metadata/photo1_ela_map.png",
    "thumbnail_mismatch":        True,
    "prnu_score":                0.013,
    "prnu_absent":               True,
    "anomaly_score":             0.973,
    # Required additions:
    "jpeg_quantisation_anomaly": False,    # stub
    "cosine_dist_authentic":     None,     # reference agent (future)
    "cosine_dist_fake":          None,
    "facenet_dist":              None,
    "arcface_dist":              None,
    "shape_3dmm_dist":           None,
    "reference_verdict":         "PENDING",
    "errors":                    [],
}
```

### Report connection
ELA map figure and EXIF table in **§5.6 — Metadata & Provenance Analysis**.
Reference comparison fields in **§7** (show as "—" until reference agent implemented).

---

## fusion/bayesian.py

**Owner:** Dev
**Libraries:** `numpy`

### What it does
Combines all module `anomaly_score` values into a single DeepFake Prediction Score
using log-odds Bayesian fusion with bootstrap confidence intervals.

**Formula:**
```
log_odds_total = Σ [ weight_i × log(score_i / (1 - score_i)) ]
final_score    = sigmoid(log_odds_total) = 1 / (1 + exp(-log_odds_total))
```

**Scores are clamped to [1e-6, 1-1e-6]** before log transform to avoid inf.

**Bootstrap CI:** 1000 resamples over module scores, seed=42 (deterministic).
95% CI = [2.5th, 97.5th] percentile of bootstrap distribution.

### Module weights (from contracts.py FUSION_WEIGHTS)
```python
FUSION_WEIGHTS = {
    "geometry":     0.15,
    "gan_artefact": 0.25,   # EfficientNet-B4 — uses frequency.gan_probability when available
    "frequency":    0.25,   # FFT + DCT
    "texture":      0.20,
    "vlm":          0.25,
    "biological":   0.15,
    "metadata":     0.15,
}
```

**[CHANGED from v1.0]:** Weights in v1.0 documentation were actually the **test case anomaly scores**
from DFA-2025-TC-00471, not fusion weights. Actual weights come from `contracts.py FUSION_WEIGHTS`.
The weights intentionally do NOT sum to 1.0 — they are log-odds multipliers, not a simplex.

### Decision thresholds
```python
DEEPFAKE_THRESHOLD  = 0.70   # final_score >= 0.70 → DEEPFAKE
AUTHENTIC_THRESHOLD = 0.35   # final_score <= 0.35 → AUTHENTIC; between → UNCERTAIN
```

### Output
```python
{
    "final_score":         0.957,
    "confidence_interval": [0.91, 0.98],
    "per_module_scores":   {"geometry": 0.884, "frequency": 0.912, ...},
    "decision":            "DEEPFAKE",
    "interpretation":      "Very High Confidence",
    "model_auc_roc":       0.983,
    "false_positive_rate": 0.021,
    "calibration_ece":     0.014,
    "decision_threshold":  0.70,
}
```

---

## master_agent.py

**Owner:** Dev
**Libraries:** `langgraph`, `langchain`, `langchain-ollama`, `langchain-core`
**LLM:** Mistral-7B via Ollama — used ONLY in report_node for narrative text

**[CHANGED from v1.0]:** Uses **LangGraph StateGraph**, not `create_tool_calling_agent`.
This gives explicit node topology, conditional edges, and future reflection loop support.

### LangGraph topology
```
START
  │
  ▼
preprocess_node ──(face_detected=False)──→ abort_node → END
  │
  │ (face_detected=True)
  ▼
parallel_analysis_node  (asyncio.gather over 6 agent tools)
  │
  ▼
fusion_node             (bayesian_fusion)
  │
  ▼
report_node             (Mistral-7B narrative + PDF)
  │
  ▼
END
```

### Adding a new agent (future)
1. Create agent module with `run()` returning dict with `anomaly_score`
2. Add `@tool` wrapper function in master_agent.py AGENT TOOLS section
3. Add entry to `_make_registry()` with `name`, `tool`, `invoke_args`, `fusion_module`
4. Add score key mapping in `fusion_node`
5. Add `MODULE_WEIGHTS` entry in `fusion/bayesian.py`

### Reflection loop (future, currently disabled)
Located in `should_reflect()` and commented-out conditional edge in `build_graph()`.
Activates when `0.45 ≤ final_score ≤ 0.65` and `reflection_passes < 2`.
Re-runs parallel_analysis_node with a tighter face bbox crop.

---

## report_agent/template.py

**Owner:** Saania '27
**Libraries:** `reportlab`

### Pages / sections implemented
| Section | Location in PDF |
|---|---|
| Cover — dark navy, gold accents, verdict, score | Page 1 |
| Table of Contents | Page 2 |
| §1 Chain of Custody (12 fields, CustodyField rows) | Page 3 |
| §5.1 Geometric Analysis (table, 10 rows) | Page 3–4 |
| §5.2 Frequency / GAN Analysis (table, 6 rows) | Page 4 |
| §5.3 Texture Consistency (table, 9 rows) | Page 4–5 |
| §5.4 VLM Explainability (table + HeatmapFrame figure) | Page 5–6 |
| §5.5 Biological Plausibility (table, 5 rows) | Page 6 |
| §5.6 + §7 Metadata & Reference (table, 13 rows) | Page 7 |
| §6 Bayesian Scoring (VerdictPanel + ModuleScoreBar × 7) | Page 7–8 |
| §8 Executive Summary (narrative text from Mistral or auto-generated) | Page 8 |
| §9–§10 Legal Certification + Analyst Declaration | Page 9 |

### Design tokens (class T)
```python
NAVY   = "#0B1628"    GOLD  = "#C9A84C"    CRIMSON = "#C0152A"
AMBER  = "#B86A00"    EMERALD = "#0A6B3C"
```

### Auto-highlighting logic
Every table that has a "Status" or "Finding" column uses `_flag_rows()` to
detect "FLAGGED" string and applies `CRIMSON_BG` background automatically.
No manual flagging needed — the functions derive flag conditions from the data.

---

## report_agent/generate.py

**Owner:** Saania '27
**Libraries:** `reportlab`, `ollama`

### What it does
1. Generates report ID: `DFA-{YEAR}-TC-{6-char hex}`
2. Calls Mistral-7B via `_call_ollama()` with all case data injected in prompt
3. Falls back to template.py auto-generated narrative if Ollama unavailable
4. Calls `build_report(ctx, report_path)` from template.py
5. Returns `REPORT_KEYS` dict validated by `contracts.validate()`

### Ollama setup
```bash
curl -fsSL https://ollama.com/install.sh | sh    # Linux
ollama pull mistral                               # ~4 GB download once
ollama serve                                      # keep running in background
```
The report generates correctly without Ollama — template.py produces auto-narrative.

---

## config.json

Centralised runtime config for all agents. Key sections:

```json
{
    "stub_mode": false,              // SET THIS TO FALSE before running
    "frequency_agent": {
        "image_size": 224,
        "fft_bands": { "low_pct": 0.15, "mid_pct": 0.50, "high_pct": 0.75, "ultra_pct": 0.90 },
        "fft_thresholds": { ... },   // calibrate on FF++ data for best results
        "dct_thresholds": { ... },   // calibrate on FF++ data for best results
        "submethod_weights": { "fft": 0.55, "dct": 0.45 }
    },
    "verdict_thresholds": { "fake": 0.65, "real": 0.35 },  // for reference only
    "timeouts_seconds": { "vlm": 120, "biological": 60, ... }
}
```

**Note:** `config.json` `"stub_mode"` is documentation-only in the current codebase —
no agent reads it at runtime. `frequency_agent.py` has its own `STUB_MODE = False`
variable. Update `config.json` to `"stub_mode": false` to avoid confusion.

---

## Report Reference — Page by Page

| Report Section | Page | Data Source | Template Function |
|---|---|---|---|
| Cover + verdict + score | 1 | `fusion.final_score`, `decision` | `_build_cover()` |
| SHA-256 hash | 1 | `preprocessing.hash_sha256` | `_build_cover()` |
| Table of Contents | 2 | Static | `_build_toc()` |
| §1 Chain of Custody | 3 | `preprocessing` + `metadata` agent outputs | `_build_custody()` |
| §5.1 Geometric Analysis | 3–4 | `geometry` agent output | `_build_geometry()` |
| §5.2 Frequency / GAN | 4 | `frequency` agent output | `_build_frequency()` |
| §5.3 Texture Analysis | 4–5 | `texture` agent output | `_build_texture()` |
| §5.4 VLM + Heatmap | 5–6 | `vlm` agent output | `_build_vlm()` |
| §5.5 Biological | 6 | `biological` agent output (proxy mapped) | `_build_biological()` |
| §5.6 + §7 Metadata | 7 | `metadata` agent output | `_build_metadata()` |
| §6 Bayesian Scoring | 7–8 | `fusion` output | `_build_fusion()` |
| §8 Executive Summary | 8 | Mistral-7B narrative / auto-generated | `_build_narrative()` |
| §9–§10 Legal Certification | 9 | Static + analyst config | `_build_legal()` |

---

## How to Run

### Prerequisites (one-time setup)
```bash
# 1. Python dependencies
pip install langgraph langchain langchain-community langchain-ollama langchain-core \
    transformers torch torchvision pillow piexif numpy opencv-python dlib mediapipe \
    scipy scikit-image scikit-learn reportlab ollama tqdm matplotlib pydantic \
    nest_asyncio

# 2. dlib landmark model (~100 MB, place in repo root)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# 3. Ollama + Mistral (optional — PDF still generates without it)
ollama pull mistral
ollama serve   # keep running in a separate terminal

# 4. Set config.json stub_mode to false
# Edit config.json: "stub_mode": false
```

### Running
```bash
python master_agent.py test_images/image1.jpg --analyst "Your Name"

# With full JSON output:
python master_agent.py test_images/image1.jpg --analyst "Your Name" --output-json
```

### Expected output
```
────────────────────────────────────────────────────────
  Case ID  : DFA-2026-TC-XXXXXXXX
  Decision : DEEPFAKE
  Score    : 95.7%
  CI 95%   : [0.91, 0.98]
  Report   : outputs/DFA-2026-TC-XXXXXXXX.pdf (or .json)
────────────────────────────────────────────────────────
```

### Validation tests
```bash
# Should produce score > 0.70, verdict = DEEPFAKE
python master_agent.py test_images/sample_fake.jpg

# Should produce score < 0.40, verdict = AUTHENTIC
python master_agent.py test_images/sample_real.jpg
```

---

## Week 1 Completion Checklist

### P0 — Must fix before pipeline runs
- [ ] `preprocessing_agent.py`: add `normalized_path`, `face_crop_path`, `landmarks_path`, `icc_profile`
- [ ] `geometry.py`: add 5 missing output keys; rename `philtrum_length` → `philtrum_length_mm`
- [ ] `master_agent.py`: update face_crop_path lookup (prefer new key name)
- [ ] `config.json`: set `"stub_mode": false`

### P1 — For complete report data
- [ ] `frequency_agent.py`: add `fft_ultrahigh_anomaly_db`, `gan_probability` stub, `upsampling_grid_detected` stub
- [ ] `metadata_agent.py`: add 6 reference field stubs

### P2 — Accuracy improvements
- [ ] `texture.py`: split jaw_emd into jaw_emd_l + jaw_emd_r
- [ ] `biological_tool` in master_agent: fix iou_ref clipping formula
- [ ] Calibrate FFT/DCT thresholds in config.json on FF++ dataset

### P3 — Future features
- [ ] Create `agents/reference.py` — FaceNet cosine similarity vs real/fake corpus
- [ ] Wire real EfficientNet-B4 Grad-CAM into `vlm.py`
- [ ] Wire real EfficientNet-B4 GAN classifier into `frequency_agent.py`
- [ ] Enable reflection loop in LangGraph for ambiguous scores
- [ ] Implement multi-face scene loop for group photos
- [ ] Extract ICC colour profile in preprocessing

---

*Documentation version 2.0 — MFAD Project — Corrected and updated from v1.0*
*Key changes from v1.0: VLM model is LLaVA-1.5-7b not BLIP-2 · Orchestrator is LangGraph not tool-calling agent · Biological agent computes pupil BIoU not rPPG · Fusion weights corrected · Preprocessing returns JSON path not dict · Reference agent does not yet exist*