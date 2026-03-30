# MFAD Agent Review — What Needs Fixing Before Integration
# ==========================================================
# Reviewed against: contracts.py, master_agent.py tool wrappers,
# and DEEPFAKE_IMAGE_DETECTION_DOCUMENTATION_INITIAL.md
# ============================================================

---

## 1. agents/preprocessing_agent.py  ✅ Nearly complete

**Owner:** Aditi

### What's correct
- SHA-256 + MD5 hashing works as documented.
- ELA score computation is correct.
- Saves normalised image to `outputs/preprocessing/<stem>_normalised.jpg`.
- Returns a JSON file path (not a dict) — master_agent's `preprocessing_tool`
  opens the JSON and inlines the dict, so this is fine.

### What's missing / wrong

| Issue | Fix needed |
|---|---|
| Output JSON has `"normalised_img_path"` but contracts.py PREPROCESSING_KEYS expects `"normalized_path"` and `"face_crop_path"` | Add `"normalized_path"` key (same value as `normalised_img_path`) and `"face_crop_path"` (same — it IS the face crop) to the saved JSON |
| `"landmarks_path"` key is missing from output | Add `"landmarks_path": ""` (empty string for now; geometry agent produces this) |
| `"icc_profile"` key missing from output | Add `"icc_profile": ""` |
| `anomaly_score` is set to `ela_score` directly. ELA alone is a weak proxy — inflated on PNGs | Acceptable for now, document the limitation |

### Required addition to output dict
```python
result = {
    ...existing keys...,
    "normalized_path":  norm_img_path,   # alias for contracts.py
    "face_crop_path":   norm_img_path,   # same file — the face crop
    "landmarks_path":   "",              # filled by geometry agent later
    "icc_profile":      "",              # not extracted yet
}
```

---

## 2. agents/geometry.py  ⚠️ Significant schema mismatch

**Owner:** Veedhee

### What's correct
- dlib 68-point landmark detection implemented correctly.
- `compute_symmetry_index`, `compute_jaw_curvature_deg`,
  `compute_ear_alignment_px`, `compute_philtrum_length` all look sound.
- Anomaly score z-scoring logic is correct.

### What's wrong / missing

| Issue | Fix needed |
|---|---|
| Entry point is `run(image_bgr, face_present)` — takes a numpy array | master_agent passes `image_path` (str) + `face_bbox` (list). `geometry_tool` reads the image with cv2 and passes the array — **this is handled in master_agent** — no change needed in geometry.py UNLESS you want it to accept a path directly |
| Output dict does NOT include many contracts.py GEOMETRY_KEYS fields | See table below |
| `philtrum_length` in output vs `philtrum_length_mm` in contracts.py | Rename to `philtrum_length_mm` (or add alias) |
| `interocular_dist_px`, `eye_aspect_ratio_l`, `eye_aspect_ratio_r`, `lip_thickness_ratio`, `neck_face_boundary` — all missing | Add these computations (details below) |

### Missing computations to add

```python
# interocular_dist_px — distance between eye centres (landmarks 36 and 45)
from scipy.spatial import distance as dist
def compute_interocular_dist(lm):
    left_eye_centre  = lm[36:42].mean(axis=0)
    right_eye_centre = lm[42:48].mean(axis=0)
    return round(float(dist.euclidean(left_eye_centre, right_eye_centre)), 4)

# eye_aspect_ratio (EAR) — standard 6-point formula per eye
def compute_ear(eye_pts):
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return round(float((A + B) / (2.0 * C)), 4)

# lip_thickness_ratio — vertical height of lips / face height
def compute_lip_thickness_ratio(lm):
    upper_lip_top    = lm[51]   # centre of upper lip top edge
    lower_lip_bottom = lm[57]   # centre of lower lip bottom edge
    lip_height = dist.euclidean(upper_lip_top, lower_lip_bottom)
    face_height = dist.euclidean(lm[8], lm[27])
    return round(float(lip_height / (face_height + 1e-6)), 4)

# neck_face_boundary — "smooth" vs "sharp_edge"
# Simplest proxy: compare texture variance at bottom edge of face_bbox
# vs interior of face. Large variance jump = sharp_edge.
# For now return "smooth" as default — biological agent covers this better.
def compute_neck_face_boundary(face_crop_bgr):
    h = face_crop_bgr.shape[0]
    bottom_strip = face_crop_bgr[int(h*0.85):, :]
    interior     = face_crop_bgr[int(h*0.40):int(h*0.70), :]
    if bottom_strip.size == 0 or interior.size == 0:
        return "smooth"
    var_bottom = float(np.var(bottom_strip.astype(float)))
    var_interior = float(np.var(interior.astype(float)))
    return "sharp_edge" if var_bottom > var_interior * 1.8 else "smooth"
```

### Correct return dict
```python
return {
    "face_bbox":            face_bbox,
    "symmetry_index":       symmetry_index,
    "jaw_curvature_deg":    jaw_curvature_deg,
    "ear_alignment_px":     ear_alignment_px,
    "philtrum_length_mm":   philtrum_length,   # renamed
    "interocular_dist_px":  interocular_dist_px,
    "eye_aspect_ratio_l":   ear_l,
    "eye_aspect_ratio_r":   ear_r,
    "lip_thickness_ratio":  lip_ratio,
    "neck_face_boundary":   neck_boundary,
    "landmark_confidence":  landmark_confidence,
    "anomaly_score":        anomaly_score,
    "agent_applicable":     True,
}
```

---

## 3. agents/frequency_agent.py  ⚠️ Input contract mismatch

**Owner:** Pushp

### What's correct
- FFT radial spectrum (Method A) and Block DCT (Method B) are both solid.
- Config-driven thresholds via `config.json` — good design.
- Returns `fft_mid_anomaly_db`, `fft_high_anomaly_db`, `anomaly_score`.

### What's wrong / missing

| Issue | Fix needed |
|---|---|
| `run(input: dict)` expects `input["path"]` — a pre-cropped face image path | master_agent's `frequency_tool` already handles this: it crops the face, saves a temp file, passes its path. No change needed in frequency_agent.py itself. |
| Missing output keys: `fft_ultrahigh_anomaly_db`, `gan_probability`, `upsampling_grid_detected` | These are in contracts.py FREQUENCY_KEYS but NOT computed in frequency_agent.py |
| `gan_probability` not present (docs say EfficientNet-B4 for this) | The model integration is not implemented. Add a stub: `"gan_probability": float(np.clip(anomaly_score * 1.05, 0, 1))` until the real model is wired |
| `fft_ultrahigh_anomaly_db` — third FFT band not computed | Add a third band above `high_pct` in `_run_fft_analysis()` |
| `upsampling_grid_detected` — DCGAN 4x4 grid check not implemented | Add as `False` stub for now |

### Required additions to run() output
```python
output = {
    "fft_mid_anomaly_db":       fft_result["fft_mid_anomaly_db"],
    "fft_high_anomaly_db":      fft_result["fft_high_anomaly_db"],
    "fft_ultrahigh_anomaly_db": fft_result.get("fft_ultra_anomaly_db", 0.0),  # add to _run_fft_analysis
    "gan_probability":          float(np.clip(anomaly_score, 0.0, 1.0)),       # stub until EfficientNet wired
    "upsampling_grid_detected": False,                                          # stub
    "anomaly_score":            anomaly_score,
}
```

### Add ultra-high band to _run_fft_analysis()
```python
ultra_end = int(_FFT.get("ultra_pct", 0.90) * max_r)
ultra_energy  = radial_mean[high_end:ultra_end].mean() if ultra_end > high_end else high_energy
expected_ultra = low_energy + _FTHR.get("ultra_expected_offset", -55.0)

return {
    "fft_mid_anomaly_db":      float(max(0.0, mid_energy  - expected_mid)),
    "fft_high_anomaly_db":     float(max(0.0, high_energy - expected_high)),
    "fft_ultra_anomaly_db":    float(max(0.0, ultra_energy - expected_ultra)),
}
```

---

## 4. agents/texture.py  ✅ Mostly correct

**Owner:** Manya

### What's correct
- LBP histograms, Gabor filter bank, EMD computation all implemented.
- Pydantic TextureOutput schema with correct fields.
- `run_texture_agent(image_path, face_bbox)` signature matches master_agent usage.

### What's wrong / missing

| Issue | Fix needed |
|---|---|
| Output has `jaw_emd`, `neck_emd`, `cheek_emd` but contracts.py TEXTURE_KEYS expects `forehead_cheek_emd`, `cheek_jaw_emd_l`, `cheek_jaw_emd_r`, `periorbital_nasal_emd`, `lip_chin_emd`, `neck_face_emd` | These are mapped in master_agent's report_node (jaw_emd → cheek_jaw_emd_l etc.). The agent itself is fine but the mapping is approximate. Ideally add separate zone pair EMDs. |
| `lbp_uniformity` is the max histogram bin, not a ratio | Rename internally for clarity but the value is usable |
| `zone_scores` key: values for `cheek_R` missing from stub | Fixed in full implementation — just verify |

### The mapping in master_agent (report_node) is:
```python
"forehead_cheek_emd":    tex.get("jaw_emd"),     # approximate — improve later
"cheek_jaw_emd_l":       tex.get("jaw_emd"),
"cheek_jaw_emd_r":       tex.get("jaw_emd"),
"periorbital_nasal_emd": tex.get("cheek_emd"),
"lip_chin_emd":          tex.get("cheek_emd"),
"neck_face_emd":         tex.get("neck_emd"),
```
This works for the report but the values are duplicated. When Manya has time,
add distinct per-pair EMD computations to get accurate values in the report.

---

## 5. agents/vlm.py  ⚠️ Output schema has extra keys; Grad-CAM is placeholder

**Owner:** Aryan

### What's correct
- LLaVA-1.5-7b integration with forensic prompt works correctly.
- Zone classification into HIGH / MID / LOW activation regions.
- Heatmap PNG save logic is solid.
- `_parse_verdict()` keyword scanning is reasonable.
- `_fallback()` gracefully handles missing inputs.

### What's wrong / missing

| Issue | Fix needed |
|---|---|
| Grad-CAM returns a neutral `np.full(0.5)` placeholder | This is documented in the file. Aryan needs to wire in EfficientNet-B4 via `pytorch-grad-cam`. The placeholder is acceptable for now. |
| `VLMAgent.run(ctx)` expects `ctx["face_crop_path"]` — must exist on disk | master_agent passes `preprocessing["normalised_img_path"]` as `face_crop_path`. Make sure preprocessing saves the file before VLM runs. |
| `validate(output, VLM_KEYS, "VLMAgent")` call — VLM_KEYS in contracts.py includes `"vlm_verdict"` and `"vlm_confidence"` but the old stub master_agent didn't have these | The real vlm.py DOES return them. No issue — just confirming alignment. |
| model loading: `LlavaProcessor` / `LlavaForConditionalGeneration` — these are correct HuggingFace classes for LLaVA-1.5. Correct. | No change needed. |

### Call contract (what master_agent passes)
```python
ctx = {
    "image_path":     image_path,    # str — original image
    "face_bbox":      face_bbox,     # list [x1,y1,x2,y2]
    "face_crop_path": face_crop_path, # str — 512x512 normalised crop from preprocessing
}
agent = VLMAgent()
result = agent.run(ctx)
```
This is what `vlm_tool` in master_agent does. ✅

---

## 6. agents/biological_plausibility_agent.py  ⚠️ Not written for this pipeline

**Owner:** Diksha

### What's correct
- Pupil BIoU (boundary IoU), corneal reflection IoU, shape features (solidity, convexity, aspect, Hu moment) all implemented.
- `analyse_image(img_bgr, face_mesh, cfg)` returns a clean result dict.
- Logistic Regression evaluation mode for dataset benchmarking works.

### What's wrong — this is a standalone classifier, NOT a pipeline agent

The file is structured as an **evaluation script** (runs on a whole dataset),
not as a single-image pipeline agent. It does not:
- Accept `image_path` + `face_bbox` as its primary interface
- Return `anomaly_score` directly

**master_agent's `biological_tool` wrapper handles all of this**, calling
`analyse_image()` directly and mapping the output. No change needed to
the agent file itself — the tool wrapper is the adapter.

### Key mapping issue documented (for Diksha)
The documentation says biological agent should return:
```
rppg_snr, corneal_deviation_deg, micro_texture_var, vascular_pearson_r
```
But the actual implementation computes:
```
avg_biou, iou_reflect, solidity, convexity, aspect, hu1
```
These measure the same underlying phenomena (GAN face synthesis quality)
through different proxies. The master_agent tool wrapper maps them:
```python
"rppg_snr"             → avg_biou           (proxy)
"corneal_deviation_deg"→ (1 - iou_reflect) * 20  (scaled proxy)
"micro_texture_var"    → solidity * 0.031   (scaled proxy)
"vascular_pearson_r"   → None               (not available)
```
This is clearly labelled as proxy mapping in the tool wrapper. The report
template will display these values correctly.

---

## 7. agents/metadata_agent.py  ✅ Correct and production-ready

**Owner:** Sania / Aditi

### What's correct
- EXIF parsing (piexif + Pillow fallback) is solid.
- ELA chi-squared implementation is correct and well-documented.
- Thumbnail MSE mismatch is correctly implemented.
- PRNU proxy via Gaussian residual variance is reasonable.
- `anomaly_score` computed as weighted combination of 4 signals.
- Saves output JSON to `outputs/metadata/<stem>.json` and returns path.
- LangChain `BaseTool` wrapper (`MetadataAgent`) is complete.

### What's missing vs contracts.py METADATA_KEYS

| Missing key | Note |
|---|---|
| `jpeg_quantisation_anomaly` | Not implemented. Add `False` stub. |
| `cosine_dist_authentic`, `cosine_dist_fake`, `facenet_dist`, `arcface_dist`, `shape_3dmm_dist`, `reference_verdict` | These belong to the reference agent (§7), not metadata. Contracts.py merged them into METADATA_KEYS. Add as `None` stubs. |

### Required stubs to add to result dict
```python
result = {
    ...existing keys...,
    "jpeg_quantisation_anomaly": False,   # not yet implemented
    "cosine_dist_authentic":     None,    # reference agent (future)
    "cosine_dist_fake":          None,
    "facenet_dist":              None,
    "arcface_dist":              None,
    "shape_3dmm_dist":           None,
    "reference_verdict":         "PENDING",
}
```

---

## 8. Documentation .md file — Errors Found

The following items in `DEEPFAKE_IMAGE_DETECTION_DOCUMENTATION_INITIAL.MD`
are incorrect relative to the actual codebase:

| Location in doc | What it says | What the code actually does |
|---|---|---|
| `agents/vlm.py` section | "Model: `Salesforce/blip2-opt-2.7b`" | vlm.py uses **LLaVA-1.5-7b** (`llava-hf/llava-1.5-7b-hf`), not BLIP-2 |
| `agents/biological.py` section | "rPPG SNR: Extract the green channel... 0.75–2.5Hz component" | The actual agent computes **pupil BIoU + corneal highlight IoU** — video-based rPPG is not possible from a single image. The doc describes a video approach. |
| `contracts.py` section (doc version) | Shows `REFERENCE_OUTPUT` as a separate contract | Actual contracts.py merges reference keys into `METADATA_KEYS`. There is no separate `REFERENCE_OUTPUT`. |
| `agents/preprocessing.py` section (doc) | Returns `hash_md5` and `ela_score` directly in dict | Actual agent saves a **JSON file** and returns the **file path** (str), not the dict. All downstream agents must open the JSON. |
| `agents/metadata.py` section (doc) | "Owner: Aditi (same file as preprocessing, same PR)" | They are **separate files**: `preprocessing_agent.py` and `metadata_agent.py`. |
| `fusion/bayesian.py` section (doc) | `MODULE_WEIGHTS = {geometry: 0.884, frequency: 0.967, ...}` | These are the **module anomaly scores** from the reference test case, not weights. Actual weights in `contracts.py FUSION_WEIGHTS` are `{geometry: 0.15, frequency: 0.25, ...}`. The doc confused test scores with fusion weights. |
| `master_agent.py` section (doc) | "Creates `create_tool_calling_agent` bound to local Mistral via Ollama" | Actual master_agent uses **LangGraph StateGraph**, not `create_tool_calling_agent`. Mistral is only used in `report_node` for narrative text. |
| `agents/reference.py` section (doc) | Describes FaceNet cosine similarity agent | This agent **does not exist yet** in the codebase. All reference keys in contracts.py are stubs returning `None`. |
| `config.json` section (doc) | Not mentioned | `config.json` has `"stub_mode": true` — all agents are in stub mode. The doc doesn't mention this flag. |

---

## Summary — Priority Order for Fixes

| Priority | Agent | Fix |
|---|---|---|
| P0 | `preprocessing_agent.py` | Add `normalized_path`, `face_crop_path`, `landmarks_path`, `icc_profile` keys to output JSON |
| P0 | `geometry.py` | Add `interocular_dist_px`, `eye_aspect_ratio_l/r`, `lip_thickness_ratio`, `neck_face_boundary`; rename `philtrum_length` → `philtrum_length_mm` |
| P1 | `frequency_agent.py` | Add `fft_ultrahigh_anomaly_db`, `gan_probability` stub, `upsampling_grid_detected` stub to output |
| P1 | `metadata_agent.py` | Add `jpeg_quantisation_anomaly`, reference field stubs to output |
| P2 | `biological_plausibility_agent.py` | No code change needed — master_agent wrapper handles the mapping |
| P2 | `texture.py` | Add separate per-zone-pair EMD keys for more accurate report values |
| P3 | `vlm.py` | Wire real EfficientNet-B4 Grad-CAM (currently placeholder) |
| P3 | `agents/reference.py` | Create this agent (FaceNet embedding cosine similarity vs real/fake clusters) |