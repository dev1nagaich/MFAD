y# MFAD — Week 1 Technical Documentation
### Multimodal Forensic Agent for Deepfake Detection (Image-Only)

**Target output:** A forensic report identical in structure to `DeepFake_Forensic_Report_DFA2025TC00471.pdf`  
**Stack:** LangChain · HuggingFace · PyTorch · OpenCV · ReportLab · Ollama (Mistral-7B)  
**Rule:** Every agent returns a Python `dict` with an `anomaly_score` key (float 0–1). No exceptions.

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Data Flow — End to End](#data-flow)
3. [contracts.py](#contractspy)
4. [agents/preprocessing.py](#agentspreprocessingpy) — aditi
5. [agents/geometry.py](#agentsgeometrypy) — Veedhee
6. [agents/frequency.py](#agentsfrequencypy) — Pushp
7. [agents/texture.py](#agentsTexturepy) — Manya
8. [agents/vlm.py](#agentsvlmpy) — Aryan
9. [agents/biological.py](#agentsbiologicalpy) — Diksha
10. [agents/metadata.py](#agentsmetadatapy) — Sania
11. [agents/reference.py](#agentsreferencepy) — (shared)
12. [fusion/bayesian.py](#fusionbayesianpy) — Dev
13. [master_agent.py](#master_agentpy) — Dev
14. [report/template.py](#reporttemplatepy) — Saania '27
15. [report/generator.py](#reportgeneratorpy) — Saania '27
16. [Report Reference — Page by Page](#report-reference)

---

## Repository Structure

```
mfad/
├── contracts.py              # shared output schema — read this first
├── master_agent.py           # LangChain AgentExecutor + orchestration
├── agents/
│   ├── preprocessing.py      # Aditi — pipeline entry, everyone depends on this
│   ├── geometry.py           # Veedhee — 68-pt landmarks, anthropometric ratios
│   ├── frequency.py          # Pushp — FFT anomaly + EfficientNet-B4 GAN detection
│   ├── texture.py            # Manya — LBP, Gabor, EMD skin zone analysis
│   ├── vlm.py                # Aryan — BLIP-2 caption + Grad-CAM heatmap
│   ├── biological.py         # Diksha — rPPG, corneal geometry, micro-texture
│   ├── metadata.py           # Sania — EXIF, ELA, PRNU, thumbnail check
│   └── reference.py          # cosine similarity vs real/fake corpus
├── fusion/
│   └── bayesian.py           # log-odds Bayesian fusion of all anomaly_scores
├── report/
│   ├── template.py           # ReportLab page layout, tables, fonts, colors
│   └── generator.py          # Mistral-7B narrative + final PDF assembly
├── test_images/
│   ├── sample_real.jpg       # verified authentic face (from Celeb-DF)
│   └── sample_fake.jpg       # known deepfake (from FaceForensics++ sample)
└── outputs/                  # generated heatmaps, PDFs go here
```

---

## Data Flow

```
INPUT IMAGE (jpg/png)
        │
        ▼
preprocessing.py  ──►  {face_bbox, hash_sha256, image_path, ela_score, exif_flags}
        │
        ├──────────────────────────────────────────────────────────────┐
        ▼                ▼              ▼           ▼         ▼        ▼
  geometry.py    frequency.py    texture.py    vlm.py   biological  metadata.py
        │                │              │           │         │        │
        └──────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                              reference.py (embedding comparison)
                                        │
                                        ▼
                               fusion/bayesian.py
                          (combines all anomaly_scores)
                                        │
                                        ▼
                              master_agent.py output JSON
                                        │
                              ┌─────────┴──────────┐
                              ▼                    ▼
                       report/template.py    report/generator.py
                       (PDF layout)          (Mistral-7B narrative)
                              └─────────┬──────────┘
                                        ▼
                               FORENSIC PDF REPORT
                         (matches DFA2025TC00471 structure)
```

**Critical dependency:** `preprocessing.py` must run first and complete before any other agent. All other agents run in parallel after it, fed by the `face_bbox` and `image_path` it returns.

---

## contracts.py

**Owner:** Everyone reads this. No one modifies output keys without team agreement.  
**Purpose:** Single source of truth for every agent's output schema. If your dict keys don't match this, the fusion will fail silently.

```python
# What each agent MUST return — keys are non-negotiable

PREPROCESSING_OUTPUT = {
    "image_path":        str,   # absolute path to verified image
    "face_bbox":         list,  # [x1, y1, x2, y2] in pixels
    "hash_sha256":       str,   # hex digest
    "anomaly_score":     float  # 0.0 = clean, 1.0 = suspicious
}

GEOMETRY_OUTPUT = {
    "symmetry_index":    float, # 0-1, lower = more asymmetric
    "jaw_curvature_deg": float, # deviation from norm in degrees
    "ear_alignment_px":  float, # pixel offset between ears
    "philtrum_length":   float, # normalised ratio
    "anomaly_score":     float
}

FREQUENCY_OUTPUT = {
    "fft_mid_anomaly_db":  float, # mid-frequency energy deviation
    "fft_high_anomaly_db": float, # high-frequency energy deviation
    "gan_probability":     float, # EfficientNet-B4 output
    "anomaly_score":       float
}

TEXTURE_OUTPUT = {
    "jaw_emd":           float, # Earth Mover's Distance, jaw zone
    "neck_emd":          float, # EMD, neck-face boundary
    "lbp_uniformity":    float, # LBP histogram uniformity score
    "seam_detected":     bool,
    "anomaly_score":     float
}

VLM_OUTPUT = {
    "heatmap_path":            str,   # path to saved heatmap overlay PNG
    "vlm_caption":             str,   # BLIP-2 forensic finding text
    "saliency_score":          float, # mean activation in face region
    "high_activation_regions": list,  # e.g. ["eyes", "jaw boundary"]
    "anomaly_score":           float
}

BIOLOGICAL_OUTPUT = {
    "rppg_snr":             float, # signal-to-noise ratio of rPPG
    "corneal_deviation_deg":float, # highlight angle mismatch between eyes
    "micro_texture_var":    float, # perioral micro-texture variance
    "anomaly_score":        float
}

METADATA_OUTPUT = {
    "exif_camera_present": bool,
    "ela_chi2":            float, # ELA chi-squared score
    "thumbnail_mismatch":  bool,
    "prnu_absent":         bool,
    "anomaly_score":       float
}

REFERENCE_OUTPUT = {
    "cosine_dist_authentic": float, # distance from real face cluster
    "cosine_dist_fake":      float, # distance from known fake cluster
    "verdict":               str,   # "CLOSER_TO_FAKE" or "CLOSER_TO_REAL"
    "anomaly_score":         float
}

FUSION_OUTPUT = {
    "final_score":         float,  # 0-1, probability of deepfake
    "confidence_interval": list,   # [low, high] at 95%
    "verdict":             str,    # "DEEPFAKE" or "LIKELY REAL"
    "per_module_scores":   dict    # {module_name: anomaly_score}
}
```

---

## agents/preprocessing.py

**Owner:** ADITI  
**Priority:** HIGHEST — everyone is blocked until this runs  
**Libraries:** `Pillow`, `piexif`, `retinaface` (or `mediapipe`), `hashlib`, `numpy`

### What it does
1. Receives raw image path from the user
2. Computes SHA-256 hash for chain-of-custody (appears on page 1 of the report)
3. Detects face region using RetinaFace — returns bounding box coordinates
4. Normalises image (resize to 512×512 within face region, convert to RGB)
5. Runs basic ELA (Error Level Analysis) to get a preliminary tampering score
6. Packages everything into the standard output dict

### Input
```python
image_path: str  # e.g. "test_images/sample.jpg"
```

### Output
```python
{
    "image_path":    "/abs/path/to/image.jpg",
    "face_bbox":     [x1, y1, x2, y2],        # pixel coords
    "hash_sha256":   "a3f9b2...",
    "hash_md5":      "c4e8a1...",
    "ela_score":     0.34,                     # 0=clean, 1=heavily altered
    "face_detected": True,
    "image_dims":    [1920, 1080],
    "anomaly_score": 0.34
}
```

### Report connection
The `hash_sha256` and `hash_md5` values populate the **Chain of Custody** table on page 2 of the report (section titled "Evidence Integrity Verification"). The `face_bbox` is used by all downstream agents to crop to the same face region.

### Stub to start with
```python
@tool
def preprocessing_agent(image_path: str) -> dict:
    """Validates image integrity, detects face, computes hash."""
    return {
        "image_path":    image_path,
        "face_bbox":     [120, 80, 420, 460],
        "hash_sha256":   "stub_hash_replace_me",
        "hash_md5":      "stub_md5_replace_me",
        "ela_score":     0.34,
        "face_detected": True,
        "image_dims":    [512, 512],
        "anomaly_score": 0.34
    }
```

---

## agents/geometry.py

**Owner:** Veedhee  
**Libraries:** `dlib`, `mediapipe`, `numpy`, `scipy`

### What it does
1. Takes the `face_bbox` from preprocessing output
2. Detects 68 facial landmarks using dlib's pretrained shape predictor
3. Computes anthropometric ratios: facial symmetry index, jaw curvature, ear alignment offset, philtrum length ratio
4. Compares ratios against known authentic norms (hardcoded reference ranges from literature)
5. Any deviation beyond ±2 standard deviations contributes to the anomaly score

### Input
```python
image_path: str   # same image
face_bbox:  list  # [x1, y1, x2, y2] from preprocessing
```

### Output
```python
{
    "symmetry_index":       0.74,  # 1.0 = perfectly symmetric
    "jaw_curvature_deg":    11.2,  # degrees deviation
    "ear_alignment_px":     8.7,   # pixel offset
    "philtrum_length":      0.21,  # normalised ratio
    "landmark_confidence":  0.91,
    "anomaly_score":        0.884
}
```

### Report connection
These values populate **Section 4.1 — Geometric & Structural Analysis** in the report. The symmetry index and jaw deviation are shown in a table with green/red highlighting based on whether they fall within normal ranges. The landmark overlay image (68 dots on the face) appears as Figure 1 in the report.

### Key note for Veedhee
Download `shape_predictor_68_face_landmarks.dat` from dlib's official release — it is ~100MB and must be in the repo root or pointed to with an absolute path. Do not commit it to git — add to `.gitignore` and share via a shared folder link.

---

## agents/frequency.py

**Owner:** Pushp  
**Libraries:** `numpy`, `scipy`, `torch`, `transformers`, `Pillow`

### What it does
**Part A — FFT Analysis (pure numpy, no model)**
1. Crop face region using `face_bbox`
2. Convert to grayscale
3. Apply 2D Fast Fourier Transform
4. Shift zero-frequency component to centre
5. Compute mean energy in mid-frequency band (radius 30–80px from centre) and high-frequency band (radius 80–200px)
6. GANs and diffusion models leave characteristic energy spikes in mid-frequency — measure deviation from natural image statistics

**Part B — GAN Artefact CNN**
1. Load `shaoanlu/faceforensics-efficientnet` from HuggingFace
2. Preprocess face crop (normalise, resize to 224×224)
3. Run forward pass — output is P(fake)
4. This is the `gan_probability` score

### Input
```python
image_path: str
face_bbox:  list
```

### Output
```python
{
    "fft_mid_anomaly_db":  9.4,   # dB above natural image baseline
    "fft_high_anomaly_db": 3.2,
    "gan_probability":     0.967, # EfficientNet output
    "freq_spectrum_path":  "outputs/fft_spectrum.png",  # saved visualisation
    "anomaly_score":       0.967
}
```

### Report connection
The FFT spectrum image (a circular heatmap showing energy distribution) appears in **Section 4.2 — Frequency Domain Analysis** of the report. The GAN probability score is shown as a large percentage with a coloured confidence bar — red if > 0.7, amber if 0.4–0.7, green otherwise.

---

## agents/texture.py

**Owner:** Manya  
**Libraries:** `scikit-image`, `scipy`, `numpy`, `opencv-python`  
**Note:** No model download needed — entirely algorithmic.

### What it does
1. Crop face into 5 zones: forehead, left cheek, right cheek, nose, jaw
2. For each zone, compute LBP (Local Binary Pattern) histogram — captures micro-texture
3. Apply Gabor filter bank (4 orientations × 3 frequencies = 12 filters) per zone
4. Compute Earth Mover's Distance (Wasserstein distance) between adjacent zone histograms — large EMD = discontinuous texture = blending seam
5. Flag `seam_detected = True` if any adjacent pair has EMD above threshold (empirically ~0.45)

### Input
```python
image_path: str
face_bbox:  list
```

### Output
```python
{
    "jaw_emd":         0.61,   # high = seam at jaw boundary
    "neck_emd":        0.48,
    "cheek_emd":       0.22,
    "lbp_uniformity":  0.31,   # lower = more artificial
    "seam_detected":   True,
    "zone_scores":     {"forehead": 0.2, "cheek_L": 0.3, "jaw": 0.8},
    "anomaly_score":   0.895
}
```

### Report connection
The zone-level scores map to the **facial region heatmap** shown in Section 4.3 of the report — a face outline with zones colour-coded red/amber/green based on their EMD score. The `seam_detected` flag triggers the finding "Blending artefact detected at jaw-neck boundary" in the narrative section.

---

## agents/vlm.py

**Owner:** Aryan  
**Libraries:** `transformers`, `torch`, `pytorch-grad-cam`, `opencv-python`, `Pillow`  
**Model:** `Salesforce/blip2-opt-2.7b` (~5GB, downloads on first run)

### What it does
**Part A — Grad-CAM Heatmap**
1. Load BLIP-2's vision encoder (ViT backbone)
2. Hook into the last attention layer: `model.vision_model.encoder.layers[-1].layer_norm1`
3. Run forward pass with the face image
4. Compute gradient-weighted class activation map
5. Resize activation map to original image dimensions
6. Overlay as colour heatmap (blue=low activation, red=high activation) on original image
7. Save overlay as PNG to `outputs/`

**Part B — VLM Forensic Caption**
1. Run 3 structured forensic prompts against the image:
   - "Are there unnatural boundaries or blending artifacts on the face?"
   - "Is the skin texture consistent across the entire face?"
   - "Are the lighting and shadows physically consistent?"
2. Decode BLIP-2's responses
3. Concatenate into a single forensic finding string

**Part C — Saliency Score**
1. Extract grayscale activation map from Part A
2. Compute mean activation within the central face region (middle 60% of bbox)
3. This scalar becomes the `saliency_score` and `anomaly_score`

### Input
```python
image_path: str
face_bbox:  list
```

### Output
```python
{
    "heatmap_path":            "outputs/heatmap_overlay.png",
    "vlm_caption":             "The central facial region shows unnatural texture smoothing near the jaw boundary. Skin tone appears inconsistent between cheek zones. Shadow direction is inconsistent with ambient lighting.",
    "saliency_score":          0.91,
    "high_activation_regions": ["jaw boundary", "eyes", "nose bridge"],
    "anomaly_score":           0.931
}
```

### Report connection
This is the most visually prominent section of the report. The heatmap overlay appears as a full-width figure in **Section 4.4 — VLM Explainability Analysis**. The `vlm_caption` text becomes the verbatim "AI Forensic Findings" paragraph. The `high_activation_regions` list is displayed as a bullet list of flagged zones. This section is what makes the report look credible and scientific to a reviewer.

### Critical setup note
Use `device_map="auto"` when loading BLIP-2 — this handles automatic CPU offloading if you don't have enough GPU VRAM. Loading takes 2–3 minutes on first run, do not kill the process. The model stays in RAM for the session.

---

## agents/biological.py

**Owner:** Diksha  
**Libraries:** `opencv-python`, `mediapipe`, `numpy`  
**Note:** rPPG from a single image means analysing colour channel variance across skin zones — not video-based rPPG.

### What it does
1. **rPPG SNR:** Extract the green channel (most sensitive to blood volume pulse) from forehead and cheek regions. Compute frequency-domain SNR of the signal. Real faces have a characteristic 0.75–2.5Hz component; synthesised faces often show flat or noisy spectra.
2. **Corneal highlight consistency:** Use MediaPipe to locate both eyes. Extract specular highlight position in each cornea. The angle and position of highlights must be consistent between eyes for a real image (same light source). Discrepancy > 15° = anomalous.
3. **Micro-texture variance:** Compute local variance of pixel intensities in the perioral region (mouth area). GAN-generated skin tends to be over-smoothed — very low variance is suspicious.

### Input
```python
image_path: str
face_bbox:  list
```

### Output
```python
{
    "rppg_snr":              2.1,    # dB, <3.0 is suspicious
    "corneal_deviation_deg": 22.4,   # degrees, >15 is suspicious
    "micro_texture_var":     0.012,  # <0.02 suggests over-smoothing
    "highlight_positions":   {"left": [x,y], "right": [x,y]},
    "anomaly_score":         0.826
}
```

### Report connection
Values appear in **Section 4.5 — Biological Plausibility Assessment** with a small diagram showing the corneal highlight positions marked on an eye close-up. The rPPG spectrum plot (frequency vs power) is shown as a small chart. Low micro-texture variance triggers the finding "Perioral region shows anomalous smoothing consistent with GAN synthesis."

---

## agents/metadata.py

**Owner:** Aditi (same file as preprocessing, same PR)  
**Libraries:** `piexif`, `Pillow`, `numpy`, `hashlib`

### What it does
1. **EXIF parsing:** Check for camera make/model, GPS data, software tag, original datetime. Real photos have these; deepfakes often have stripped or fabricated EXIF.
2. **Software tag check:** If EXIF shows Photoshop, DALL-E, Stable Diffusion, or similar — flag it.
3. **ELA (Error Level Analysis):** Re-save the image at 95% JPEG quality, compute absolute difference from original. Regions with abnormally high difference were recently edited/inserted.
4. **Thumbnail mismatch:** Many cameras embed a thumbnail in EXIF. Extract it and compare with the main image — mismatch indicates post-processing.
5. **PRNU absence:** Real cameras leave a unique sensor noise pattern (PRNU). Absence of any detectable PRNU signature is suspicious for synthetic images.

### Input
```python
image_path: str
```

### Output
```python
{
    "exif_camera_present":  False,
    "software_tag":         "Adobe Photoshop 24.0",
    "ela_chi2":             847.3,   # chi-squared score, >500 is suspicious
    "ela_map_path":         "outputs/ela_map.png",
    "thumbnail_mismatch":   True,
    "prnu_absent":          True,
    "anomaly_score":        0.973
}
```

### Report connection
The ELA map appears in **Section 4.6 — Metadata & Provenance Analysis** alongside a table of EXIF findings with green checkmarks and red crosses. The thumbnail mismatch and PRNU absence appear as specific findings: "Digital fingerprint analysis reveals absence of characteristic sensor noise pattern."

---

## agents/reference.py

**Owner:** shared utility  
**Libraries:** `deepface`, `numpy`, `os`

### What it does
1. Load `test_images/reference/real/` — folder of 10–20 verified authentic images
2. Load `test_images/reference/fake/` — folder of 10–20 known deepfakes from Celeb-DF
3. Compute 512-dimensional FaceNet embedding for the input image
4. Compute cosine distance to the centroid of real cluster and fake cluster
5. Verdict = whichever cluster it is closer to

### Input
```python
image_path: str
```

### Output
```python
{
    "cosine_dist_authentic": 0.71,
    "cosine_dist_fake":      0.18,
    "verdict":               "CLOSER_TO_FAKE",
    "embedding_norm":        0.994,
    "anomaly_score":         0.910
}
```

### Report connection
This produces the **Comparative Reference Analysis** section — a small scatter plot showing the input image's embedding position relative to the real and fake clusters (Figure 5 in the sample report).

---

## fusion/bayesian.py

**Owner:** Dev 
**Libraries:** `numpy`

### What it does
Combines all 7 module `anomaly_score` values into a single DeepFake Prediction Score using log-odds Bayesian fusion.

**The math:** Each module score is treated as P(fake | evidence from that module). The prior is P(fake) = 0.5 (uninformative). Each score is converted to log-odds, weighted by that module's reliability weight, and summed. The total log-odds is converted back to a probability.

```
log_odds_total = Σ [ weight_i × log(score_i / (1 - score_i)) ]
final_score = sigmoid(log_odds_total) = 1 / (1 + exp(-log_odds_total))
```

### Module weights (based on published AUC from papers)
```python
MODULE_WEIGHTS = {
    "geometry":   0.884,
    "frequency":  0.967,
    "texture":    0.895,
    "vlm":        0.931,
    "biological": 0.826,
    "metadata":   0.973,
    "reference":  0.910,
}
```

### Output
```python
{
    "final_score":         0.957,         # DeepFake Prediction Score
    "confidence_interval": [0.91, 0.98],  # 95% CI
    "verdict":             "DEEPFAKE",    # threshold: 0.70
    "per_module_scores": {
        "geometry":   0.884,
        "frequency":  0.967,
        "texture":    0.895,
        "vlm":        0.931,
        "biological": 0.826,
        "metadata":   0.973,
        "reference":  0.910,
    }
}
```

### Report connection
The `final_score` (expressed as percentage: 95.7%) is the large bold number on the cover page of the report — the most prominent element. The `confidence_interval` appears beneath it. The `per_module_scores` table appears in Section 3 as the "Module Confidence Matrix."

---

## master_agent.py

**Owner:** Dev 
**Libraries:** `langchain`, `langchain-community`, `ollama`  
**LLM:** Mistral-7B via Ollama — run `ollama pull mistral` before testing

### What it does
1. Registers all 7 sub-agents as LangChain `@tool` functions
2. Creates a `ChatPromptTemplate` with a forensic system prompt instructing the LLM to call agents methodically
3. Creates a `create_tool_calling_agent` bound to local Mistral via Ollama
4. Wraps in `AgentExecutor` with `return_intermediate_steps=True` — this captures every agent's raw output
5. After execution, extracts per-agent dicts from `intermediate_steps`
6. Calls `bayesian_fusion()` with the `anomaly_score` from each agent
7. Returns a single master JSON dict that becomes the report's data source

### System prompt (key part)
```
You are a forensic AI master agent. For each image:
1. Call preprocessing_agent FIRST.
2. Call all other agents using the face_bbox from preprocessing.
3. Collect every agent's anomaly_score.
4. Return all findings as structured JSON.
Be methodical. Do not skip agents.
```

### Final output shape
```python
{
    "case_id":       "DFA-2026-TC-XXXXX",  # auto-generated
    "image_path":    "...",
    "timestamp":     "2026-03-17T22:14:00",
    "agent_outputs": {
        "preprocessing":  { ...full dict... },
        "geometry":       { ...full dict... },
        "frequency":      { ...full dict... },
        "texture":        { ...full dict... },
        "vlm":            { ...full dict... },
        "biological":     { ...full dict... },
        "metadata":       { ...full dict... },
        "reference":      { ...full dict... },
    },
    "fusion": {
        "final_score":         0.957,
        "confidence_interval": [0.91, 0.98],
        "verdict":             "DEEPFAKE",
        "per_module_scores":   { ... }
    }
}
```

This JSON dict is passed directly to `report/generator.py`.

---

## report/template.py

**Owner:** Saania '27  
**Libraries:** `reportlab`

### What it does
Defines the visual layout of every page in the PDF. Think of this as the CSS of the report — no content, just structure, fonts, colours, and positioning.

### Pages to implement (from the sample report)

**Page 1 — Cover / Case Summary**
- Top: Logo area + "DIGITAL FORENSICS ANALYSIS REPORT" title in dark navy
- Case ID, submission date, analyst name
- Large bold deepfake score (e.g. "95%") centred with a red/amber/green colour bar
- Verdict badge: "CLASSIFIED: SYNTHETIC MEDIA DETECTED"
- SHA-256 hash of the input image
- Bottom: classification footer with legal warning text

**Page 2 — Chain of Custody**
- Table: hash values at each pipeline stage (intake → preprocessing → analysis)
- Analyst credentials section
- ISO/IEC 27037 compliance statement

**Page 3 — Executive Summary**
- 3-column module confidence matrix table (module name | score | status)
- Status: green "PASS" / red "ANOMALOUS" based on threshold 0.70
- One-paragraph executive finding

**Pages 4–10 — Per-Module Analysis Sections**
- Section heading + module description
- Input/output parameter table
- Key findings text (from VLM narrative)
- Figure: heatmap, FFT spectrum, facial landmark overlay, ELA map, etc.
- Module confidence score shown as a horizontal progress bar

**Final Page — Legal Certification**
- Expert analyst declaration
- ISO/IEC 27037, Daubert Standard, SWGDE compliance checkboxes
- Signature block

### Key ReportLab setup
```python
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

PAGE_WIDTH, PAGE_HEIGHT = A4  # 595 x 842 pts
MARGIN = 50

# Colour palette from the sample report
NAVY    = colors.HexColor('#1B2A4A')
RED     = colors.HexColor('#C0392B')
AMBER   = colors.HexColor('#E67E22')
GREEN   = colors.HexColor('#27AE60')
LIGHT   = colors.HexColor('#F4F6F9')
```

---

## report/generator.py

**Owner:** Saania '27  
**Libraries:** `reportlab`, `ollama`, `json`, `os`

### What it does
1. Receives the master JSON dict from `master_agent.py`
2. Calls Mistral-7B via Ollama to generate narrative text for each section — passing the module's findings as structured context
3. Calls `template.py` functions to lay out each page
4. Embeds image files (heatmap, ELA map, FFT spectrum) as figures
5. Saves final PDF to `outputs/DFA-{case_id}.pdf`

### LLM prompt pattern for each section
```python
prompt = f"""
You are writing a forensic report section for a digital forensics expert.
Module: VLM Explainability Analysis
Findings: {vlm_output['vlm_caption']}
Saliency score: {vlm_output['saliency_score']}
High activation regions: {vlm_output['high_activation_regions']}

Write a 3-sentence technical forensic paragraph. Use precise language.
Do not speculate. State only what the data shows.
"""
```

### Output
```
outputs/DFA-2026-TC-XXXXX.pdf   ← the final forensic report
```

This PDF must visually match `DeepFake_Forensic_Report_DFA2025TC00471.pdf` as closely as possible.

---

## Report Reference — Page by Page

This section maps every section of the sample report to the agent/file that produces its content.

| Report Section | Page | Data Source | File |
|---|---|---|---|
| Case cover + DeepFake score | 1 | `fusion.final_score` | `generator.py` |
| SHA-256 hash | 1 | `preprocessing.hash_sha256` | `generator.py` |
| Chain of custody table | 2 | All agent timestamps + hashes | `generator.py` |
| Module confidence matrix | 3 | `fusion.per_module_scores` | `generator.py` |
| Section 4.1 — Geometric analysis | 4 | `geometry` agent output | `generator.py` |
| Landmark overlay figure | 4 | Saved by `geometry.py` | `template.py` |
| Section 4.2 — Frequency analysis | 5 | `frequency` agent output | `generator.py` |
| FFT spectrum figure | 5 | Saved by `frequency.py` | `template.py` |
| Section 4.3 — Texture analysis | 6 | `texture` agent output | `generator.py` |
| Zone heatmap figure | 6 | Computed in `texture.py` | `template.py` |
| Section 4.4 — VLM explainability | 7 | `vlm` agent output | `generator.py` |
| Grad-CAM heatmap figure | 7 | `vlm.heatmap_path` | `template.py` |
| Section 4.5 — Biological plausibility | 8 | `biological` agent output | `generator.py` |
| Section 4.6 — Metadata & provenance | 9 | `metadata` agent output | `generator.py` |
| ELA map figure | 9 | `metadata.ela_map_path` | `template.py` |
| Reference embedding scatter | 10 | `reference` agent output | `generator.py` |
| Legal certification + signature | 11 | Static template | `template.py` |

---

## Week 1 Completion Checklist

- [ ] All agents return correct dict schema matching `contracts.py`
- [ ] `preprocessing.py` runs successfully on both test images
- [ ] All agents run successfully using the preprocessing output
- [ ] `bayesian_fusion()` produces a score between 0 and 1
- [ ] `master_agent.py` orchestrates all agents end-to-end via LangChain
- [ ] PDF report generates without errors
- [ ] PDF visually matches sample report structure page by page
- [ ] Run on `sample_fake.jpg` → score > 0.70, verdict = DEEPFAKE
- [ ] Run on `sample_real.jpg` → score < 0.40, verdict = LIKELY REAL

---

*Documentation version 1.0 — Week 1 — MFAD Project*