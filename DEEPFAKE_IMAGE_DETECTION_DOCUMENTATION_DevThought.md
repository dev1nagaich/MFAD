# MFAD — Multimodal Forensic Agent for Deepfake Detection
### Technical Documentation · Version 3.0

**Target output:** Forensic PDF report identical in structure to `DeepFake_Forensic_Report_DFA2025TC00471.pdf`  
**Stack:** LangGraph · LangChain · CLIP · BLIP-2 · PyTorch · OpenCV · ReportLab · Ollama (Mistral-7B)  
**Rule:** Every agent returns a Python `dict` with `anomaly_score` (float 0–1) **and** `agent_applicable` (bool). No exceptions.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Key Design Decisions](#2-key-design-decisions)
3. [Repository Structure](#3-repository-structure)
4. [Data Flow — End to End](#4-data-flow--end-to-end)
5. [MFADState — Shared State Object](#5-mfadstate--shared-state-object)
6. [contracts.py — Output Schemas](#6-contractspy--output-schemas)
7. [Stage 0 — preprocessing.py](#7-stage-0--preprocessingpy)
8. [Stage 1 — content_router.py](#8-stage-1--content_routerpy)
9. [Stage 2 — Agent Pools](#9-stage-2--agent-pools)
   - [9.1 Face Pool (ACTIVE)](#91-face-pool-active)
   - [9.2 Scene Pool (FUTURE)](#92-scene-pool-future)
   - [9.3 Forgery Pool (FUTURE)](#93-forgery-pool-future)
   - [9.4 Doc Pool (FUTURE)](#94-doc-pool-future)
   - [9.5 Synth Pool (FUTURE)](#95-synth-pool-future)
10. [Stage 3 — Universal Agents](#10-stage-3--universal-agents)
11. [Stage 4 — reconciler.py](#11-stage-4--reconcilerpy)
12. [Stage 5 — bayesian.py](#12-stage-5--bayesianpy)
13. [Stage 6 — generator.py + template.py](#13-stage-6--generatorpy--templatepy)
14. [master_agent.py — LangGraph Orchestrator](#14-master_agentpy--langgraph-orchestrator)
15. [How to Add a New Agent or Pool](#15-how-to-add-a-new-agent-or-pool)
16. [Report Reference — Page by Page](#16-report-reference--page-by-page)
17. [Installation & Setup](#17-installation--setup)
18. [Week 1 Completion Checklist](#18-week-1-completion-checklist)

---

## 1. Architecture Overview

MFAD processes a single image through **7 sequential stages**. Each stage is a LangGraph node. The graph is deterministic — the only LLM calls are CLIP (Stage 1, for routing) and Mistral-7B (Stage 6, for narrative prose).

```
Stage 0  ──  preprocessing.py
             Universal intake: SHA-256 · EXIF · ELA · image dims · PRNU baseline
             NO face detection here.
             Gate: aborts graph if image is unreadable / corrupt.
                │
Stage 1  ──  content_router.py
             ONE CLIP zero-shot call → routing flags
             face_present · has_text · scene_type · subject_count
             full_synthesis_likely · document_type
                │
Stage 2  ──  Conditional agent pools  (run in parallel via asyncio.gather)
             ┌──────────────┬─────────────┬──────────────┬────────────┬────────────┐
             │  Face pool   │ Scene pool  │ Forgery pool │  Doc pool  │ Synth pool │
             │ if face_     │ if scene_   │ always       │ if         │ always     │
             │ present      │ type set    │ active       │ has_text   │ active     │
             ├──────────────┼─────────────┼──────────────┼────────────┼────────────┤
             │ geometry.py  │struct_      │ ela_deep     │text_       │frequency   │
             │ biological   │consist      │ dct_artifact │integrity   │texture_    │
             │ vlm_face     │inpaint_     │ prnu_field   │ocr_consist │global      │
             │ reference_   │detect       │ resampling   │font_       │vlm_general │
             │ face         │physics_     │ noise_       │analysis    │reference_  │
             │ texture_face │check        │ consist      │layout_     │clip        │
             │              │copy_move    │              │check       │            │
             │              │splicing     │              │            │            │
             │ ACTIVE ✓     │ FUTURE      │ FUTURE       │ FUTURE     │ FUTURE     │
             └──────────────┴─────────────┴──────────────┴────────────┴────────────┘
                │
Stage 3  ──  Universal agents  (always run, no gate)
             metadata.py · steganography.py · provenance.py
                │
Stage 4  ──  reconciler.py
             Reads agent_applicable flags · resolves contradictions
             abstain = 0.5 (not 0.0) · flags conflicting signals
                │
Stage 5  ──  bayesian.py
             Weights only applicable agents · 95% CI · final score
                │
Stage 6  ──  generator.py
             ONLY LLM call for prose (Mistral-7B via Ollama)
             per-section forensic prose · cross-module synthesis
             verdict language · executive summary → PDF
```

---

## 2. Key Design Decisions

| Decision | Detail |
|---|---|
| **LLM used twice only** | CLIP at Stage 1 (routing, not prose). Mistral-7B at Stage 6 (prose only). Never for orchestration. |
| **asyncio.gather runs all pools** | Wall time = max(slowest agent), not sum. Each pool fires in parallel. |
| **agent_applicable flag** | abstain = 0.5, not 0.0. Fusion ignores non-applicable agents entirely — 0.0 would bias toward "real". |
| **Reconciler catches contradictions** | e.g. ELA clean but PRNU absent — flags before fusion, appears in report. |
| **No face detection in preprocessing** | preprocessing.py does universal work only. `geometry_agent` (Face pool, Stage 2) runs RetinaFace and owns `face_bbox`. |
| **CLIP ≠ LangGraph CLIP gate** | "CLIP gate" in older docs referred to a LangGraph Conditional Loop / Interrupt Point pattern. The model used for routing is OpenAI CLIP. These are two different things. |
| **face_bbox is face-pool-local** | No other pool needs or receives `face_bbox`. Each pool owns its own spatial detection. |
| **Week 1 scope** | Face pool active. All other pools are stubbed with commented-out `safe_run()` calls. |

---

## 3. Repository Structure

```
mfad/
├── contracts.py                   # shared output schemas — read first
├── master_agent.py                # LangGraph StateGraph + all 7 stage nodes
│
├── agents/
│   ├── preprocessing.py           # Stage 0 — Aditi + Sania
│   ├── content_router.py          # Stage 1 — CLIP routing
│   │
│   ├── face/                      # Stage 2 — Face pool (ACTIVE)
│   │   ├── geometry.py            # Veedhee — RetinaFace + 68-pt landmarks
│   │   ├── biological.py          # Diksha — rPPG, corneal, micro-texture
│   │   ├── vlm_face.py            # Aryan — BLIP-2 Grad-CAM (face crop)
│   │   ├── texture_face.py        # Manya — LBP, Gabor, EMD (face zones)
│   │   └── reference_face.py      # (shared) — FaceNet embedding
│   │
│   ├── scene/                     # Stage 2 — Scene pool (FUTURE)
│   │   ├── struct_consist.py
│   │   ├── inpaint_detect.py
│   │   ├── physics_check.py
│   │   ├── copy_move.py
│   │   └── splicing.py
│   │
│   ├── forgery/                   # Stage 2 — Forgery pool (FUTURE)
│   │   ├── ela_deep.py
│   │   ├── dct_artifact.py
│   │   ├── prnu_field.py
│   │   ├── resampling.py
│   │   └── noise_consist.py
│   │
│   ├── doc/                       # Stage 2 — Doc pool (FUTURE)
│   │   ├── text_integrity.py
│   │   ├── ocr_consist.py
│   │   ├── font_analysis.py
│   │   └── layout_check.py
│   │
│   ├── synth/                     # Stage 2 — Synth pool (FUTURE)
│   │   ├── frequency.py
│   │   ├── texture_global.py
│   │   ├── vlm_general.py
│   │   └── reference_clip.py
│   │
│   └── universal/                 # Stage 3 — always run
│       ├── metadata.py            # Sania — EXIF spoof, ELA chi2, PRNU
│       ├── steganography.py       # LSB payload detection
│       └── provenance.py          # C2PA, hash-db, chain-of-custody
│
├── fusion/
│   ├── reconciler.py              # Stage 4 — agent_applicable + contradiction flags
│   └── bayesian.py                # Stage 5 — log-odds fusion
│
├── report/
│   ├── template.py                # Saania '27 — ReportLab layout
│   └── generator.py               # Saania '27 — Mistral narrative + PDF assembly
│
├── test_images/
│   ├── sample_real.jpg
│   ├── sample_fake.jpg
│   └── reference/
│       ├── real/                  # 10–20 verified authentic faces
│       └── fake/                  # 10–20 known deepfakes (Celeb-DF)
│
└── outputs/                       # heatmaps, ELA maps, PDFs
```

---

## 4. Data Flow — End to End

```
INPUT IMAGE (jpg/png)
        │
        ▼
[Stage 0] preprocessing.py
        SHA-256 · MD5 · EXIF baseline · ELA map · image dims · PRNU baseline
        preprocess_ok=False  ──►  abort_node (writes error JSON, graph ends)
        preprocess_ok=True   ──►  continues
        │
        ▼
[Stage 1] content_router.py  (CLIP zero-shot — 1 VLM call)
        Returns: face_present · scene_type · has_text · subject_count
                 full_synthesis_likely · document_type
        │
        ▼
[Stage 2] pool_dispatch_node  (asyncio.gather — wall time = slowest pool)
        │
        ├── face_present=True  ──►  FACE POOL
        │   geometry.py  →  runs RetinaFace  →  sets face_bbox
        │                ↓ (face_bbox passed to remaining face agents)
        │   biological.py · vlm_face.py · texture_face.py · reference_face.py
        │   If RetinaFace finds no face → all face agents: agent_applicable=False
        │
        ├── scene_type set  ──►  SCENE POOL  [FUTURE]
        │   struct_consist · inpaint_detect · physics_check · copy_move · splicing
        │   (each agent does its own spatial detection — no shared bbox)
        │
        ├── always  ──►  FORGERY POOL  [FUTURE]
        │   ela_deep · dct_artifact · prnu_field · resampling · noise_consist
        │
        ├── has_text=True  ──►  DOC POOL  [FUTURE]
        │   text_integrity  →  runs OCR  →  sets text_bboxes
        │   ocr_consist · font_analysis · layout_check
        │
        └── always  ──►  SYNTH POOL  [FUTURE]
            frequency · texture_global · vlm_general · reference_clip
            (operates on FULL image, not face crop)
        │
        ▼
[Stage 3] universal_agents_node  (asyncio.gather — always runs)
        metadata.py · steganography.py · provenance.py
        │
        ▼
[Stage 4] reconciler.py
        Reads agent_applicable flags from every agent output
        non-applicable  ──►  reconciled_score = 0.5  (abstain, not 0.0)
        applicable       ──►  reconciled_score = agent anomaly_score
        flags contradictions (e.g. ELA clean but PRNU absent)
        │
        ▼
[Stage 5] bayesian.py
        log_odds = SUM [ weight_i × log(score_i / (1 − score_i)) ]
        final_score = sigmoid(log_odds)
        log(0.5/0.5) = 0  →  abstained agents contribute exactly zero
        outputs: final_score · confidence_interval · verdict · per_module_scores
        │
        ▼
[Stage 6] generator.py
        Mistral-7B via Ollama  →  per-section narrative prose
        assembles master JSON  →  calls report/template.py
        │
        ▼
FORENSIC PDF REPORT
```

---

## 5. MFADState — Shared State Object

`MFADState` is a `TypedDict(total=False)` — every key is optional. Nodes read only what they need and write only their own keys. No node calls another directly.

```python
class MFADState(TypedDict, total=False):

    # Input
    image_path:   str
    case_id:      str

    # Stage 0
    preprocessing:  dict    # full preprocessing_agent output
    preprocess_ok:  bool    # False → abort_node

    # Stage 1
    routing_flags:  dict    # full content_router_agent output
    face_present:   bool    # shortcut from routing_flags
    has_text:       bool    # shortcut
    scene_type:     Optional[str]  # shortcut

    # Stage 2 — Face pool
    face_bbox:       list   # set by geometry_agent, used by rest of face pool
    geometry:        dict
    biological:      dict
    vlm_face:        dict
    texture_face:    dict
    reference_face:  dict

    # Stage 2 — Scene pool (FUTURE — uncomment when implementing)
    # struct_consist:  dict
    # inpaint_detect:  dict
    # physics_check:   dict
    # copy_move:       dict
    # splicing:        dict

    # Stage 2 — Forgery pool (FUTURE)
    # ela_deep:        dict
    # dct_artifact:    dict
    # prnu_field:      dict
    # resampling:      dict
    # noise_consist:   dict

    # Stage 2 — Doc pool (FUTURE)
    # text_integrity:  dict
    # ocr_consist:     dict
    # font_analysis:   dict
    # layout_check:    dict

    # Stage 2 — Synth pool (FUTURE)
    # frequency:       dict
    # texture_global:  dict
    # vlm_general:     dict
    # reference_clip:  dict

    # Stage 3 — Universal
    metadata:      dict
    steganography: dict
    provenance:    dict

    # Stage 4
    reconciler_output: dict   # {reconciled_scores, contradiction_flags}

    # Stage 5
    fusion: dict              # {final_score, CI, verdict, per_module_scores}

    # Stage 6
    report_path:    str
    master_output:  dict

    # Reflection loop counter (inactive — see build_graph comments)
    reflection_passes: int

    # Error tracking
    errors:      list[str]
    fatal_error: Optional[str]
```

---

## 6. contracts.py — Output Schemas

**Owner:** Everyone reads. No one changes keys without team agreement.  
**Rule:** Every agent must return `anomaly_score: float` (0–1) and `agent_applicable: bool`. Fusion will silently use `0.5` (abstain) if `anomaly_score` is missing.

### Stage 0

```python
PREPROCESSING_OUTPUT = {
    "image_path":    str,
    "hash_sha256":   str,
    "hash_md5":      str,
    "ela_score":     float,
    "image_dims":    list,   # [width, height]
    "prnu_baseline": float,
    "exif_raw":      dict,
    "preprocess_ok": bool,
    "anomaly_score": float,
    "agent_applicable": bool,
}
```

### Stage 1

```python
CONTENT_ROUTER_OUTPUT = {
    "face_present":          bool,
    "scene_type":            Optional[str],  # "outdoor"|"indoor"|"studio"|None
    "has_text":              bool,
    "full_synthesis_likely": bool,
    "subject_count":         int,
    "document_type":         str,   # "photo"|"screenshot"|"document"|"artwork"
    "agent_applicable":      bool,
    "anomaly_score":         float, # always 0.0 — router is neutral
}
```

### Stage 2 — Face pool

```python
GEOMETRY_OUTPUT = {
    "face_bbox":           list,   # [x1, y1, x2, y2] — from RetinaFace
    "symmetry_index":      float,
    "jaw_curvature_deg":   float,
    "ear_alignment_px":    float,
    "philtrum_length":     float,
    "landmark_confidence": float,
    "anomaly_score":       float,
    "agent_applicable":    bool,   # False if RetinaFace finds no face
}

BIOLOGICAL_OUTPUT = {
    "rppg_snr":              float,
    "corneal_deviation_deg": float,
    "micro_texture_var":     float,
    "highlight_positions":   dict,  # {"left": [x,y], "right": [x,y]}
    "anomaly_score":         float,
    "agent_applicable":      bool,
}

VLM_FACE_OUTPUT = {
    "heatmap_path":            str,
    "vlm_caption":             str,
    "saliency_score":          float,
    "high_activation_regions": list,
    "anomaly_score":           float,
    "agent_applicable":        bool,
}

TEXTURE_FACE_OUTPUT = {
    "jaw_emd":          float,
    "neck_emd":         float,
    "cheek_emd":        float,
    "lbp_uniformity":   float,
    "seam_detected":    bool,
    "zone_scores":      dict,
    "anomaly_score":    float,
    "agent_applicable": bool,
}

REFERENCE_FACE_OUTPUT = {
    "cosine_dist_authentic": float,
    "cosine_dist_fake":      float,
    "verdict":               str,
    "embedding_norm":        float,
    "anomaly_score":         float,
    "agent_applicable":      bool,
}
```

### Stage 2 — Future pool schemas (add to contracts.py when implementing)

```python
# Scene pool
SCENE_AGENT_OUTPUT = {
    # Each agent in the scene pool follows this pattern:
    # ... agent-specific keys ...
    "anomaly_score":    float,
    "agent_applicable": bool,
}

# Forgery pool — example
ELA_DEEP_OUTPUT = {
    "ela_deep_map_path": str,
    "ela_deep_chi2":     float,
    "anomaly_score":     float,
    "agent_applicable":  bool,
}

# Doc pool — example
TEXT_INTEGRITY_OUTPUT = {
    "text_bboxes":         list,  # OCR bounding boxes
    "ocr_confidence":      float,
    "text_anomaly_score":  float,
    "anomaly_score":       float,
    "agent_applicable":    bool,
}

# Synth pool — example
FREQUENCY_OUTPUT = {
    "fft_mid_anomaly_db":  float,
    "fft_high_anomaly_db": float,
    "gan_probability":     float,
    "freq_spectrum_path":  str,
    "anomaly_score":       float,
    "agent_applicable":    bool,
}
```

### Stage 3 — Universal

```python
METADATA_OUTPUT = {
    "exif_camera_present": bool,
    "software_tag":        str,
    "ela_chi2":            float,
    "ela_map_path":        str,
    "thumbnail_mismatch":  bool,
    "prnu_absent":         bool,
    "anomaly_score":       float,
    "agent_applicable":    bool,
}

STEGANOGRAPHY_OUTPUT = {
    "lsb_payload_detected": bool,
    "lsb_capacity_bits":    int,
    "anomaly_score":        float,
    "agent_applicable":     bool,
}

PROVENANCE_OUTPUT = {
    "c2pa_manifest_present": bool,
    "hash_known_fake":       bool,
    "origin_url":            Optional[str],
    "anomaly_score":         float,
    "agent_applicable":      bool,
}
```

### Stage 4–5

```python
RECONCILER_OUTPUT = {
    "reconciled_scores":   dict,   # {module_name: float}  — abstain=0.5
    "contradiction_flags": list,   # list of contradiction description strings
}

FUSION_OUTPUT = {
    "final_score":         float,  # 0–1, probability of deepfake
    "confidence_interval": list,   # [low, high] at 95%
    "verdict":             str,    # "DEEPFAKE" or "LIKELY REAL"
    "per_module_scores":   dict,   # {module_name: reconciled_score}
}
```

---

## 7. Stage 0 — preprocessing.py

**Owner:** Aditi  
**Priority:** HIGHEST — the gate node. Everything is blocked until this completes.  
**Libraries:** `Pillow`, `piexif`, `hashlib`, `numpy`

### What it does

1. Validates the image file is readable and not corrupt
2. Computes SHA-256 and MD5 for chain-of-custody
3. Reads raw EXIF metadata (no parsing yet — that's `metadata.py` in Stage 3)
4. Runs basic ELA (Error Level Analysis) — preliminary tampering indicator
5. Records image dimensions
6. Computes PRNU baseline (sensor noise fingerprint template)
7. Sets `preprocess_ok=True/False` — this is the gate flag

> **No face detection here.** `geometry.py` in the Face pool runs RetinaFace and owns `face_bbox`.

### Output

```python
{
    "image_path":    "/abs/path/to/image.jpg",
    "hash_sha256":   "a3f9b2...",
    "hash_md5":      "c4e8a1...",
    "ela_score":     0.34,
    "image_dims":    [1920, 1080],
    "prnu_baseline": 0.021,
    "exif_raw":      {...},
    "preprocess_ok": True,
    "anomaly_score": 0.34,
    "agent_applicable": True,
}
```

### Report connection
`hash_sha256` and `hash_md5` populate the **Chain of Custody** table on page 2. `image_dims` appears in the case summary header.

---

## 8. Stage 1 — content_router.py

**Owner:** TBD (can be implemented alongside Stage 2)  
**Model:** CLIP (`openai/clip-vit-base-patch32`) — zero-shot image classification  
**Libraries:** `transformers`, `torch`, `Pillow`

### What it does

One forward pass of CLIP with multiple text prompts to produce all routing flags simultaneously. This is the **only VLM call in Stage 1** — it is not Mistral, it is not BLIP-2.

```python
text_prompts = [
    "a photo of a human face",           # → face_present
    "a photo of a scene without people", # → scene_type candidate
    "a document or screenshot with text",# → has_text
    "a computer-generated or AI image",  # → full_synthesis_likely
    "a photo of a group of people",      # → subject_count > 1
]
# softmax over prompts → routing flags
```

### Routing logic

| Flag | Value | Activates |
|---|---|---|
| `face_present` | True | Face pool (Stage 2) |
| `scene_type` | `"outdoor"` / `"indoor"` / `"studio"` / `None` | Scene pool (Stage 2) |
| `has_text` | True | Doc pool (Stage 2) |
| `full_synthesis_likely` | True | Boosts Synth pool weight in fusion |
| `subject_count` | > 1 | Future multi-face fan-out in Face pool |

### Current status
Stub — hardcoded `face_present=True`, all others `False/None`. Replace stub body with real CLIP inference when ready. The `router_node` in `master_agent.py` does not need to change.

### Output

```python
{
    "face_present":          True,
    "scene_type":            None,
    "has_text":              False,
    "full_synthesis_likely": False,
    "subject_count":         1,
    "document_type":         "photo",
    "agent_applicable":      True,
    "anomaly_score":         0.0,
}
```

---

## 9. Stage 2 — Agent Pools

### Pool architecture rules

- All active pools run in **parallel** via `asyncio.gather` inside `pool_dispatch_node`
- Each pool is **independent** — no pool reads another pool's output
- Each pool owns its own **spatial detection** (`face_bbox` belongs to Face pool only)
- Failed agents return `anomaly_score=0.5, agent_applicable=False` — they abstain, not fail
- Wall time = `max(slowest agent)`, not sum

---

### 9.1 Face Pool (ACTIVE)

**Activated by:** `face_present=True` from content_router  
**Face bbox lifecycle:** `geometry_agent` runs RetinaFace → gets `face_bbox` → `pool_dispatch_node` passes it to remaining face agents

#### agents/face/geometry.py

**Owner:** Veedhee  
**Libraries:** `retinaface-pytorch` (or `mediapipe`), `dlib`, `numpy`, `scipy`

**Why RetinaFace and not CLIP or MediaPipe for face detection:**
- RetinaFace returns a tight `[x1, y1, x2, y2]` bounding box with landmark confidence — forensic work requires exact pixel coordinates
- CLIP answers "is there a face?" semantically but gives no bounding box — useless for cropping
- MediaPipe is faster but less accurate for small/occluded faces in forensic contexts

**What it does:**
1. Runs RetinaFace on the full image → `face_bbox`
2. Crops face region using `face_bbox`
3. Runs dlib 68-point shape predictor on the face crop
4. Computes anthropometric ratios: symmetry index, jaw curvature, ear alignment, philtrum length
5. Compares against known authentic norms (±2 standard deviations threshold)
6. Sets `agent_applicable=False` if RetinaFace confidence < 0.5 or no face found

> **Note:** Download `shape_predictor_68_face_landmarks.dat` from dlib's official release (~100MB). Add to `.gitignore`. Share via team shared folder.

```python
# Output
{
    "face_bbox":           [120, 80, 420, 460],
    "symmetry_index":      0.74,
    "jaw_curvature_deg":   11.2,
    "ear_alignment_px":    8.7,
    "philtrum_length":     0.21,
    "landmark_confidence": 0.91,
    "anomaly_score":       0.884,
    "agent_applicable":    True,
}
```

**Report connection:** Section 4.1 — Geometric & Structural Analysis. Symmetry/jaw table with green/red highlighting. Landmark overlay figure (68 dots on face).

---

#### agents/face/biological.py

**Owner:** Diksha  
**Libraries:** `opencv-python`, `mediapipe`, `numpy`

**What it does:**
1. **rPPG SNR:** Green channel frequency-domain SNR from forehead/cheeks. Real faces show 0.75–2.5 Hz component. SNR < 3.0 dB is suspicious.
2. **Corneal highlights:** Specular highlight angle must match between both eyes (same light source). Discrepancy > 15° is anomalous.
3. **Micro-texture variance:** Local variance in perioral region. GAN skin is over-smoothed — variance < 0.02 is suspicious.

```python
# Output
{
    "rppg_snr":              2.1,
    "corneal_deviation_deg": 22.4,
    "micro_texture_var":     0.012,
    "highlight_positions":   {"left": [210, 180], "right": [310, 182]},
    "anomaly_score":         0.826,
    "agent_applicable":      True,
}
```

**Report connection:** Section 4.5 — Biological Plausibility. Corneal diagram, rPPG spectrum plot, micro-texture finding.

---

#### agents/face/vlm_face.py

**Owner:** Aryan  
**Libraries:** `transformers`, `torch`, `pytorch-grad-cam`, `opencv-python`, `Pillow`  
**Model:** `Salesforce/blip2-opt-2.7b` (~5 GB — auto-downloads on first run)

**What it does:**
- **Part A — Grad-CAM:** Hooks into BLIP-2's last attention layer, generates gradient-weighted activation map over the face crop, overlays as colour heatmap
- **Part B — Forensic caption:** 3 structured prompts (boundary artifacts, skin texture, lighting consistency) → concatenated forensic finding string
- **Part C — Saliency score:** Mean activation in central face region → `anomaly_score`

> Use `device_map="auto"` for automatic CPU offloading. First load takes 2–3 min — do not kill the process.

> **Distinction from vlm_general (Synth pool):** `vlm_face` analyses only the face crop. `vlm_general` will analyse the full image for synthesis patterns. Two different agents, two different report sections.

```python
# Output
{
    "heatmap_path":            "outputs/heatmap_face.png",
    "vlm_caption":             "Unnatural texture smoothing at jaw boundary...",
    "saliency_score":          0.91,
    "high_activation_regions": ["jaw boundary", "eyes", "nose bridge"],
    "anomaly_score":           0.931,
    "agent_applicable":        True,
}
```

**Report connection:** Section 4.4 — VLM Explainability. Full-width Grad-CAM figure. `vlm_caption` becomes the "AI Forensic Findings" paragraph.

---

#### agents/face/texture_face.py

**Owner:** Manya  
**Libraries:** `scikit-image`, `scipy`, `numpy`, `opencv-python`  
**Note:** No model download — entirely algorithmic.

**What it does:**
1. Crops face into 5 zones (forehead, cheeks, nose, jaw) using `face_bbox`
2. Computes LBP histogram per zone
3. Applies Gabor filter bank (4 orientations × 3 frequencies)
4. Earth Mover's Distance between adjacent zones — detects blending seams
5. `seam_detected=True` if any EMD > 0.45

> **Distinction from texture_global (Synth pool):** `texture_face` analyses face-zone EMD seams. `texture_global` will analyse full-image frequency texture for GAN synthesis patterns globally.

```python
# Output
{
    "jaw_emd":          0.61,
    "neck_emd":         0.48,
    "cheek_emd":        0.22,
    "lbp_uniformity":   0.31,
    "seam_detected":    True,
    "zone_scores":      {"forehead": 0.2, "cheek_L": 0.3, "jaw": 0.8},
    "anomaly_score":    0.895,
    "agent_applicable": True,
}
```

**Report connection:** Section 4.3 — Texture Analysis. Zone heatmap figure (face outline, colour-coded). `seam_detected=True` triggers "Blending artefact detected at jaw-neck boundary" finding.

---

#### agents/face/reference_face.py

**Owner:** Shared  
**Libraries:** `deepface`, `numpy`, `os`

**What it does:**
1. Computes 512-dim FaceNet embedding for the face crop
2. Cosine distance to centroid of real cluster (`test_images/reference/real/`) and fake cluster (`test_images/reference/fake/`)
3. Verdict = closer cluster

> **Distinction from reference_clip (Synth pool):** `reference_face` uses FaceNet embeddings on face crops for identity-space comparison. `reference_clip` will use CLIP embeddings on full images for synthesis-style comparison.

```python
# Output
{
    "cosine_dist_authentic": 0.71,
    "cosine_dist_fake":      0.18,
    "verdict":               "CLOSER_TO_FAKE",
    "embedding_norm":        0.994,
    "anomaly_score":         0.910,
    "agent_applicable":      True,
}
```

**Report connection:** Reference embedding scatter plot (Figure 5) — input image position relative to real/fake clusters.

---

### 9.2 Scene Pool (FUTURE)

**Activated by:** `scene_type is not None` from content_router  
**Owner:** TBD  
**No shared bbox** — each agent does its own spatial parsing independently

| Agent | Purpose |
|---|---|
| `struct_consist.py` | Structural consistency — perspective, horizon line, vanishing points |
| `inpaint_detect.py` | Inpainting artefact detection — repeated patch patterns |
| `physics_check.py` | Physical plausibility — shadow direction, reflection consistency |
| `copy_move.py` | Copy-move forgery — duplicated regions via keypoint matching |
| `splicing.py` | Image splicing — noise level inconsistencies between regions |

**To implement:** Follow the 5-step checklist in [Section 15](#15-how-to-add-a-new-agent-or-pool). Uncomment the scene pool `safe_run()` block in `pool_dispatch_node`.

---

### 9.3 Forgery Pool (FUTURE)

**Activated by:** Always — no routing gate  
**Owner:** TBD

| Agent | Purpose |
|---|---|
| `ela_deep.py` | Deep ELA — multi-quality-level error level analysis |
| `dct_artifact.py` | DCT coefficient anomalies — JPEG blocking artefacts |
| `prnu_field.py` | PRNU field analysis — per-pixel sensor noise inconsistency |
| `resampling.py` | Resampling artefacts — interpolation traces |
| `noise_consist.py` | Noise consistency — Gaussian noise level map |

---

### 9.4 Doc Pool (FUTURE)

**Activated by:** `has_text=True` from content_router  
**Owner:** TBD  
**Note:** `text_integrity.py` runs OCR first and owns `text_bboxes` — analogous to how `geometry.py` owns `face_bbox` in the Face pool.

| Agent | Purpose |
|---|---|
| `text_integrity.py` | OCR + text coherence — runs OCR, detects text bboxes |
| `ocr_consist.py` | OCR consistency — font/layout vs text content |
| `font_analysis.py` | Font rendering anomalies — AI-generated text signatures |
| `layout_check.py` | Document layout plausibility |

---

### 9.5 Synth Pool (FUTURE)

**Activated by:** Always — no routing gate  
**Owner:** TBD  
**Operates on the full image** — not face crops

| Agent | Purpose |
|---|---|
| `frequency.py` | FFT mid/high band anomaly + EfficientNet-B4 GAN probability |
| `texture_global.py` | Global LBP/Gabor texture analysis — full image synthesis patterns |
| `vlm_general.py` | BLIP-2 Grad-CAM on full image — general synthesis findings |
| `reference_clip.py` | CLIP embedding cosine distance to real/synth clusters |

---

## 10. Stage 3 — Universal Agents

**Activated by:** Always — no routing gate, no agent_applicable check  
**Owner:** Sania (metadata.py), TBD (steganography, provenance)  
**All three run in parallel via `asyncio.gather`**

### agents/universal/metadata.py

**Libraries:** `piexif`, `Pillow`, `numpy`

Five checks:
1. **EXIF parsing:** Camera make/model, GPS, software tag. Missing = suspicious.
2. **Software tag check:** Photoshop, DALL-E, Stable Diffusion → flag immediately.
3. **ELA chi-squared:** Re-save at 95% JPEG quality, compute absolute difference. chi2 > 500 suspicious.
4. **Thumbnail mismatch:** EXIF thumbnail vs main image — mismatch = post-processing.
5. **PRNU absence:** No sensor noise signature = suspicious for synthetic images.

```python
# Output
{
    "exif_camera_present": False,
    "software_tag":        "Adobe Photoshop 24.0",
    "ela_chi2":            847.3,
    "ela_map_path":        "outputs/ela_map.png",
    "thumbnail_mismatch":  True,
    "prnu_absent":         True,
    "anomaly_score":       0.973,
    "agent_applicable":    True,
}
```

### agents/universal/steganography.py

**Libraries:** `numpy`, `Pillow`  
Detects LSB (Least Significant Bit) payload in image channels. Hidden data = suspicious provenance.

### agents/universal/provenance.py

**Libraries:** `requests` (for hash-db lookup), `c2pa-python`  
Checks C2PA content authenticity manifest. Computes reverse-lookup against known-fake hash databases. Records `origin_url` if found.

---

## 11. Stage 4 — reconciler.py

**Owner:** Dev  
**Libraries:** `numpy`

### Why the reconciler exists

Without it, a metadata agent that returns `anomaly_score=0.0` because it has `agent_applicable=False` (e.g., a document image with no face) would bias Bayesian fusion heavily toward "LIKELY REAL". The reconciler replaces `0.0` with `0.5` (abstain) so non-applicable agents contribute **zero log-odds** to fusion.

### abstain = 0.5 rule

```
log(0.5 / (1 - 0.5)) = log(1) = 0
```

A score of 0.5 contributes exactly 0 log-odds to the fusion sum. The agent is invisible to the final score.

### Contradiction detection

The reconciler flags cross-agent contradictions before fusion. These appear in the report as explicit findings.

| Contradiction | Meaning |
|---|---|
| ELA chi2 low but PRNU absent | Image was generated cleanly — targeted synthesis |
| geometry symmetry OK but vlm_face high activation | Face geometry is real but subtle synthesis in texture layer |
| provenance C2PA present but metadata EXIF stripped | Content manifest exists but camera metadata was removed post-capture |

```python
# Output
{
    "reconciled_scores": {
        "geometry":      0.884,  # applicable
        "biological":    0.826,  # applicable
        "vlm_face":      0.931,  # applicable
        "texture_face":  0.895,  # applicable
        "reference_face":0.910,  # applicable
        "metadata":      0.973,  # applicable
        "steganography": 0.100,  # applicable
        "provenance":    0.200,  # applicable
    },
    "contradiction_flags": [
        "ELA clean but PRNU absent — possible targeted synthesis"
    ]
}
```

---

## 12. Stage 5 — bayesian.py

**Owner:** Dev  
**Libraries:** `numpy`

### Formula

```
log_odds_total = Σ [ weight_i × log(score_i / (1 − score_i)) ]
final_score    = sigmoid(log_odds_total) = 1 / (1 + exp(−log_odds_total))
```

Scores are clamped to `[1e-6, 1 − 1e-6]` to avoid `log(0)`.

### Module weights

| Module | Weight | Basis |
|---|---|---|
| geometry | 0.884 | Landmark-based deepfake detection AUC |
| biological | 0.826 | rPPG + corneal combined AUC estimate |
| vlm_face | 0.931 | BLIP-2 saliency correlation with ground truth |
| texture_face | 0.895 | LBP+EMD cross-dataset generalisation |
| reference_face | 0.910 | FaceNet cosine similarity on Celeb-DF |
| metadata | 0.973 | PRNU + ELA combined specificity |
| steganography | 0.700 | Estimated — update after validation |
| provenance | 0.750 | Estimated — update after validation |
| **Future pools** | **TBD** | **Set after measuring AUC on validation set** |

### Output

```python
{
    "final_score":         0.957,
    "confidence_interval": [0.91, 0.98],
    "verdict":             "DEEPFAKE",   # threshold ≥ 0.70
    "per_module_scores":   { ... }
}
```

---

## 13. Stage 6 — generator.py + template.py

**Owner:** Saania '27  
**Libraries:** `reportlab`, `ollama`, `json`, `os`

### LLM narrative pattern (Mistral-7B)

```python
# One prompt per report section — per-section forensic prose
prompt = f"""
You are writing a forensic report section for a digital forensics expert.
Module: VLM Face Explainability Analysis
Findings: {vlm_face_output['vlm_caption']}
Saliency score: {vlm_face_output['saliency_score']}
High activation regions: {vlm_face_output['high_activation_regions']}
Contradictions flagged: {contradiction_flags}

Write a 3-sentence technical forensic paragraph. Use precise language.
Do not speculate. State only what the data shows.
"""
```

### Report page map

| Page | Content | Data source |
|---|---|---|
| 1 | Cover — case ID, DeepFake score (%), verdict badge, SHA-256 | `fusion.final_score`, `preprocessing.hash_sha256` |
| 2 | Chain of custody table, ISO/IEC 27037 statement | All agent timestamps + hashes |
| 3 | Executive summary + Module Confidence Matrix | `fusion.per_module_scores`, `reconciler.contradiction_flags` |
| 4 | Section 4.1 — Geometric Analysis + landmark figure | `geometry` output |
| 5 | Section 4.2 — Texture Analysis + zone heatmap | `texture_face` output |
| 6 | Section 4.3 — VLM Explainability + Grad-CAM figure | `vlm_face` output |
| 7 | Section 4.4 — Biological Plausibility | `biological` output |
| 8 | Section 4.5 — Reference Embedding scatter | `reference_face` output |
| 9 | Section 4.6 — Metadata & Provenance + ELA map | `metadata`, `provenance` output |
| 10 | Section 4.7 — Steganography | `steganography` output |
| 11+ | *Future pool sections added here* | Scene, Forgery, Doc, Synth outputs |
| Last | Legal Certification — ISO/IEC 27037, Daubert, SWGDE | Static template |

### Colour palette

```python
NAVY   = colors.HexColor('#1B2A4A')
RED    = colors.HexColor('#C0392B')
AMBER  = colors.HexColor('#E67E22')
GREEN  = colors.HexColor('#27AE60')
LIGHT  = colors.HexColor('#F4F6F9')
```

---

## 14. master_agent.py — LangGraph Orchestrator

**Owner:** Dev  
**Framework:** LangGraph (StateGraph) + LangChain `@tool` + Ollama (Mistral-7B)

### Node map

| Node | Stage | Key work | On failure |
|---|---|---|---|
| `preprocess_node` | 0 | SHA-256, ELA, dims, PRNU | Sets `preprocess_ok=False` → `abort_node` |
| `router_node` | 1 | CLIP routing flags | Fails safe: `face_present=True` |
| `pool_dispatch_node` | 2 | asyncio.gather all active pools | Per-agent: abstain stub |
| `universal_agents_node` | 3 | metadata, stego, provenance | Per-agent: abstain stub |
| `reconciler_node` | 4 | agent_applicable → abstain, contradictions | Never fails |
| `fusion_node` | 5 | Bayesian log-odds | Never fails |
| `report_node` | 6 | Mistral narrative + JSON/PDF | LLM fail → template string |
| `abort_node` | — | Writes error JSON | Never fails |

### Compiled graph (current active path)

```
START
  → preprocess_node
  → [gate] router_node         (abort_node if gate fails)
  → pool_dispatch_node
  → universal_agents_node
  → reconciler_node
  → fusion_node
  → report_node
  → END
```

### Reflection loop (currently inactive)

```python
# To activate: in build_graph(), replace:
graph.add_edge("fusion_node", "report_node")

# With:
graph.add_conditional_edges(
    "fusion_node",
    should_reflect,   # returns "reflect" or "report"
    {
        "reflect": "pool_dispatch_node",  # re-run all pools
        "report":  "report_node",
    },
)

# should_reflect() fires when:
#   0.45 <= final_score <= 0.65  AND  reflection_passes < 2
```

### Public API

```python
from master_agent import analyse_image

result = analyse_image(
    image_path="test_images/sample_fake.jpg",
    analyst_name="Aryan Sharma"
)

result["fusion"]["verdict"]          # "DEEPFAKE"
result["fusion"]["final_score"]      # 0.957
result["fusion"]["confidence_interval"]  # [0.91, 0.98]
result["reconciler"]["contradiction_flags"]  # [...]
result["report_path"]                # "outputs/DFA-2026-TC-A3F9.pdf"
```

### Master JSON shape

```python
{
    "case_id":       "DFA-2026-TC-A3F9B200",
    "image_path":    "...",
    "timestamp":     "2026-03-17T22:14:00",
    "routing_flags": { "face_present": True, "scene_type": None, ... },
    "agent_outputs": {
        "preprocessing":  { ... },
        "geometry":       { ... },
        "biological":     { ... },
        "vlm_face":       { ... },
        "texture_face":   { ... },
        "reference_face": { ... },
        "metadata":       { ... },
        "steganography":  { ... },
        "provenance":     { ... },
        # Future pools added here
    },
    "reconciler": {
        "reconciled_scores":   { ... },
        "contradiction_flags": [ ... ],
    },
    "fusion": {
        "final_score":         0.957,
        "confidence_interval": [0.91, 0.98],
        "verdict":             "DEEPFAKE",
        "per_module_scores":   { ... }
    },
    "executive_summary": "...",
    "errors": []
}
```

---

## 15. How to Add a New Agent or Pool

**5 files to change. Do not skip any.**

### Step-by-step

**Step 1 — `contracts.py`**  
Add the new output schema dict with `anomaly_score` and `agent_applicable` keys.

**Step 2 — `MFADState` in `master_agent.py`**  
Add the new optional state key:
```python
new_module: dict   # output from new_module_agent
```

**Step 3 — `pool_dispatch_node` in `master_agent.py`**  
Add one `safe_run()` call inside the appropriate pool block (or new pool block if it's a new pool). If the pool has a routing gate, wrap it in the appropriate `if` check:
```python
# Inside run_pools() → appropriate pool block:
safe_run("new_module", lambda: new_module_agent.invoke({
    "image_path": image_path,
    "face_bbox":  face_bbox,  # only if face pool
}))
```

**Step 4 — `reconciler_node` in `master_agent.py`**  
Add the new module to `all_agent_outputs`:
```python
"new_module": state.get("new_module", {}),
```

**Step 5 — `fusion/bayesian.py`**  
Add to `MODULE_WEIGHTS`:
```python
"new_module": 0.85,   # set based on AUC measured on your validation set
```

### For a new pool (not just a new agent)

Additionally:
- Add a pool block comment in `pool_dispatch_node` with the routing gate condition
- Add pool-level state keys in `MFADState` (all commented out for future pools)
- Add a new report section in `report/generator.py`
- Add pool agents to `pool_dispatch_node` → `all_outputs` unpacking at the bottom

---

## 16. Report Reference — Page by Page

| Section | Page | Data source | File |
|---|---|---|---|
| Cover — DeepFake score | 1 | `fusion.final_score` | `generator.py` |
| SHA-256 hash | 1 | `preprocessing.hash_sha256` | `generator.py` |
| Chain of custody | 2 | All agent timestamps + hashes | `generator.py` |
| Executive summary | 3 | LLM + `fusion.per_module_scores` | `generator.py` |
| Module Confidence Matrix | 3 | `fusion.per_module_scores` | `generator.py` |
| Contradiction flags | 3 | `reconciler.contradiction_flags` | `generator.py` |
| 4.1 Geometric Analysis | 4 | `geometry` | `generator.py` |
| Landmark overlay figure | 4 | Saved by `geometry.py` | `template.py` |
| 4.2 Texture Analysis | 5 | `texture_face` | `generator.py` |
| Zone heatmap figure | 5 | Computed in `texture_face.py` | `template.py` |
| 4.3 VLM Explainability | 6 | `vlm_face` | `generator.py` |
| Grad-CAM figure | 6 | `vlm_face.heatmap_path` | `template.py` |
| 4.4 Biological Plausibility | 7 | `biological` | `generator.py` |
| 4.5 Reference Embedding | 8 | `reference_face` | `generator.py` |
| 4.6 Metadata & Provenance | 9 | `metadata`, `provenance` | `generator.py` |
| ELA map figure | 9 | `metadata.ela_map_path` | `template.py` |
| 4.7 Steganography | 10 | `steganography` | `generator.py` |
| Future pool sections | 11+ | Future pool outputs | `generator.py` |
| Legal Certification | Last | Static | `template.py` |

---

## 17. Installation & Setup

```bash
# 1. Clone and create environment
git clone https://github.com/your-org/mfad.git
cd mfad
python -m venv .venv && source .venv/bin/activate

# 2. Install Python dependencies
pip install langgraph langchain langchain-ollama
pip install torch torchvision
pip install transformers   # CLIP + BLIP-2
pip install retinaface-pytorch
pip install mediapipe dlib
pip install scikit-image scipy opencv-python Pillow
pip install deepface piexif reportlab
pip install pytorch-grad-cam

# 3. Install and start Ollama (Mistral-7B — Stage 6 only)
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama serve   # keep running in a separate terminal

# 4. Download dlib shape predictor (geometry.py)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
# Add to .gitignore — do NOT commit

# 5. Create reference image folders (reference_face.py)
mkdir -p test_images/reference/real   # add 10–20 verified authentic faces
mkdir -p test_images/reference/fake   # add 10–20 known deepfakes from Celeb-DF

# 6. BLIP-2 model (vlm_face.py) — auto-downloads on first run (~5 GB)
# Ensure 8 GB+ free disk space before running
```

---

## 18. Week 1 Completion Checklist

### Core pipeline

- [ ] `preprocessing.py` runs on both test images, returns correct schema (no `face_bbox`)
- [ ] `content_router.py` stub returns `face_present=True` correctly
- [ ] `geometry.py` runs RetinaFace, returns `face_bbox`, runs dlib landmarks
- [ ] `biological.py`, `vlm_face.py`, `texture_face.py`, `reference_face.py` all run using `face_bbox` from geometry
- [ ] `metadata.py`, `steganography.py`, `provenance.py` all run (universal stage)
- [ ] All agents return `anomaly_score` and `agent_applicable` — verified against `contracts.py`
- [ ] `reconciler.py` produces `reconciled_scores` (abstain=0.5 for non-applicable) and `contradiction_flags`
- [ ] `bayesian_fusion()` produces `final_score` between 0 and 1
- [ ] `master_agent.py` orchestrates all 7 stages end-to-end via LangGraph
- [ ] PDF report generates without errors and matches `DFA2025TC00471.pdf` structure

### Validation

- [ ] `sample_fake.jpg` → `final_score` > 0.70, `verdict = DEEPFAKE`
- [ ] `sample_real.jpg` → `final_score` < 0.40, `verdict = LIKELY REAL`

### Future readiness (verify hooks exist, do not implement yet)

- [ ] `MFADState` has commented-out state keys for all 4 future pools
- [ ] `pool_dispatch_node` has clearly marked comment blocks for Scene, Forgery, Doc, Synth pools
- [ ] `reconciler_node`'s `all_agent_outputs` has commented-out entries for future pool agents
- [ ] `MODULE_WEIGHTS` in `bayesian.py` has commented-out entries for future pool weights
- [ ] `contracts.py` has future pool schemas as comments

---

*Documentation v3.0 — MFAD Project — Reflects 7-stage LangGraph architecture with content router*