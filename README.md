# MFAD — Multi-Factor AI Deepfake Detection

A multi-agent forensic pipeline that inspects a single still image from seven independent angles (preprocessing, geometry, frequency, texture, VLM, biology, metadata), fuses their verdicts with Bayesian log-odds, and emits a court-grade PDF report narrated by a local LLM.

The project is orchestrated as a **LangGraph `StateGraph`**, where each forensic agent is a **LangChain `@tool`**. Agents are isolated, stateless, and run concurrently via `asyncio.gather`. The only contract they share: every agent must return a dict containing `anomaly_score ∈ [0.0, 1.0]`.

---

## 1. High-level flow

```
                          image_path
                              │
                              ▼
                   ┌────────────────────┐
                   │  preprocess_node   │  SHA-256, face bbox, ELA, crops
                   └────────┬───────────┘
                            │ face_detected?
                 ┌──────────┴──────────┐
              no │                     │ yes
                 ▼                     ▼
           abort_node       ┌──────────────────────────┐
                            │  parallel_analysis_node  │
                            │  (asyncio.gather)        │
                            │                          │
                            │  geometry_tool           │
                            │  frequency_tool          │
                            │  texture_tool            │
                            │  vlm_tool                │
                            │  biological_tool         │
                            │  metadata_tool           │
                            └────────────┬─────────────┘
                                         │ agent_outputs[name] = {...}
                                         ▼
                               ┌───────────────────┐
                               │   fusion_node     │  Bayesian log-odds
                               └─────────┬─────────┘
                                         │ final_score, decision, CI
                                         ▼
                               ┌───────────────────┐
                               │   report_node     │  Mistral narrative
                               │                   │  + ReportLab PDF
                               └─────────┬─────────┘
                                         ▼
                                        END
```

Compiled graph lives in [`master_agent.py`](master_agent.py) (`build_graph`, `analyse_image`). A reflection loop (re-run on ambiguous scores 0.45–0.65) is scaffolded but wired as a direct edge today.

---

## 2. Entry point

CLI:

```bash
python master_agent.py path/to/image.jpg --analyst "Dev Nagaich"
```

Programmatic:

```python
from master_agent import analyse_image
result = analyse_image("test_images/IMG_6930.JPG")
result["fusion"]["decision"]      # "DEEPFAKE" | "UNCERTAIN" | "AUTHENTIC"
result["fusion"]["final_score"]   # 0.957
result["report_path"]             # outputs/DFA-2026-TC-XXXX.pdf (or .json fallback)
```

A fresh `case_id` like `DFA-2026-TC-544A1D84` is minted per run (UUID4 prefix).

---

## 3. Shared state

All nodes read/write one `MFADState` TypedDict ([`master_agent.py:398`](master_agent.py#L398)):

| Field | Set by | Purpose |
|---|---|---|
| `image_path`, `case_id`, `analyst_name` | CLI / `analyse_image` | Input + identity |
| `preprocessing`, `face_bbox`, `preprocess_ok` | `preprocess_node` | Gate + shared crop |
| `agent_outputs` | `parallel_analysis_node` | `{agent_name: dict}` |
| `fusion` | `fusion_node` | Bayesian result |
| `reflection_passes` | `parallel_analysis_node` | Loop counter |
| `report_path`, `master_output` | `report_node` / `abort_node` | Final artefact |
| `errors`, `fatal_error` | any node | Non-fatal / fatal |

---

## 4. The agents (`agents/`)

Each agent is a plain Python module that exports a `run(...)` function. `master_agent.py` wraps each one in a LangChain `@tool` for typed invocation. Crashes are caught by `safe_run()` and downgraded to `{"anomaly_score": 0.0, "error": str(exc)}` so fusion always completes.

### 4.1 [`agents/preprocessing_agent.py`](agents/preprocessing_agent.py) — gate & shared crop

- SHA-256 + MD5 file hash (chain of custody).
- Face detection: RetinaFace → MediaPipe → dlib fallback (configurable).
- Writes `outputs/preprocessing/<stem>.json` with `face_bbox`, `face_bboxes`, `face_count`, `image_dims`, `hash_sha256`, `ela_score`, normalised crop path, EXIF.
- Returns the JSON path; `preprocessing_tool` reloads it and passes fields downstream.
- **If `face_detected == False`, the entire graph is aborted** via `abort_node`.

### 4.2 [`agents/geometry.py`](agents/geometry.py) — §5.1 facial geometry

- 68-point dlib landmarks on the provided bbox (singleton detector/predictor).
- Computes: `symmetry_index`, `jaw_curvature_deg`, `ear_alignment_px`, `philtrum_length`, `interocular_dist_px`, left/right `eye_aspect_ratio`, `lip_thickness_ratio`, `neck_face_boundary`.
- Compares each metric to population `NORMS` (mean, std) and emits a normalised `anomaly_score`.

### 4.3 [`agents/frequency_agent.py`](agents/frequency_agent.py) — §5.2 frequency / GAN

- **Method A — FFT radial spectrum (Durall 2020)**: produces `fft_mid_anomaly_db` and `fft_high_anomaly_db` as physically-meaningful dB excess values — real ≈ 0, StyleGAN2 ≈ +9–13 dB.
- **Method B — SVM on DCT/FFT features (Frank 2020)**: reuses the original `preprocess_image → extract_features → scaler → model` pipeline; emits `svm_fake_probability`.
- Fused inside the agent: `anomaly_score = 0.40 × fft_norm + 0.60 × svm_fake_probability`.
- Also contributes to the `gan_artefact` fusion slot via its `gan_probability` key.

### 4.4 [`agents/texture.py`](agents/texture.py) — §5.3 texture / skin consistency

- Tex-ViT-inspired (Dagar & Vishwakarma, 2024). Training-free by default; optional ML head at [`agents/texture_classifier.py`](agents/texture_classifier.py).
- Signals:
  - Gram-matrix texture features from 3 ResNet-inspired conv scales (uses torchvision if present, numpy approximation otherwise).
  - LBP uniformity.
  - Per-zone Earth Mover's / Wasserstein distance across face zones (jaw / neck / cheek) to flag seams.
- Returns zone EMDs, `lbp_uniformity`, `seam_detected`, `anomaly_score` as a Pydantic model (`.model_dump()` for the tool).

### 4.5 [`agents/vlm.py`](agents/vlm.py) — §5.4 VLM explainability

- Loads **LLaVA-1.5-7b** (`llava-hf/llava-1.5-7b-hf`, ~14 GB, cached in `~/.cache/huggingface`).
- Runs a forensic prompt on the pre-cropped 512×512 face → produces `vlm_caption`, `vlm_verdict ∈ {REAL, FAKE, UNCERTAIN}`, `vlm_confidence`.
- Grad-CAM heatmap is currently a **placeholder** (documented merge point for EfficientNet-B4 integration). Returns saliency zone lists (`high_*`, `medium_*`, `low_*`), `zone_gan_probability`, `heatmap_path`, and `anomaly_score`.

### 4.6 [`agents/biological_plausibility_agent.py`](agents/biological_plausibility_agent.py) — §5.5 biology

- Pupil-shape BIoU + corneal light-reflection IoU using MediaPipe FaceMesh (with an OpenCV-cascade compat shim for MediaPipe 0.10.x).
- Native outputs: `biou_left/right/avg`, `iou_reflect`, `solidity`, `convexity`, `aspect`, `hu1`, `reflection_count`, `prediction`.
- `biological_tool` in `master_agent.py` **maps these to the contracts schema**:
  - `rppg_snr ← avg_biou` (proxy — true rPPG needs video)
  - `corneal_deviation_deg ← (1 − iou_reflect) × 20`
  - `micro_texture_var ← solidity × 0.031`
  - `vascular_pearson_r ← None` (not derivable single-image)
  - `anomaly_score = 0.6·(1 − avg_biou) + 0.4·max(0, 1 − 2·iou_reflect)`, clamped to `[0, 1]`.

### 4.7 [`agents/metadata_agent.py`](agents/metadata_agent.py) — §5.6 provenance

- Consumes the preprocessing JSON path.
- Computes EXIF camera-present flag, software tag (Photoshop etc. is flagged), ELA chi-squared + map PNG, thumbnail mismatch, PRNU score/absence.
- Writes `outputs/metadata/<stem>.json`; full dict is returned verbatim.

---

## 5. Orchestration — `master_agent.py`

Key building blocks:

- **`@tool` wrappers** ([master_agent.py:100-300](master_agent.py#L100)): lazy-import each agent so unit tests don't pay the 14 GB model cost. Each returns a plain dict.
- **`_make_registry(state)`** ([master_agent.py:315](master_agent.py#L315)): the single source of truth for which agents run. Each entry is `{name, tool, invoke_args, score_key, fusion_module, enabled}`. Adding an agent = appending one entry.
- **`preprocess_node`**: runs `preprocessing_tool`; sets `preprocess_ok`. On `face_detected=False`, sets `fatal_error` and routes to `abort_node`.
- **`parallel_analysis_node`**: iterates the enabled registry and fans out with `asyncio.to_thread(tool.invoke, args)` inside `asyncio.gather`. Wraps each in `safe_run` → swallows exceptions into zero-score stubs + `errors` list.
- **`fusion_node`**: pulls `anomaly_score` out of every agent output, plus uses frequency's `gan_probability` (or its `anomaly_score` fallback) as the `gan_artefact` slot, then delegates to `bayesian_fusion`.
- **`report_node`**: flattens all agent outputs into one `ctx` dict (so the PDF template can look up any field by name), asks **Mistral-7B via `ChatOllama`** for a 3-sentence executive summary, then calls `ReportGenerator.generate(ctx)`. On LLM failure → deterministic fallback string. On PDF failure → JSON dump to `outputs/<case_id>.json`.
- **`abort_node`**: terminal failure — writes `outputs/<case_id>_ABORTED.json` with `decision: INCONCLUSIVE`.
- **`should_reflect`** (stub): would route back to `parallel_analysis_node` when `0.45 ≤ score ≤ 0.65` and `passes < 2`. Currently bypassed by a direct edge.

---

## 6. Bayesian fusion — `fusion/bayesian.py`

Log-odds ensemble, **not** a probability simplex.

For each module `i` with score `sᵢ` and weight `wᵢ`:

```
log_oddsᵢ   = wᵢ · log( sᵢ / (1 − sᵢ) )
total       = Σ log_oddsᵢ
final_score = sigmoid(total)
```

Implementation details worth noting:

- Scores are **clamped** to `(1e-6, 1 − 1e-6)` before `log` to avoid infinities (`_clamp`, `_log_odds`).
- **Input hygiene** (in `bayesian_fusion`): `None`, `NaN`, non-numeric, and out-of-range values are dropped with a warning; the fusion proceeds on the remaining modules. If everything is invalid, returns `0.5` (max uncertainty).
- **95% CI** via deterministic bootstrap (`seed=42`, `n=1000`) over module resampling with replacement. Duplicate module picks collapse to their mean. Fewer than 2 modules → degenerate `[final, final]` CI.
- **Decision thresholds** come from `contracts.py`:
  - `final_score ≥ 0.70` → `DEEPFAKE`
  - `final_score ≤ 0.35` → `AUTHENTIC`
  - else → `UNCERTAIN`
- **Interpretation bands**: ≥0.90 Very High, ≥0.75 High, ≥0.55 Moderate, else Low.
- Returns a dict conforming to `FUSION_KEYS` and is then validated.

### Fusion weights (from `contracts.FUSION_WEIGHTS`)

| Module       | Weight | §    |
|--------------|--------|------|
| geometry     | 0.15   | 5.1  |
| gan_artefact | 0.25   | 5.2  |
| frequency    | 0.25   | 5.2  |
| texture      | 0.20   | 5.3  |
| vlm          | 0.25   | 5.4  |
| biological   | 0.15   | 5.5  |
| metadata     | 0.15   | 5.6  |

Note: `config.json` carries a *separate* lower-precision weight set used by the older pipeline; `master_agent.py` goes through `fusion/bayesian.py` and therefore uses the `contracts.py` values.

---

## 7. Schemas — `contracts.py`

Single source of truth for every agent's output keys and the fusion output. It defines:

- **Per-agent key lists**: `PREPROCESSING_KEYS`, `GEOMETRY_KEYS`, `FREQUENCY_KEYS`, `TEXTURE_KEYS`, `BIOLOGICAL_KEYS`, `VLM_KEYS`, `METADATA_KEYS`, `FUSION_KEYS`, `REPORT_KEYS`.
- **`MODULE_SCORE_KEYS`**: the 7 fusion slots.
- **`FUSION_WEIGHTS`** and **`DEEPFAKE_THRESHOLD` / `AUTHENTIC_THRESHOLD`** used by `bayesian.py`.
- **`AUTHENTIC_BASELINES`**: population norms the agents compare measurements against (symmetry 0.92–1.00, jaw curvature < 5°, per-zone EMD < 0.08, etc.).
- **`validate(output, keys, agent_name)`**: raises `ValueError` listing any missing keys — used by `bayesian_fusion` and `ReportGenerator` to fail fast on contract drift.
- **`STUB_TEST_CASE_DFA_2025_TC_00471`**: a frozen measurement set from one specific deepfake image, used only to smoke-test the pipeline plumbing end-to-end.

---

## 8. Reporting — `report_agent/`

### [`report_agent/generate.py`](report_agent/generate.py) — `ReportGenerator`

- Mints a report ID `DFA-{YYYY}-TC-{6 hex}` and timestamp.
- Optionally calls **Mistral-7B via the local Ollama daemon** (`ollama.chat(model="mistral", …)`) to write the narrative sections — system prompt enforces court-grade tone ("state only what the data shows"). If Ollama isn't up, it returns an empty string so the template falls back to its own auto-prose.
- Delegates rendering to `template.build_report(ctx)` and returns a `REPORT_KEYS`-compliant dict (`report_path`, `report_id`, `generated_at`, `compliance_standards`, `analyst_name`, `lab_accreditation`, `hash_sha256_verified`).

Note: `master_agent.report_node` runs its **own** Mistral call via `langchain_ollama.ChatOllama` before invoking `ReportGenerator` and embeds the result as `ctx["narrative_text"]`; the generator may call Mistral again for the long-form sections. Both paths have fallbacks.

### [`report_agent/template.py`](report_agent/template.py) — ReportLab layout

- Dark forensic aesthetic (navy background, cyan accent, red/orange/green verdict badges) defined in the `T` design-token namespace.
- Builds an A4 multi-page PDF with a custom `BaseDocTemplate` + `PageTemplate`.
- Consumes the flat `ctx` dict assembled in `report_node`, so any agent measurement is directly accessible by key (e.g. `ctx["symmetry_index"]`, `ctx["ela_chi2"]`, `ctx["saliency_score"]`).

---

## 9. Runtime configuration — `config.json`

Operational knobs (not fusion weights used at runtime — those come from `contracts.py`):

- `stub_mode`: toggle deterministic stub outputs.
- `models.*`: paths to `efficientnet_ff++.pth`, `fusion_mlp.pth`, `shape_predictor_68_face_landmarks.dat`; HF repo for BLIP-2 (the VLM code actually loads LLaVA-1.5-7b).
- `output.*`: `report_dir`, `temp_dir`, `log_dir`, ID prefix.
- `preprocessing.*`: target size, min face px, MediaPipe confidence, dlib fallback flag.
- `frequency_agent.*`: FFT band percentiles, dB expected offsets, DCT thresholds, sub-method weights (FFT 0.55 / DCT 0.45).
- `geometry_agent.*`: per-metric weights (`symmetry`, `landmark`, `eye_distance`, `jaw`).
- `texture_agent.*`: LBP radius/points, Gabor orientations/scales, smoothness patch size.
- `biological_agent.*`: rPPG ROI regions, corneal blob min/max area.
- `vlm_agent.*`: model id, max tokens, Grad-CAM layer, GPU flag, caption-only fallback.
- `metadata_agent.*`: ELA quality & scale, PRNU patch size.
- `timeouts_seconds.*`: per-node timeouts (preprocess 30, geometry 15, frequency 30, texture 30, biological 60, vlm 120, metadata 15, fusion 10, report 60).
- `verdict_thresholds`: `fake=0.65`, `real=0.35` (older pipeline; `contracts.py` uses 0.70/0.35).

---

## 10. Output contract — the one rule

Every agent tool **must** return a dict with:

```python
{ "anomaly_score": float }   # 0.0 = genuine, 0.5 = uncertain, 1.0 = fake
```

- `None`, `NaN`, or out-of-range values are silently skipped by fusion (with a warning). That's the graceful-degradation path — the pipeline never crashes on a single bad agent.
- Everything else in the dict is passed through to the report ctx and the template can surface it.

---

## 11. Adding a new agent

1. Create `agents/my_agent.py` exporting `run(ctx_or_image_args) -> dict` that includes `"anomaly_score"`.
2. In `master_agent.py`, add a `@tool` wrapper with primitive-typed args (LangChain requirement).
3. Append an entry to `_make_registry(state)`.
4. Add its key to `MODULE_SCORE_KEYS` and `FUSION_WEIGHTS` in `contracts.py` if you want it fused.
5. If you need it surfaced in the PDF, read the agent's keys in `report_node` where the flat `ctx` is built.

---

## 12. Failure semantics (summary)

| Failure            | Handler                          | Effect on final score            |
|--------------------|----------------------------------|----------------------------------|
| Agent crash        | `safe_run` → `{score: 0.0}`      | Counted as strong-genuine signal unless you return `None` instead |
| Score `None`       | `bayesian_fusion` skips          | Module is dropped from the ensemble |
| Score `NaN` / OOR  | `bayesian_fusion` skips + warn   | Module is dropped                  |
| No face detected   | `abort_node`                     | `decision: INCONCLUSIVE`, no PDF   |
| Ollama unreachable | Fallback summary string          | Report still generated             |
| ReportLab crash    | JSON dump fallback               | `.json` at `outputs/<case>.json`   |
