# Texture Agent — Status & Implementation Audit

**Owner:** Manya Gupta
**Last updated:** 2026-04-25
**Module:** [agents/texture_agent.py](agents/texture_agent.py)
**Papers:** TAD (Gao et al., EAAI 2024) + NPR (Tan et al., CVPR 2024)

This is the single source of truth for the texture agent. It replaces the
deleted scratch docs (`TEXTURE_AGENT_README.md`, `TEXTURE_AGENT_TRAINING_GUIDE.md`,
`IMPLEMENTATION_SUMMARY.md`, `FINAL_CHECKLIST.txt`, `DELIVERY_SUMMARY.py`,
`IMPLEMENTATION_COMPLETE.py`, `REORGANIZATION_SUMMARY.py`,
`TEXTURE_AGENT_STRUCTURE.py`).

---

## 1. What's in place today

| Asset | Path | Purpose |
|---|---|---|
| Core agent | [agents/texture_agent.py](agents/texture_agent.py) | TAD + NPR forensics, `TextureAgent.analyze`, `run()` legacy wrapper |
| Evaluator | [texture_agent_evaluator.py](texture_agent_evaluator.py) | Standalone face detection + dataset sweep |
| Report formatter | [texture_report_formatter.py](texture_report_formatter.py) | Text / Markdown / JSON forensic reports |
| Smoke tests | [test_texture_agent.py](test_texture_agent.py), [test_texture_complete.py](test_texture_complete.py), [quick_test.py](quick_test.py), [simple_test.py](simple_test.py) | CLI tests on local images |
| Master integration | `texture_tool` in [master_agent.py:184](master_agent.py#L184) | Fans out from `parallel_analysis_node` |
| Contract | `TEXTURE_KEYS` in [contracts.py:69](contracts.py#L69) | Output schema (now expanded; see §3) |

The agent runs in **heuristic-fusion mode** by default. A trained
`RandomForestClassifier` path exists (`train_classifier`, `_predict_with_classifier`)
but **no checkpoint is loaded** — the 187 MB files in [checkpoints/](checkpoints/)
are EfficientNet snapshots from [train_efficientnet.py](train_efficientnet.py),
not from this agent.

### Output schema produced by `analyze()`

`TextureAnalysisResult` (dataclass) → `to_dict()`:

```
texture_fake_probability  is_fake  anomaly_score
jaw_emd  neck_emd  cheek_emd  lbp_uniformity  seam_detected
zone_results  zone_scores  gram_distances  multi_scale_consistency
analyst_note  processing_notes
```

Per-zone (`ZoneScore`): `zone_name, emd_score, lbp_uniformity, npr_residual,
texture_variance, color_delta_e, risk_level`. Seven zones: forehead, nose,
cheek_L, cheek_R, perioral, jaw, neck.

---

## 2. Implementation audit — TAD + NPR

The implementation is functional but has **five real correctness issues**
that should be fixed before any meaningful evaluation. None of them are
catastrophic (the agent will run end-to-end and produce a number), but the
"EMD" and "NPR" labels currently overstate what is being computed.

### 2.1 `_compute_emd_matrix` is not EMD — BUG
[agents/texture_agent.py:386-434](agents/texture_agent.py#L386-L434)

The function ingests only the **scalar uniformity** per zone and computes
`abs(uni_a - uni_b)`. That is a one-dimensional difference, not Earth Mover's
Distance. True TAD-style EMD needs the **full LBP histogram** per zone (e.g.
59 bins for `method="uniform"`, P=8) and Wasserstein distance over those bins.

**Fix:** in `_compute_zone_lbp`, also store
`np.bincount(lbp.astype(int).ravel(), minlength=59) / N` as
`zone_lbp[f"{zone}_hist"]`, then in `_compute_emd_matrix` use
`scipy.stats.wasserstein_distance(hist_a, hist_b)` (or
`cv2.EMD` with a ground-distance matrix).

### 2.2 `nose_avg` and `perioral_avg` referenced but never set — BUG
[agents/texture_agent.py:840-841](agents/texture_agent.py#L840-L841) reads
`emd_scores.get('nose_avg', 0.05)` and `emd_scores.get('perioral_avg', 0.05)`,
but `_compute_emd_matrix` only sets `jaw_avg`, `neck_avg`, `cheek_L_avg`.
Both features default to the constant 0.05 on every image, so two of the 14
classifier features carry no signal.

**Fix:** add `nose_avg = mean of (forehead↔nose, nose↔cheek_L, nose↔cheek_R)`
and `perioral_avg = mean of (cheek_L↔perioral?, cheek_R↔perioral?)` — but the
current adjacency list does not even include `perioral`, so it has to be
extended.

### 2.3 LBP risk direction is inverted — BUG
[agents/texture_agent.py:773](agents/texture_agent.py#L773)

The fusion correctly treats **high** LBP uniformity as fake (line 973:
`(lbp_uni − 0.70) / 0.30`, max at uni=1.0 → max fake). The TAD paper says
GAN over-smoothing **raises** the uniform-pattern fraction.

But `_assess_zone_risk` adds risk when `lbp_uniformity < CRITICAL=0.70`
— the opposite direction. So per-zone `risk_level` ("critical" / "elevated")
contradicts the global probability. The analyst note that names "critical
zones" therefore picks the wrong zones.

**Fix:** flip to `if lbp_uniformity > LBP_UNIFORM_AUTHENTIC: risk += 0.5`
(or pick a high threshold like 0.92).

### 2.4 `_pixel_correlation` is misnamed and is not the NPR paper's signal
[agents/texture_agent.py:482-507](agents/texture_agent.py#L482-L507)

The function computes `std(diffs) / mean(|diffs|)` over neighbour
differences — a contrast / coefficient-of-variation measure, **not** a
correlation. The NPR difference (`abs(corr_4 − corr_8)`) is still a valid
"how similar are 4-conn and 8-conn neighborhoods" feature, but it is far
from what the paper proposes.

The NPR paper (Tan et al. CVPR 2024) defines the **NPR residual** as
`r = x − up(down(x, 2))` (resample residual that exposes upsampling grids),
then trains a small CNN on `r`. A faithful analytical proxy is:

```
r = x.astype(float) - cv2.resize(cv2.resize(x, half), full, INTER_NEAREST)
npr_4 = np.corrcoef(r[:-1].ravel(), r[1:].ravel())[0,1]   # vertical
npr_8 = np.corrcoef(r[:-1,:-1].ravel(), r[1:,1:].ravel())[0,1]  # diagonal
score = abs(npr_4 - npr_8)
```

**Fix:** introduce the resample residual step before measuring 4↔8
asymmetry. Drop the misleading "correlation" comment.

### 2.5 No actual histograms anywhere
The agent never materialises an LBP histogram, a colour histogram (the gram
"descriptor" in [_compute_gram_distances](agents/texture_agent.py#L669-L715)
is a 10-bin per-channel sum, not a Gram matrix), or an NPR feature map. This
is fine for a heuristic detector, but it caps the ceiling — the trained-RF
path on a 14-d hand-crafted vector cannot match a CNN on the residual.

### 2.6 Things that are correct
- 7-zone definition matches the report's anatomy.
- Boundary seam via SSIM at jaw↔neck is a sensible proxy.
- Multi-scale consistency (down + up + SSIM) captures upsampling fragility.
- CIE Lab ΔE per zone is correct.
- Gabor bank + per-zone variance is correct.
- `texture_log_odds` for Bayesian fusion is correct.
- `run()` legacy interface returns all `TEXTURE_KEYS` (after the contract
  expansion in §3).

---

## 3. `contracts.py` changes

`TEXTURE_KEYS` was the minimal 6-key legacy schema; the agent already returns
8 more keys. Expanded the contract to match what is actually produced:

```
jaw_emd, neck_emd, cheek_emd, lbp_uniformity, seam_detected,
texture_fake_probability, is_fake,
zone_results, zone_scores, gram_distances, multi_scale_consistency,
analyst_note, processing_notes,
anomaly_score
```

`AUTHENTIC_BASELINES` gained NPR / Gabor / colour-ΔE / decision-threshold
entries so the report and zone risk logic share one source of truth:

```
emd_suspicious, emd_anomalous, lbp_uniformity_critical,
npr_authentic_max, npr_anomalous_min,
gabor_variance_min, gabor_variance_critical,
color_delta_e_min, texture_decision_threshold
```

The fusion contract (`MODULE_SCORE_KEYS`, `FUSION_WEIGHTS`) is unchanged —
`texture` still carries weight 0.20.

---

## 4. Do we need to train anything?

**Short answer:** not strictly required to ship, but yes if we want the
texture agent's number to be calibrated rather than hand-tuned.

There are three viable training options, in increasing cost and payoff:

### (a) Skip training — heuristic fusion only
- `TextureAgent()` with no `classifier_path` runs the weighted-fusion path
  ([_fuse_scores](agents/texture_agent.py#L949)).
- Pros: deterministic, interpretable, zero deps.
- Cons: probability is hand-tuned; expect AUC in the high-0.7s on
  cross-dataset eval; weights `(emd 0.25, lbp 0.25, npr 0.20, seam/colour/
  gabor 0.10)` were never empirically validated.

### (b) Train the existing 14-d RandomForest — RECOMMENDED FIRST PASS
- `TextureAgent.train_classifier(X, y)` is already wired up; saves a
  pickle the agent loads via `classifier_path`.
- Cost: minutes on CPU once features are extracted. Feature extraction over
  ~20 k images takes ≈30-60 min on a single CPU thread, trivially
  parallelisable.
- Suggested split:
  - Train: balanced ~10 k real (FFHQ + celebA-HQ) + ~10 k fake
    (100KFake_10K / TPDNE).
  - In-domain val: held-out 20 % of the same.
  - Cross-domain test: FF++ Deepfakes + FaceSwap + NeuralTextures, plus
    stable-diffusion 512/768/1024.
- **Blockers before training:** §2.1, §2.2, §2.3 must be fixed first or two
  features carry zero signal and the LBP feature has the wrong sign.

### (c) Train an NPR-CNN on the resample residual — SOTA path
- Faithful reproduction of the NPR paper: tiny ResNet on
  `r = x − up(down(x))`, binary cross-entropy over real/fake.
- Cost: a few hours on one A6000; can use 1-2 GPUs of the 8 available.
- Replaces `npr_residual` in the per-zone metrics with a learned probability.
- Worth it if the project wants competitive numbers vs the NPR paper's
  reported 92.2 % mean accuracy across 28 generators.

### Recommendation
Do (b) immediately after fixing §2.1-§2.3. If we end up needing more
generalisation across diffusion / unseen GANs, layer (c) on top — they
are not exclusive (the CNN's output becomes one more feature into the RF).

---

## 5. Changes applied (2026-04-25 session)

- **Bug §2.1 fixed**: `_compute_zone_lbp` now stores the full 10-bin
  uniform-LBP histogram per zone; `_compute_emd_matrix` uses
  `scipy.stats.wasserstein_distance` over those histograms — true EMD.
- **Bug §2.2 fixed**: per-zone aggregates now include `nose_avg`,
  `perioral_avg`, `forehead_avg`, `cheek_R_avg` (and the cheek↔cheek edge is
  what `cheek_emd` reports). The 14-d feature vector no longer carries two
  constant entries.
- **Bug §2.3 fixed**: `_assess_zone_risk` now flags **high** LBP uniformity
  as suspicious, matching the fusion direction and the TAD paper.
- **Bug §2.4 fixed**: `_compute_npr_residuals` now computes the true NPR
  resample residual `r = x − up(down(x, ½))` and measures Pearson
  correlation asymmetry between 4-conn and 8-conn neighbourhoods of `r`.
- **Inference auto-loads** the trained classifier from
  `checkpoints/texture_checkpoint/texture_rf.pkl` (default in
  `TextureAgent.__init__`). Falls back to heuristic fusion if absent.
- **Checkpoints reorganised**:
  `checkpoints/vlm_checkpoint/*.pt` (the EfficientNet snapshots driven by
  `train_efficientnet.py`) and `checkpoints/texture_checkpoint/`
  (trained RF + metrics from `train_texture.py`).
- **Trainer added**: [train_texture.py](train_texture.py) runs feature
  extraction in parallel and fits a `HistGradientBoostingClassifier` with
  `class_weight="balanced"`, early stopping, and a held-out 20 % test split.
- **Redundant scratch removed**: `test_texture_complete.py`,
  `evaluation_script.py`, `quick_test.py`, `simple_test.py` were stripped
  in favour of the canonical `test_texture_agent.py` (pytest) +
  `texture_agent_evaluator.py` (FaceDetector + dataset eval) +
  `texture_report_formatter.py` (report).

## 6. Why this dataset selection (and not a broader one)

Texture's UNIQUE strength inside MFAD is the **face-swap boundary seam**
(jaw↔neck SSIM dissimilarity, EMD jump at the swap boundary). Other
agents already own the rest of the texture-adjacent failure modes:

| Failure mode | Owning agent | Reason texture is *not* trained on it |
|---|---|---|
| GAN spectral fingerprint / upsampling grids | Frequency agent | spectral, not pixel-domain |
| Stable-diffusion semantic anomalies | VLM agent | no upsampling grid |
| Mouth-only reenactment (Face2Face, NeuralTextures) | Biological agent | localised to mouth + eyes |
| Attribute edits (AttGAN, STGAN, …) | Geometry agent | landmark drift |

Including (e.g.) 100KFake/TPDNE in texture training would teach the agent
to overlap with the frequency agent. The whole point of the multi-agent
split is to make each detector confident on its own domain and abstain
elsewhere, then let Bayesian fusion combine them. So texture trains on:

- Real (label 0): `original` (FF++), `Flickr-Faces-HQ_10K`, `celebA-HQ_10K`
- Fake (label 1): `Deepfakes`, `FaceSwap`, `FaceShifter` (FF++ face-swaps)

## 7. Open questions for next step

- Re-verify metrics on the held-out per-source test split after training
  completes (currently the trainer reports metrics on a 20 % split inside
  training data, not on the dataset's official `test/` folders).
- Optional follow-up: add an NPR-CNN head trained on resample residuals on
  one A6000 — would replace the analytical NPR feature with a learned
  probability and lift AUC further on unseen face-swap variants.
