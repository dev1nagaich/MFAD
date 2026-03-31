# MFAD Agent Interface Contract

**Version:** 1.1  
**Updated:** 2026-03-31  
**Audience:** Agent developers, handlers, and contributors  

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Agent Execution Flow](#agent-execution-flow)
4. [Output Contract](#output-contract) ⭐ **CRITICAL**
5. [Input Contract](#input-contract)
6. [Agent Registry](#agent-registry)
7. [Adding a New Agent](#adding-a-new-agent)
8. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## Overview

The MFAD (Multi-modal Forensic Analysis for Deepfakes) system is a **LangGraph-based orchestrator** that:
- Runs all analysis agents **in parallel** on a single input image
- **Collects anomaly scores** from each agent (0.0 = genuine, 1.0 = fake)
- Applies **Bayesian log-odds fusion** to combine scores into a final decision
- Generates a **forensic PDF report** with Mistral-7B narrative

**Key principle:** Agents are **independent, stateless, and concurrent**. They:
- ✅ Receive well-defined input dicts
- ✅ Must return a dict with an `anomaly_score` (float 0.0–1.0)
- ⚠️ Are expected to gracefully handle missing/invalid inputs
- ❌ Cannot communicate with other agents

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  MFAD Master Agent (master_agent.py)                             │
│  ─ LangGraph StateGraph                                          │
│  ─ Single case = Single state dict flowing through nodes         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  preprocess_node │  ← Image read, face detect,
                    └──────────────────┘     ELA computation
                              │
                        ┌─────▼─────┐   No face detected?
                        │ CLIP gate? │──→ ABORT PIPELINE
                        └─────┬─────┘
                              │ face_bbox = [x1,y1,x2,y2]
                              ▼
          ┌───────────────────────────────────────┐
          │  parallel_analysis_node                │
          │  ─ asyncio.gather() all agents        │
          │  ─ Safe-run wrapper catches crashes   │
          │                                       │
          │  Agents run concurrently:             │
          │    • geometry_tool()                  │
          │    • frequency_tool()                 │
          │    • texture_tool()                   │
          │    • vlm_tool()                       │
          │    • biological_tool()                │
          │    • metadata_tool()                  │
          │                                       │
          │  agent_outputs = {                    │
          │    "geometry": {...},                 │
          │    "frequency": {...},                │
          │    "texture": {...},                  │
          │    "vlm": {...},                      │
          │    "biological": {...},               │
          │    "metadata": {...}                  │
          │  }                                    │
          └───────────────────┬───────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  fusion_node  │   ← Extract anomaly_score
                      └───────────────┘     from each agent
                              │
                              ▼
          ┌────────────────────────────────────┐
          │  Bayesian Log-Odds Fusion          │
          │  (fusion/bayesian.py)              │
          │                                    │
          │  per_module_scores = {             │
          │    "geometry": 0.412,              │
          │    "frequency": 0.410,             │
          │    "texture": 0.033,               │
          │    "vlm": 0.390,                   │
          │    "biological": 0.000,            │
          │    "metadata": 0.300,              │
          │    "gan_artefact": 0.410           │
          │  }                                 │
          │                                    │
          │  final_score = sigmoid(Σ weights*  │
          │                log(p/(1-p)))       │
          │  final_score ≈ 0.261               │
          │  decision = AUTHENTIC              │
          │  CI = [0.044, 0.424]               │
          └────────────┬───────────────────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  report_node       │
              │  ─ Flatten all     │
              │    agent outputs   │
              │  ─ Mistral-7B      │
              │    narrative       │
              │  ─ PDF generation  │
              └────────────────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  Pipeline Complete │
              │  ─ PDF report ✓    │
              │  ─ JSON metadata ✓ │
              └────────────────────┘
```

---

## Agent Execution Flow

### **Phase 1: Orchestration (master_agent.py)**

```python
# Step 1: User invokes CLI
python master_agent.py test_images/IMG_6930.JPG --analyst "Dev Nagaich"

# Step 2: Create initial state
state = MFADState(
    image_path="test_images/IMG_6930.JPG",
    case_id="DFA-2026-TC-544A1D84",  # Auto-generated UUID
    analyst_name="Dev Nagaich",
)

# Step 3: Run preprocessing (CLIP gate)
state = preprocess_node(state)
if not state["preprocess_ok"]:
    raise RuntimeError("No face detected")
# Now state["face_bbox"] = [359, 904, 1122, 1667]

# Step 4: Run all agents in parallel
state = parallel_analysis_node(state)
# state["agent_outputs"] now contains results from all 6 agents

# Step 5: Fuse scores
state = fusion_node(state)
# state["fusion"] = {"final_score": 0.261, "decision": "AUTHENTIC", ...}

# Step 6: Generate report
state = report_node(state)
# state["report_path"] = "reports/DFA-2026-TC-544A1D84.pdf"
```

### **Phase 2: Agent Registration & Invocation**

Each agent is wrapped as a LangChain **@tool** in `master_agent.py`:

```python
@tool
def geometry_tool(image_path: str, face_bbox: list) -> dict:
    """Computes 68-point dlib landmarks + symmetry analysis."""
    from agents.geometry import run
    
    img = cv2.imread(image_path)
    return run(image_bgr=img, face_present=True)
```

The **agent registry** (`_make_registry(state)`) maps:

```python
{
    "name":          "geometry",                    # Logical name
    "tool":          geometry_tool,                 # LangChain @tool
    "invoke_args":   {"image_path": "...", "face_bbox": [...]},
    "score_key":     "anomaly_score",               # Key to extract from result
    "fusion_module": "geometry",                    # Fusion dict key
    "enabled":       True,
}
```

### **Phase 3: Safe Execution**

Each agent runs in a **safe_run()** wrapper:

```python
async def safe_run(descriptor):
    try:
        result = await asyncio.to_thread(tool_fn.invoke, invoke_args)
        score = result.get("anomaly_score", "N/A")
        log.info("  ✓ %s | anomaly_score=%s", name, score)
        return name, result
    except Exception as exc:
        log.error("  ✗ %s | %s", name, exc)
        return name, {"anomaly_score": 0.0, "error": str(exc)}
```

**Behavior:**
- ✅ Agent crashes? Returns `{"anomaly_score": 0.0, "error": "..."}`
- ✅ Agent timeout? Killed, returns score=0.0
- ✅ Agent returns `None` anomaly_score? Bayesian fusion **skips** it gracefully
- ✅ Agent returns out-of-range score (e.g., 172767168.0)? Bayesian fusion **validates & skips**

---

## Output Contract

### ⭐ **CRITICAL: Every agent MUST return a dict with `anomaly_score`**

```python
# Minimal valid output:
{
    "anomaly_score": 0.412  # REQUIRED — float in [0.0, 1.0]
}

# Full output (recommended):
{
    "anomaly_score": 0.412,        # REQUIRED
    "measurement_1": 123.45,       # Optional — any domain-specific measurements
    "measurement_2": "REAL",       # Optional — any data type
    "confidence": 0.95,            # Optional — your agent's self-confidence
    # ... (any other fields)
}
```

### **Anomaly Score Semantics**

- **0.0** = Definitely genuine / no signs of manipulation
- **0.5** = Maximize uncertainty (same as random chance)
- **1.0** = Definitely fake / maximum manipulation evidence
- **None / NaN / negative / > 1.0** = **Bayesian fusion automatically skips**

### **Example Outputs**

#### Geometry Agent
```python
return {
    "symmetry_index": 0.89,         # Measurement
    "jaw_curvature_deg": 15.3,      # Measurement
    "ear_alignment_px": 2.1,        # Measurement
    "philtrum_length": 22.5,        # Measurement
    ...
    "anomaly_score": 0.412,         # REQUIRED ✓
}
```

#### Frequency Agent
```python
return {
    "fft_mid_anomaly_db": -12.5,           # dB spectrum deviation
    "fft_high_anomaly_db": -8.3,           # dB spectrum deviation
    "fft_ultrahigh_anomaly_db": None,      # Not always computed
    "gan_probability": 0.41,                # Optional GAN probability
    "upsampling_grid_detected": False,      # Boolean finding
    ...
    "anomaly_score": 0.410,                # REQUIRED ✓
}
```

#### VLM Agent
```python
return {
    "heatmap_path": "temp/heatmap_1774965330.png",
    "vlm_caption": "This is a face...",
    "vlm_verdict": "REAL",
    "vlm_confidence": 0.700,
    "saliency_score": 0.39,
    "high_activation_regions": ["left_eye", "mouth"],
    "medium_activation_regions": ["cheek"],
    "low_activation_regions": ["forehead"],
    "zone_gan_probability": {"left_eye": 0.1, "mouth": 0.05},
    ...
    "anomaly_score": 0.390,                # REQUIRED ✓
}
```

#### Biological Agent
```python
return {
    # Raw measurements
    "biou_left": 0.52,
    "biou_right": 0.54,
    "avg_biou": 0.53,
    "iou_reflect": 0.02,
    "solidity": 0.95,
    "convexity": 0.98,
    "aspect": 1.02,
    "hu1": 0.001,
    "reflection_count": 2,
    "landmarks_found": True,
    "prediction": "real",
    # Mapped to contracts.py BIOLOGICAL_KEYS
    "rppg_snr": 0.5300,                    # Proxy from avg_biou
    "corneal_deviation_deg": 0.4000,       # Proxy from iou_reflect
    "micro_texture_var": 0.0295,           # Proxy from solidity
    "vascular_pearson_r": None,            # Not available
    ...
    "anomaly_score": 0.000,                # REQUIRED ✓
}
```

#### Metadata Agent
```python
return {
    "exif_camera_present": False,
    "software_tag": None,
    "ela_chi2": 172767168.0,               # (Will be skipped if > 1.0 by Bayesian)
    "ela_map_path": "outputs/metadata/IMG_6930_ELA.png",
    "thumbnail_mismatch": False,
    "prnu_absent": True,
    "prnu_score": 0.00490,
    ...
    "anomaly_score": 0.300,                # REQUIRED ✓
}
```

---

## Input Contract

### **All agents always receive:**

```python
state: MFADState = {
    "image_path": "test_images/IMG_6930.JPG",    # Absolute or relative path
    "case_id": "DFA-2026-TC-544A1D84",           # Unique case identifier
    "analyst_name": "Dev Nagaich",               # Who submitted the case
    "face_bbox": [359, 904, 1122, 1667],        # [x1, y1, x2, y2] from preprocessing
    "preprocessing": {...},                      # Full preprocessing dict
}
```

### **Individual Agent Inputs**

Each agent's `@tool` function signature determines what it receives:

#### Geometry
```python
def geometry_tool(image_path: str, face_bbox: list) -> dict:
    image_path = "test_images/IMG_6930.JPG"  # Will be provided
    face_bbox = [359, 904, 1122, 1667]       # Will be provided
```

#### Frequency
```python
def frequency_tool(image_path: str, face_bbox: list) -> dict:
    image_path = "test_images/IMG_6930.JPG"  # Will be provided
    face_bbox = [359, 904, 1122, 1667]       # Will be cropped, saved as temp file
```

#### Texture
```python
def texture_tool(image_path: str, face_bbox: list) -> dict:
    image_path = "test_images/IMG_6930.JPG"  # Will be provided
    face_bbox = [359, 904, 1122, 1667]       # Will be provided
```

#### VLM
```python
def vlm_tool(image_path: str, face_bbox: list, face_crop_path: str) -> dict:
    image_path = "test_images/IMG_6930.JPG"         # Will be provided
    face_bbox = [359, 904, 1122, 1667]             # Will be provided
    face_crop_path = "temp/normalised_face.png"    # Pre-cropped 512x512 face
```

#### Biological
```python
def biological_tool(image_path: str, face_bbox: list) -> dict:
    image_path = "test_images/IMG_6930.JPG"  # Will be provided
    face_bbox = [359, 904, 1122, 1667]       # Will be provided
```

#### Metadata
```python
def metadata_tool(preprocessing_json_path: str) -> dict:
    preprocessing_json_path = "outputs/preprocessing/IMG_6930.json"  # Will be provided
```

---

## Agent Registry

### **Current Registry** (in `master_agent.py`)

```python
def _make_registry(state) -> list[dict]:
    """
    Builds agent descriptors from current state.
    Called fresh at start of parallel_analysis_node.
    """
    return [
        {
            "name":          "geometry",
            "tool":          geometry_tool,
            "invoke_args":   {"image_path": ..., "face_bbox": ...},
            "score_key":     "anomaly_score",
            "fusion_module": "geometry",
            "enabled":       True,
        },
        {
            "name":          "frequency",
            "tool":          frequency_tool,
            "invoke_args":   {"image_path": ..., "face_bbox": ...},
            "score_key":     "anomaly_score",
            "fusion_module": "frequency",
            "enabled":       True,
        },
        # ... (4 more agents)
    ]
```

### **How Fusion Uses the Registry**

The `fusion_node` extracts anomaly scores and maps them to fusion module names:

```python
# In fusion_node():
direct_map = {
    "geometry":   "geometry",      # agent name → fusion module key
    "frequency":  "frequency",
    "texture":    "texture",
    "vlm":        "vlm",
    "biological": "biological",
    "metadata":   "metadata",
}

per_module_scores = {}
for agent_name, fusion_key in direct_map.items():
    output = agent_outputs.get(agent_name, {})
    score = output.get("anomaly_score")
    if score is not None:
        per_module_scores[fusion_key] = float(score)

# Then: bayesian_fusion(per_module_scores, compute_ci=True)
```

---

## Adding a New Agent

### **Step 1: Create Your Agent Module**

File: `agents/my_agent.py`

```python
"""
agents/my_agent.py — My Custom Forensic Analysis
================================================
Owner: Your Name
"""

def run(ctx: dict) -> dict:
    """
    Analyze image for my custom metric.
    
    Args:
        ctx: {
            "image_path": str,
            "face_bbox": [x1, y1, x2, y2],
            # ... (any other fields you need)
        }
    
    Returns:
        {
            "my_measurement_1": float,
            "my_measurement_2": str,
            "anomaly_score": float,  # REQUIRED — must be in [0.0, 1.0]
        }
    """
    image_path = ctx["image_path"]
    face_bbox = ctx["face_bbox"]
    
    # Your analysis here
    my_metric = compute_my_metric(image_path, face_bbox)
    my_anomaly_score = 0.5 if my_metric > threshold else 0.2
    
    return {
        "my_measurement_1": my_metric,
        "anomaly_score": my_anomaly_score,
    }
```

### **Step 2: Create a @tool Wrapper**

In `master_agent.py`, add:

```python
@tool
def my_agent_tool(image_path: str, face_bbox: list) -> dict:
    """Your description here."""
    from agents.my_agent import run
    
    ctx = {
        "image_path": image_path,
        "face_bbox": face_bbox,
    }
    return run(ctx)
```

### **Step 3: Register in _make_registry()**

Add entry to the list in `_make_registry()`:

```python
{
    "name":          "my_agent",
    "tool":          my_agent_tool,
    "invoke_args":   {"image_path": image_path, "face_bbox": face_bbox},
    "score_key":     "anomaly_score",
    "fusion_module": "my_agent",            # Or map to existing module
    "enabled":       True,
},
```

### **Step 4: Add Fusion Weights** (Optional, but recommended)

In `contracts.py`, update:

```python
FUSION_WEIGHTS = {
    # ... existing weights ...
    "my_agent": 0.15,  # Weight between 0 and 1
}
```

### **Step 5: Test Standalone**

```bash
python -m pytest tests/test_my_agent.py
```

### **Step 6: Run Full Pipeline**

```bash
python master_agent.py test_images/IMG_6930.JPG --analyst "Dev"
```

---

## Bayesian Fusion Logic

### **Why Bayesian?**

Each agent's anomaly_score is treated as $P(\text{fake} | \text{evidence from agent})$.

We use **log-odds**, not probabilities, because:
1. ✅ Avoids underflow (log keeps numbers well-scaled)
2. ✅ Handles many independent sources naturally (addition)
3. ✅ Weights are interpretable (more = more reliable)

### **The Math**

For each module $i$ with score $s_i$ and weight $w_i$:

$$\text{log-odds}_i = w_i \times \log\left(\frac{s_i}{1-s_i}\right)$$

Total:

$$\text{log-odds}_{\text{total}} = \sum_i w_i \times \log\left(\frac{s_i}{1-s_i}\right)$$

Final score (back to probability):

$$\text{final\_score} = \text{sigmoid}(\text{log-odds}_{\text{total}}) = \frac{1}{1 + e^{-\text{log-odds}_{\text{total}}}}$$

### **Example Calculation**

From your last run:

```
geometry:     0.412   × 0.15  → log-odds_geometry     ≈ -0.375
frequency:    0.410   × 0.25  → log-odds_frequency    ≈ -0.360
texture:      0.033   × 0.20  → log-odds_texture      ≈ -3.396
vlm:          0.390   × 0.25  → log-odds_vlm          ≈ -0.451
biological:   0.000   × 0.15  → log-odds_biological   ≈ -9.210 (clamped)
metadata:     0.300   × 0.15  → log-odds_metadata     ≈ -0.811
gan_artefact: 0.410   × 0.25  → log-odds_gan          ≈ -0.360
                                ─────────────────────────────
                                log-odds_total         ≈ -14.963

final_score = sigmoid(-14.963) ≈ 0.0357 ← AUTHENTIC
confidence_interval (95%) = [0.044, 0.424]
decision: AUTHENTIC (final_score ≤ 0.35)
```

### **Handling Invalid Scores**

The **updated bayesian.py** now validates all scores:

```python
def bayesian_fusion(scores: dict[str, float]) -> dict:
    """
    Fuses validated scores. Skips:
      • None values
      • NaN (float('nan'))
      • Out-of-range (< 0 or > 1)
    
    Returns final_score, decision, confidence_interval.
    """
    valid_scores = {}
    for module, score in scores.items():
        if score is None:
            log.debug("fusion: module '%s' returned None — skipped", module)
            continue
        if math.isnan(score):
            log.warning("fusion: module '%s' returned NaN — skipped", module)
            continue
        if not (0.0 <= score <= 1.0):
            log.warning(
                "fusion: module '%s' score %.2e out of range [0, 1] — skipped",
                module, score
            )
            continue
        valid_scores[module] = score
    
    return _fuse_scores(valid_scores)
```

---

## Debugging & Troubleshooting

### **Common Issues**

#### **Agent returns `anomaly_score: None`**
- ✅ **Expected behavior** — Bayesian fusion skips it with a debug log
- 📋 Check agent logs; it may have failed gracefully
- ✅ Pipeline continues with remaining agents

#### **Agent returns out-of-range score (e.g., 172767168.0)**
- ✅ **Expected behavior** — Bayesian fusion validates and skips it
- 📋 Agent may have a normalization bug
- 🔧 Fix: Ensure agent clamps/normalizes output to [0, 1]

```python
# In your agent's run() function:
anomaly_score = np.clip(computed_score, 0.0, 1.0)
```

#### **Agent crashes entirely**
- ✅ Orchestrator catches it and returns `{"anomaly_score": 0.0, "error": "..."}`
- 📋 Check master_agent logs for exception traceback
- 🔧 Fix: Add error handling in your agent's run() function

```python
try:
    result = compute_some_analysis()
except Exception as e:
    log.warning("My agent failed: %s", e)
    return {"anomaly_score": 0.0, "error": str(e)}
```

#### **Final score is always 0.5 (maximum uncertainty)**
- 📋 All agents may be returning None/NaN/invalid scores
- 🔧 Check agent output keys match `anomaly_score` exactly
- 🔧 Check validation is not too strict

#### **Report path shows as `None` in final output**
- ✅ **Fixed in v1.1** — `report_node` now includes `report_path` in `master_output`
- 🔧 If still broken, check that `report_node` is setting the output:
  ```python
  master_output["report_path"] = report_path
  ```

### **Logging Levels**

```bash
# See all debug info (very verbose)
python master_agent.py test_images/IMG_6930.JPG 2>&1 | grep -E "DEBUG|INFO|WARNING|ERROR"

# See only warnings and errors
python master_agent.py test_images/IMG_6930.JPG 2>&1 | grep -E "WARNING|ERROR"

# See fusion logic in detail
python master_agent.py test_images/IMG_6930.JPG 2>&1 | grep "fusion"
```

### **Test a Single Agent Independently**

```python
# test_my_agent_standalone.py
from agents.my_agent import run

ctx = {
    "image_path": "test_images/IMG_6930.JPG",
    "face_bbox": [359, 904, 1122, 1667],
}

result = run(ctx)
print(f"anomaly_score = {result.get('anomaly_score')}")
assert 0.0 <= result['anomaly_score'] <= 1.0, "Score out of range!"
```

---

## Summary: The Output Contract (TL;DR)

| Requirement | Value | Notes |
|---|---|---|
| **Return type** | `dict` | Python dictionary |
| **Required key** | `anomaly_score` | Float, EXACT spelling |
| **Score range** | `[0.0, 1.0]` | 0=genuine, 1=fake |
| **Invalid values** | `None`, `NaN`, `<0`, `>1` | Automatically skipped by fusion |
| **Extra keys** | Any | Passed to report; won't break pipeline |
| **Timeout** | 2 seconds | Longer → agent killed, score=0.0 |
| **Crash handling** | Caught | Error logged, score=0.0, pipeline continues |

**Remember:** Your agent's `anomaly_score` is one vote in a democratic fusion. Make it honest, well-calibrated, and grounded in your forensic measurements.

---

**Document maintained by:** Dev Nagaich  
**Last updated:** 2026-03-31
