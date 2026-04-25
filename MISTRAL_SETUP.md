# Running MFAD Report Generation with Ollama Mistral

This guide explains how to set up and use Ollama Mistral-7B for the MFAD report narrative generation pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Pull Mistral to Ollama](#step-1-pull-mistral-to-ollama)
3. [Step 2: Verify Mistral is Running](#step-2-verify-mistral-is-running)
4. [Step 3: Run MFAD Inference with Mistral](#step-3-run-mfad-inference-with-mistral)
5. [Step 4: Verify Report Generation](#step-4-verify-report-generation)
6. [Troubleshooting](#troubleshooting)
7. [Model Options](#model-options)

---

## Prerequisites

- **Ollama** installed and running
  - Service command: `systemctl status ollama` or check if already running
  - Server endpoint: `http://127.0.0.1:11434` (default)
  
- **MFAD dependencies** installed (from `requirements.txt`)
  
- **Disk space**: ~6–10 GB free (Mistral-7B ≈ 4 GB)

---

## Step 1: Pull Mistral to Ollama

### Option A: Using Ollama CLI (Recommended)

Open a new terminal and run:

```bash
ollama pull mistral
```

**What happens:**
- Downloads Mistral-7B model (~4 GB)
- Caches it in `~/.ollama/models/` (Linux) or `/Users/<user>/.ollama/models/` (macOS)
- Takes 5–15 minutes depending on internet speed

**Example output:**
```
pulling manifest ⠋
pulling 2665d9e8c9f9... 100% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 4.1 GB
pulling 8c59f6386c2f... 100% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   29 B
pulling 8f685e4dc57d... 100% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   60 B
pulling 15151c2f0666... 100% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  141 B
pulling 5c10c4d589a5... 100% ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 6.8 B
Verifying sha256 digest
writing manifest
success
```

### Option B: Keep Current Model (GPT-OSS)

If you prefer **not** to switch, the pipeline is already configured with `gpt-oss:20b`. Skip to [Step 3](#step-3-run-mfad-inference-with-mistral) and omit the `OLLAMA_MODEL` environment variable.

---

## Step 2: Verify Mistral is Running

### Check Available Models

```bash
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool
```

**Expected output** (if successful pull):
```json
{
    "models": [
        {
            "name": "mistral:latest",
            "model": "mistral:latest",
            ...
        },
        {
            "name": "gpt-oss:20b",
            "model": "gpt-oss:20b",
            ...
        }
    ]
}
```

### Test Ollama Server is Accepting Requests

```bash
curl -X POST http://127.0.0.1:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "Say OK"}],
    "stream": false
  }'
```

**Expected**: JSON response with `"content": "OK"` (or similar).

---

## Step 3: Run MFAD Inference with Mistral

### Full Command (with Mistral)

```bash
cd /usershome/cs671_user10/Group26_DL/MFAD

OLLAMA_MODEL=mistral python3 master_agent.py \
  dataset/DeepFakeDetection/train/fake/fake_000000.jpg \
  --analyst "Your Name"
```

### Example with More Options

```bash
OLLAMA_MODEL=mistral python3 master_agent.py \
  dataset/DeepFakeDetection/train/fake/fake_000150.jpg \
  --analyst "Dev Nagaich"
```

### Alternative: Using GPT-OSS (Current Default)

If you skipped Mistral setup, just run without `OLLAMA_MODEL`:

```bash
python3 master_agent.py \
  dataset/DeepFakeDetection/train/fake/fake_000000.jpg \
  --analyst "Your Name"
```

---

## Step 4: Verify Report Generation

### Monitor the Pipeline

The logs will show each stage:

```
2026-04-15 12:24:21,928 | INFO | master_agent | ============================================================
2026-04-15 12:24:21,928 | INFO | master_agent |   MFAD Pipeline Start
2026-04-15 12:24:21,928 | INFO | master_agent |   case_id    : DFA-2026-TC-750F65C3
2026-04-15 12:24:21,928 | INFO | master_agent |   image_path : dataset/DeepFakeDetection/train/fake/fake_000000.jpg
2026-04-15 12:24:21,928 | INFO | master_agent |   analyst    : Your Name
...
▶ preprocess_node
▶ parallel_analysis_node (all 6 agents in parallel)
  ✓ geometry
  ✓ frequency
  ✓ texture
  ✓ biological
  ✓ metadata
  ✓ vlm
▶ fusion_node (Bayesian log-odds)
▶ report_node (Mistral narrative + PDF generation)
```

### Check Report Output

Once complete, look for:

```bash
ls -lh outputs/DFA-2026-TC-*.pdf
```

**Example output:**
```
-rw-r--r-- 1 cs671_u10 cs671_u10 2.3M Apr 15 12:28 outputs/DFA-2026-TC-750F65C3.pdf
```

### View Generated Files

```bash
# List all outputs for the case
ls -lh outputs/DFA-2026-TC-750F65C3*

# List agent outputs
ls -lh outputs/preprocessing/ outputs/metadata/

# View final decision JSON
cat outputs/DFA-2026-TC-750F65C3.json | python3 -m json.tool | head -40
```

---

## Troubleshooting

### Issue 1: "Cannot connect to Ollama at http://127.0.0.1:11434"

**Solution**: Start Ollama server in another terminal:

```bash
# Option A: If systemd-based
sudo systemctl start ollama

# Option B: Manual start (foreground)
ollama serve

# Option C: Check if already running
ps aux | grep "ollama serve"
```

### Issue 2: "Model 'mistral' not found"

**Solution**: Pull Mistral first:

```bash
ollama pull mistral
```

Wait for completion, then re-run the inference command.

### Issue 3: Out of Memory (OOM) During Report Generation

**Problem**: Mistral-7B may exceed available GPU/RAM.

**Solutions**:
- **Increase swap** (temporary, slow):
  ```bash
  sudo fallocate -l 16G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  ```

- **Use smaller model**:
  ```bash
  ollama pull neural-chat:latest   # ~4.1 GB, faster
  OLLAMA_MODEL=neural-chat python3 master_agent.py ...
  ```

- **Fall back to GPT-OSS** (already cached):
  ```bash
  python3 master_agent.py dataset/DeepFakeDetection/train/fake/fake_000000.jpg
  ```

### Issue 4: Report Node Hangs / Slow Narrative Generation

**Check**: GPU utilization while waiting
```bash
nvidia-smi -l 1   # Update every 1 second
```

**Expected**:
- First Mistral call: ~10–30 seconds (model loading + inference)
- Subsequent calls: ~1–3 seconds (model already in memory)

**If hanging**: Press `Ctrl+C` and check Ollama logs:
```bash
journalctl -u ollama -f   # If systemd
# or in the ollama serve terminal, check for errors
```

---

## Model Options

### Recommended Models for MFAD (in order)

| Model | Size | Speed | Quality | VRAM | Command |
|-------|------|-------|---------|------|---------|
| **neural-chat** | 4.1 GB | Fast | Good | <6 GB | `OLLAMA_MODEL=neural-chat` |
| **mistral** | 4.0 GB | Fast | Excellent | <6 GB | `OLLAMA_MODEL=mistral` |
| **llama2** | 3.8 GB | Fast | Good | <5 GB | `OLLAMA_MODEL=llama2` |
| **gpt-oss** | 13.7 GB | Medium | Very Good | ~12 GB | `OLLAMA_MODEL=gpt-oss:20b` (current) |
| **mixtral** | 26.0 GB | Slow | Excellent | ~20 GB | `OLLAMA_MODEL=mixtral` (requires high-end GPU) |

---

## Quick Reference: Full Workflow

```bash
# 1. Pull Mistral (one-time)
ollama pull mistral

# 2. Verify Ollama is running
curl http://127.0.0.1:11434/api/tags

# 3. Navigate to project
cd /usershome/cs671_user10/Group26_DL/MFAD

# 4. Run inference with Mistral
OLLAMA_MODEL=mistral python3 master_agent.py \
  dataset/DeepFakeDetection/train/fake/fake_000000.jpg \
  --analyst "Your Name"

# 5. Check output
ls -lh outputs/DFA-2026-TC-*.pdf
```

---

## Configuration File (Optional)

To **permanently** set Mistral as default (without environment variable each time):

Edit `config.json`:
```json
{
  "ollama_model": "mistral",
  ...
}
```

Then in `master_agent.py`, line 99, change:
```python
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")  # Changed from "gpt-oss:20b"
```

---

## Further Reading

- [Ollama Documentation](https://github.com/ollama/ollama)
- [Mistral Model Card](https://huggingface.co/mistralai/Mistral-7B)
- [MFAD README](README.md) — Full pipeline architecture
