"""
tests/test_vlm_standalone.py
============================
Standalone test for the VLM Agent.
Run this WITHOUT needing any other agent to be working.

USAGE:
    python tests/test_vlm_standalone.py

BEFORE RUNNING:
    1. Put a suspect face image at:   test_images/suspect.jpg
    2. Put a cropped face image at:   test_images/face_crop.jpg
       (face_crop.jpg can be the same image as suspect.jpg to start)
    3. Install libraries:
         pip install transformers torch torchvision accelerate
         pip install opencv-python matplotlib Pillow
    4. Run the test. First run downloads ~14 GB LLaVA model.

WHAT THIS TEST DOES:
    - Loads tests/dummy_ctx.json
    - Updates image_path and face_crop_path to real files you provide
    - Passes the ctx to VLMAgent.run()
    - Prints every output key with its value
    - Confirms the output passes contracts.py validation
    - Saves heatmap PNG to temp/

WHAT A PASSING TEST LOOKS LIKE:
    All 10 keys printed
    vlm_verdict: FAKE or REAL or UNCERTAIN (not [STUB] or [ERROR])
    vlm_caption: a real English sentence from LLaVA (not a placeholder)
    anomaly_score: a float between 0.0 and 1.0
    PASSED contract validation
"""

import sys
import json
import logging
import shutil
import argparse
from pathlib import Path

# ── path setup so we can import from the parent mfad/ directory ──────────────
THIS_FILE   = Path(__file__).resolve()
TESTS_DIR   = THIS_FILE.parent          # mfad/tests/
PROJECT_DIR = TESTS_DIR.parent          # mfad/
AGENTS_DIR  = PROJECT_DIR / "agents"

sys.path.insert(0, str(PROJECT_DIR))    # for contracts.py
sys.path.insert(0, str(AGENTS_DIR))     # for vlm.py

# ── logging: show timestamps and level so you can follow the run ─────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CREATE TEST IMAGES IF THEY DON'T EXIST
# If you don't have a suspect face image yet, the script auto-generates a
# minimal synthetic test image (solid colour) so you can verify the pipeline
# wiring works before you have real images.
# The synthetic image won't produce meaningful forensic output but it will
# confirm that all code paths run without crashing.
# ─────────────────────────────────────────────────────────────────────────────

def ensure_test_images():
    """
    Make sure test_images/suspect.jpg and test_images/face_crop.jpg exist.

    If they already exist: use them as-is (real images = real forensic output).
    If they don't exist: create minimal synthetic 224x224 images so the
    test can still run and verify the code works end to end.
    """
    test_dir = PROJECT_DIR / "test_images"
    test_dir.mkdir(exist_ok=True)

    suspect_path = test_dir / "suspect.jpg"
    crop_path    = test_dir / "face_crop.jpg"

    if suspect_path.exists() and crop_path.exists():
        logger.info("Found existing test images — using them for real inference.")
        return str(suspect_path), str(crop_path)

    # Generate synthetic images using PIL (already a dependency)
    logger.warning(
        "No test images found. Creating synthetic 224x224 placeholders.\n"
        "  → Forensic output will be MEANINGLESS with synthetic images.\n"
        "  → Replace test_images/suspect.jpg with a real face photo for real output."
    )
    try:
        from PIL import Image as PILImage
        import numpy as np

        # Create a simple gradient image that roughly looks like a face shape
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr[:, :, 0] = 200   # reddish tint (skin-like)
        arr[:, :, 1] = 160
        arr[:, :, 2] = 140
        # Add a darker circle in the centre to vaguely suggest a face
        for y in range(224):
            for x in range(224):
                if (x - 112) ** 2 + (y - 112) ** 2 < 90 ** 2:
                    arr[y, x] = [180, 140, 120]

        img = PILImage.fromarray(arr)
        img.save(str(suspect_path), "JPEG")
        img.save(str(crop_path), "JPEG")
        logger.info("Synthetic images saved to %s", test_dir)

    except Exception as exc:
        logger.error("Could not create synthetic images: %s", exc)
        raise RuntimeError(
            "Please create test_images/suspect.jpg manually "
            "and re-run this test."
        ) from exc

    return str(suspect_path), str(crop_path)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD DUMMY CTX AND SET REAL IMAGE PATHS
# ─────────────────────────────────────────────────────────────────────────────

def load_ctx(suspect_path: str, crop_path: str) -> dict:
    """
    Load dummy_ctx.json and update image_path / face_crop_path to real files.
    All other keys (geometry scores, EXIF flags, etc.) remain as dummy values.
    """
    dummy_path = TESTS_DIR / "dummy_ctx.json"

    if not dummy_path.exists():
        # Try the outputs directory in case user hasn't moved files yet
        dummy_path = PROJECT_DIR.parent / "dummy_ctx.json"

    if not dummy_path.exists():
        logger.warning(
            "dummy_ctx.json not found. Using minimal inline ctx."
        )
        ctx = {}
    else:
        with open(dummy_path, "r") as f:
            ctx = json.load(f)
        logger.info("Loaded dummy ctx from %s", dummy_path)

    # Override paths with the real files
    ctx["image_path"]     = suspect_path
    ctx["face_crop_path"] = crop_path
    ctx["face_bbox"]      = ctx.get("face_bbox", [0, 0, 224, 224])

    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RUN THE AGENT AND PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict):
    """Print all output keys in a readable format."""
    print("\n" + "=" * 60)
    print("  VLM AGENT OUTPUT")
    print("=" * 60)

    # These are the most important keys — print them prominently
    priority = [
        "vlm_verdict", "vlm_confidence", "anomaly_score",
        "saliency_score", "zone_gan_probability",
    ]
    for key in priority:
        if key in result:
            print(f"  {key:30s}: {result[key]}")

    print()

    # Caption gets its own block — it can be long
    print(f"  {'vlm_caption':30s}:")
    caption = result.get("vlm_caption", "")
    # Wrap long captions for readability
    words = caption.split()
    line = "    "
    for word in words:
        if len(line) + len(word) > 70:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)

    print()

    # Zone lists
    print(f"  {'high_activation_regions':30s}: {result.get('high_activation_regions', [])}")
    print(f"  {'medium_activation_regions':30s}: {result.get('medium_activation_regions', [])}")
    print(f"  {'low_activation_regions':30s}: {result.get('low_activation_regions', [])}")
    print()
    print(f"  {'heatmap_path':30s}: {result.get('heatmap_path', '')}")
    print("=" * 60)


def run_test(image_path: str = None):
    """Full test sequence."""
    print("\n" + "=" * 60)
    print("  VLM AGENT STANDALONE TEST")
    print("=" * 60 + "\n")

    # Step 1: ensure test images exist
    if image_path:
        suspect_path = str(Path(image_path).resolve())
        crop_path = suspect_path
    else:
        suspect_path, crop_path = ensure_test_images()

    logger.info("Using suspect image : %s", suspect_path)
    logger.info("Using face crop     : %s", crop_path)

    # Step 2: build ctx
    ctx = load_ctx(suspect_path, crop_path)
    logger.info("ctx built with %d keys", len(ctx))

    # Step 3: import and run the agent
    logger.info("Importing VLMAgent ...")
    from vlm import VLMAgent  # imports from agents/vlm.py via sys.path above

    agent = VLMAgent()
    logger.info("Running VLMAgent.run() ...")

    try:
        result = agent.run(ctx)
    except Exception as exc:
        logger.error("VLMAgent.run() raised an exception: %s", exc, exc_info=True)
        print("\n FAILED — agent raised an exception. See error above.\n")
        sys.exit(1)

    # Step 4: print results
    print_result(result)

    # Step 5: verify contract
    from contracts import VLM_KEYS
    missing = [k for k in VLM_KEYS if k not in result]
    if missing:
        print(f"\n FAILED — missing contract keys: {missing}\n")
        sys.exit(1)

    # Step 6: check no STUB or ERROR values leaked through
    caption = result.get("vlm_caption", "")
    if "[STUB]" in caption:
        print("\n WARNING — caption contains [STUB]. LLaVA may not have run.\n")
    elif "[ERROR]" in caption:
        print("\n WARNING — caption contains [ERROR]. Check logs above.\n")
    elif "[VLM FALLBACK]" in caption:
        print("\n WARNING — fallback was triggered. Check image paths.\n")
    else:
        print("\n  Caption looks real — LLaVA produced a genuine response.")

    # Step 7: check anomaly score is a sensible float
    score = result.get("anomaly_score", -1)
    if not (0.0 <= score <= 1.0):
        print(f"\n FAILED — anomaly_score {score} is outside [0, 1]\n")
        sys.exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone test for VLM Agent.")
    parser.add_argument("--image", type=str, help="Path to a single image to test")
    args = parser.parse_args()

    run_test(args.image)