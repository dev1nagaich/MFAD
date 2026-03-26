"""
tests/test_report.py — Standalone test for the Report Agent
============================================================
Run from the project root:
    python tests/test_report.py

This loads tests/dummy_ctx.json (simulating all other agents' outputs),
passes it to ReportGenerator.generate(), and produces a PDF in reports/.

No other agents need to be running. No Ollama needed (auto-fallback).
"""

import sys
import json
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
THIS_FILE   = Path(__file__).resolve()
TESTS_DIR   = THIS_FILE.parent
PROJECT_DIR = TESTS_DIR.parent

sys.path.insert(0, str(PROJECT_DIR))

from report_agent.generate import ReportGenerator
from contracts import validate, REPORT_KEYS


def main():
    print("\n" + "=" * 60)
    print("  REPORT AGENT — STANDALONE TEST")
    print("=" * 60 + "\n")

    # ── Step 1: Load dummy ctx ────────────────────────────────
    ctx_path = TESTS_DIR / "dummy_ctx.json"
    if not ctx_path.exists():
        print(f"  ERROR: {ctx_path} not found.")
        sys.exit(1)

    with open(ctx_path, "r") as f:
        ctx = json.load(f)

    # Remove the _note field (not a real agent key)
    ctx.pop("_note", None)

    print(f"  Loaded ctx from {ctx_path}")
    print(f"  ctx has {len(ctx)} keys")
    print(f"  Decision: {ctx.get('decision', '—')}")
    print(f"  Final Score: {ctx.get('final_score', '—')}")

    # ── Step 2: Run ReportGenerator ───────────────────────────
    agent = ReportGenerator()
    print("\n  Running ReportGenerator.generate() ...")

    try:
        result = agent.generate(ctx)
    except Exception as exc:
        print(f"\n  FAILED — ReportGenerator raised: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Step 3: Print results ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  REPORT GENERATOR OUTPUT")
    print("=" * 60)
    for key, val in result.items():
        display = str(val)
        if len(display) > 80:
            display = display[:77] + "..."
        print(f"  {key:30s}: {display}")

    # ── Step 4: Validate contract ─────────────────────────────
    missing = [k for k in REPORT_KEYS if k not in result]
    if missing:
        print(f"\n  FAILED — missing contract keys: {missing}")
        sys.exit(1)

    # ── Step 5: Verify PDF exists ─────────────────────────────
    pdf_path = result["report_path"]
    if not os.path.exists(pdf_path):
        print(f"\n  FAILED — PDF not found at: {pdf_path}")
        sys.exit(1)

    pdf_size = os.path.getsize(pdf_path)
    print(f"\n  PDF file: {pdf_path}")
    print(f"  PDF size: {pdf_size:,} bytes")

    if pdf_size < 1000:
        print("  WARNING — PDF is suspiciously small, may be corrupted.")
    else:
        print("  PDF size looks good.")

    # ── Final result ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✓ PASSED — All 7 REPORT_KEYS present")
    print("  ✓ PASSED — PDF generated successfully")
    print(f"  ✓ Report ID: {result['report_id']}")
    print(f"  ✓ Open the PDF:  xdg-open {pdf_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
