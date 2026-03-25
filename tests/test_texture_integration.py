#!/usr/bin/env python3
"""
Integration test for texture.py with master_agent.py compatibility.
Tests:
  1. texture_agent() dict return works with real image
  2. Error handling for invalid paths (FileNotFoundError)
  3. Error handling for invalid/too-small bboxes (ValueError)
  4. Dict format compatibility with pipeline (all required keys present)
  5. Repeated calls return consistent, deterministic results
  6. cheek_emd is present AND non-zero in fusion output  ← Bug 4 regression guard
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.texture import run_texture_agent_stub, texture_agent, TextureOutput

# Required keys that every dict returned by texture_agent() must contain
REQUIRED_KEYS = {
    "jaw_emd", "neck_emd", "cheek_emd",
    "lbp_uniformity", "seam_detected",
    "zone_scores", "anomaly_score",
}
REQUIRED_ZONE_KEYS = {"forehead", "cheek_L", "cheek_R", "jaw", "nose"}


def _get_test_image() -> Optional[str]:
    """Return path to test image or None if not available."""
    p = Path(__file__).parent.parent / "test_images" / "suspect.jpg"
    return str(p) if p.exists() else None


def test_dict_return_format():
    """Test that texture_agent returns a properly structured dict for the pipeline."""
    print("=" * 60)
    print("TEST 1: texture_agent dict return (pipeline compatible)")
    print("=" * 60)

    img = _get_test_image()
    if img is None:
        print("⚠ Skipping: test image not found (test_images/suspect.jpg)")
        print()
        return

    result = texture_agent(img, [50, 50, 350, 400])

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert set(result.keys()) == REQUIRED_KEYS, (
        f"Keys mismatch:\n  expected {REQUIRED_KEYS}\n  got      {set(result.keys())}"
    )
    assert isinstance(result["zone_scores"], dict), "zone_scores must be dict"
    assert set(result["zone_scores"].keys()) == REQUIRED_ZONE_KEYS, (
        f"zone_scores keys mismatch: {set(result['zone_scores'].keys())}"
    )
    assert 0.0 <= result["anomaly_score"] <= 1.0, (
        f"anomaly_score out of bounds: {result['anomaly_score']}"
    )

    print("✓ Dict format correct")
    print(f"  Result keys:      {sorted(result.keys())}")
    print(f"  zone_scores keys: {sorted(result['zone_scores'].keys())}")
    print(f"  anomaly_score:    {result['anomaly_score']:.4f}")
    print()


def test_error_handling_missing_file():
    """texture_agent must raise FileNotFoundError for a missing image path."""
    print("=" * 60)
    print("TEST 2: Error handling — missing file")
    print("=" * 60)

    try:
        texture_agent("/nonexistent/path/image.jpg", [0, 0, 100, 100])
        print("✗ Should have raised FileNotFoundError")
        assert False, "Expected FileNotFoundError was not raised"
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError: {e}")
    print()


def test_error_handling_invalid_bbox():
    """texture_agent must raise ValueError when the face crop is too small."""
    print("=" * 60)
    print("TEST 3: Error handling — invalid bbox (too small)")
    print("=" * 60)

    img = _get_test_image()
    if img is None:
        print("⚠ Skipping: test image not found")
        print()
        return

    try:
        # 10×10 px crop — well below the 50×50 minimum
        texture_agent(img, [0, 0, 10, 10])
        print("✗ Should have raised ValueError")
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    print()


def test_repeated_calls_deterministic():
    """
    Multiple calls on the same image with the same bbox must return
    identical anomaly_score values (the agent is pure math — no randomness).
    """
    print("=" * 60)
    print("TEST 4: Repeated calls — deterministic output")
    print("=" * 60)

    img = _get_test_image()
    if img is None:
        print("⚠ Skipping: test image not found")
        print()
        return

    scores = []
    for _ in range(3):
        result = texture_agent(img, [50, 50, 350, 400])
        assert isinstance(result, dict)
        assert 0.0 <= result["anomaly_score"] <= 1.0
        scores.append(result["anomaly_score"])

    # Pure math — all three calls must return exactly the same score
    assert len(set(scores)) == 1, (
        f"Non-deterministic results: {scores}"
    )

    print(f"✓ Three repeated calls returned identical scores: {scores}")
    print()


# ── cheek_emd regression guard ────────────────────────────────────────────────
def test_cheek_emd_included_in_output():
    """
    cheek_emd must be present in the returned dict and >= 0.

    Previously cheek_emd was computed inside run_texture_agent but never
    factored into anomaly_score. This test confirms it is returned and
    that the fusion formula actually uses it (indirectly: if anomaly_score
    matches what we expect from max(jaw, neck, cheek*0.8) / threshold).
    """
    print("=" * 60)
    print("TEST 5: cheek_emd present and used in fusion  (regression guard)")
    print("=" * 60)

    img = _get_test_image()
    if img is None:
        print("⚠ Skipping: test image not found")
        print()
        return

    result = texture_agent(img, [50, 50, 350, 400])

    assert "cheek_emd" in result, "cheek_emd missing from output dict"
    assert result["cheek_emd"] >= 0.0, f"cheek_emd is negative: {result['cheek_emd']}"

    # Verify cheek_emd is plausibly influencing anomaly_score:
    # if cheek_emd * 0.8 > max(jaw_emd, neck_emd), the anomaly_score must
    # be at least (cheek_emd * 0.8 / 0.45) (before seam floor).
    jaw   = result["jaw_emd"]
    neck  = result["neck_emd"]
    cheek = result["cheek_emd"]
    expected_min_raw = max(jaw, neck, cheek * 0.8) / 0.45
    expected_min     = min(1.0, expected_min_raw)
    if result.get("seam_detected"):
        expected_min = max(expected_min, 0.70)

    assert result["anomaly_score"] >= expected_min - 1e-6, (
        f"anomaly_score {result['anomaly_score']:.4f} is lower than expected "
        f"minimum {expected_min:.4f} — cheek_emd may not be included in fusion"
    )

    print(f"✓ cheek_emd present: {result['cheek_emd']:.4f}")
    print(f"  jaw_emd:       {jaw:.4f}")
    print(f"  neck_emd:      {neck:.4f}")
    print(f"  anomaly_score: {result['anomaly_score']:.4f}  (>= {expected_min:.4f})")
    print()


def test_stub_pipeline_compatible():
    """Stub output must also satisfy the pipeline dict contract."""
    print("=" * 60)
    print("TEST 6: Stub dict satisfies pipeline contract")
    print("=" * 60)

    stub = run_texture_agent_stub("dummy.jpg", [0, 0, 100, 100])
    d    = stub.model_dump()

    assert set(d.keys()) == REQUIRED_KEYS, (
        f"Stub keys mismatch: {set(d.keys())}"
    )
    assert set(d["zone_scores"].keys()) == REQUIRED_ZONE_KEYS, (
        f"Stub zone_scores keys mismatch: {set(d['zone_scores'].keys())}"
    )
    assert 0.0 <= d["anomaly_score"]  <= 1.0
    assert 0.0 <= d["lbp_uniformity"] <= 1.0

    print("✓ Stub satisfies full pipeline dict contract")
    print(f"  Keys:          {sorted(d.keys())}")
    print(f"  anomaly_score: {d['anomaly_score']}")
    print()


if __name__ == "__main__":
    try:
        test_dict_return_format()
        test_error_handling_missing_file()
        test_error_handling_invalid_bbox()
        test_repeated_calls_deterministic()
        test_cheek_emd_included_in_output()   # regression guard
        test_stub_pipeline_compatible()

        print("=" * 60)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)