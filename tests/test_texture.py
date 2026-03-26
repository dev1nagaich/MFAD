#!/usr/bin/env python3
"""
Quick test for texture.py stub and full implementation.
Verifies:
  1. Stub returns correct TextureOutput with hardcoded values
  2. Full implementation runs without errors on a real image
  3. All output fields are present and correctly typed
  4. zone_scores has exactly 5 keys with values in [0, 1]
  5. anomaly_score is bounded [0, 1]
  6. lbp_uniformity is bounded [0, 1]          ← BUG 1 regression guard
  7. zone_scores values are not all 1.0         ← BUG 3 regression guard
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.texture import run_texture_agent_stub, run_texture_agent, TextureOutput


def test_stub():
    """Test that stub returns hardcoded values correctly."""
    print("=" * 60)
    print("TEST 1: Stub Implementation")
    print("=" * 60)

    result = run_texture_agent_stub("dummy.jpg", [0, 0, 100, 100])

    assert isinstance(result, TextureOutput), "Result must be TextureOutput"
    assert result.jaw_emd        == 0.61,  "jaw_emd mismatch"
    assert result.neck_emd       == 0.48,  "neck_emd mismatch"
    assert result.cheek_emd      == 0.22,  "cheek_emd mismatch"
    assert result.lbp_uniformity == 0.31,  "lbp_uniformity mismatch"
    assert result.seam_detected  is True,  "seam_detected should be True"
    assert result.anomaly_score  == 0.895, "anomaly_score mismatch"

    print("✓ Stub returns correct TextureOutput")
    print(f"  jaw_emd:        {result.jaw_emd}")
    print(f"  neck_emd:       {result.neck_emd}")
    print(f"  cheek_emd:      {result.cheek_emd}")
    print(f"  lbp_uniformity: {result.lbp_uniformity}")
    print(f"  seam_detected:  {result.seam_detected}")
    print(f"  anomaly_score:  {result.anomaly_score}")
    print()


def test_zone_scores():
    """Test that zone_scores dict has exactly 5 required keys, all in [0, 1]."""
    print("=" * 60)
    print("TEST 2: Zone Scores Structure")
    print("=" * 60)

    result = run_texture_agent_stub("dummy.jpg", [0, 0, 100, 100])

    required_keys = {"forehead", "cheek_L", "cheek_R", "jaw", "nose"}
    actual_keys   = set(result.zone_scores.keys())

    assert actual_keys == required_keys, (
        f"zone_scores keys mismatch. Expected {required_keys}, got {actual_keys}"
    )

    print("✓ zone_scores has exactly 5 required keys:")
    for key, value in result.zone_scores.items():
        assert 0.0 <= value <= 1.0, f"{key} score out of bounds: {value}"
        print(f"  {key}: {value}")
    print()


def test_anomaly_score_bounds():
    """Test that anomaly_score is in [0, 1]."""
    print("=" * 60)
    print("TEST 3: Anomaly Score Bounds")
    print("=" * 60)

    result = run_texture_agent_stub("dummy.jpg", [0, 0, 100, 100])

    assert 0.0 <= result.anomaly_score <= 1.0, (
        f"anomaly_score out of bounds: {result.anomaly_score}"
    )
    print(f"✓ anomaly_score is within [0, 1]: {result.anomaly_score}")
    print()


# ── BUG 1 regression guard ────────────────────────────────────────────────────
def test_lbp_uniformity_bounds():
    """
    lbp_uniformity must always be in [0, 1].

    The original code used density=True in np.histogram which returns a
    probability *density* — values can easily exceed 1.0 for narrow bins,
    breaking any downstream component that treats it as a probability.
    This test guards against that regression.
    """
    print("=" * 60)
    print("TEST 4: lbp_uniformity bounded [0, 1]  (Bug 1 regression guard)")
    print("=" * 60)

    result = run_texture_agent_stub("dummy.jpg", [0, 0, 100, 100])

    assert 0.0 <= result.lbp_uniformity <= 1.0, (
        f"lbp_uniformity out of bounds: {result.lbp_uniformity}  "
        f"(density=True histogram bug may have been reintroduced)"
    )
    print(f"✓ lbp_uniformity is within [0, 1]: {result.lbp_uniformity}")
    print()


# ── BUG 3 regression guard ────────────────────────────────────────────────────
def test_zone_scores_not_all_maxed():
    """
    zone_scores values must not all be 1.0 on every image.

    The original code used a hardcoded AUTHENTIC_GABOR_MEAN = 0.004 which was
    ~50× smaller than typical image values, causing gabor_dev ≈ 4.0 and every
    zone_score to clip to 1.0.  On a real image all-1.0 zone scores are
    meaningless.  This test runs the full agent on any available test image
    and checks that at least one zone score is < 1.0.
    """
    print("=" * 60)
    print("TEST 5: zone_scores not all 1.0  (Bug 3 regression guard)")
    print("=" * 60)

    test_image = Path(__file__).parent.parent / "test_images" / "suspect.jpg"
    if not test_image.exists():
        print("⚠ No test image found — skipping (run with test_images/suspect.jpg)")
        print()
        return

    result = run_texture_agent(str(test_image), [50, 50, 350, 400])

    all_maxed = all(v >= 1.0 for v in result.zone_scores.values())
    assert not all_maxed, (
        "All zone_scores are 1.0 — relative Gabor deviation may have been "
        "replaced with the hardcoded baseline again."
    )
    print("✓ zone_scores contain at least one non-saturated value:")
    for key, value in result.zone_scores.items():
        print(f"  {key}: {value:.4f}")
    print()


def test_full_implementation():
    """Test full implementation with real test image."""
    print("=" * 60)
    print("TEST 6: Full Implementation with Real Image")
    print("=" * 60)

    test_image = Path(__file__).parent.parent / "test_images" / "suspect.jpg"
    if not test_image.exists():
        print(f"⚠ Test image not found: {test_image} — skipping full test")
        print()
        return

    result = run_texture_agent(str(test_image), [50, 50, 350, 400])

    assert isinstance(result, TextureOutput), "Result must be TextureOutput"
    assert 0.0 <= result.anomaly_score  <= 1.0,  "anomaly_score out of bounds"
    assert 0.0 <= result.lbp_uniformity <= 1.0,  "lbp_uniformity out of bounds"
    assert result.jaw_emd    >= 0.0,              "jaw_emd negative"
    assert result.neck_emd   >= 0.0,              "neck_emd negative"
    assert result.cheek_emd  >= 0.0,              "cheek_emd negative"
    assert set(result.zone_scores.keys()) == {"forehead", "cheek_L", "cheek_R", "jaw", "nose"}

    print("✓ Full implementation runs without errors")
    print(f"  jaw_emd:        {result.jaw_emd:.4f}")
    print(f"  neck_emd:       {result.neck_emd:.4f}")
    print(f"  cheek_emd:      {result.cheek_emd:.4f}")
    print(f"  lbp_uniformity: {result.lbp_uniformity:.4f}")
    print(f"  seam_detected:  {result.seam_detected}")
    print(f"  anomaly_score:  {result.anomaly_score:.4f}")
    print()


def test_pydantic_serialization():
    """Test that TextureOutput can be serialized to dict."""
    print("=" * 60)
    print("TEST 7: Pydantic Serialization")
    print("=" * 60)

    result = run_texture_agent_stub("dummy.jpg", [0, 0, 100, 100])

    result_dict = result.model_dump()
    assert isinstance(result_dict, dict), "model_dump() should return dict"
    assert "anomaly_score" in result_dict, "anomaly_score missing from dict"
    assert "zone_scores"   in result_dict, "zone_scores missing from dict"

    print("✓ TextureOutput serializes correctly to dict")
    print(f"  Keys: {list(result_dict.keys())}")
    print()


if __name__ == "__main__":
    try:
        test_stub()
        test_zone_scores()
        test_anomaly_score_bounds()
        test_lbp_uniformity_bounds()     # Bug 1 regression guard
        test_zone_scores_not_all_maxed() # Bug 3 regression guard
        test_pydantic_serialization()
        test_full_implementation()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)