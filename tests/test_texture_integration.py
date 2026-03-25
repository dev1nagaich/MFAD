#!/usr/bin/env python3
"""
Integration test for texture.py with master_agent.py compatibility.
Tests:
  1. texture_agent function (dict return) works with real image
  2. Error handling for invalid paths
  3. Error handling for invalid bboxes
  4. Dict format compatibility with pipeline
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.texture import run_texture_agent_stub, texture_agent, TextureOutput

def test_dict_return_format():
    """Test that texture_agent returns proper dict for pipeline."""
    print("=" * 60)
    print("TEST: texture_agent dict return (pipeline compatible)")
    print("=" * 60)
    
    test_image = Path(__file__).parent.parent / "test_images" / "suspect.jpg"
    if not test_image.exists():
        print(f"⚠ Skipping: test image not found")
        return
    
    result = texture_agent(str(test_image), [50, 50, 350, 400])
    
    # Verify result is a dict
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    
    # Verify all keys present
    required_keys = {"jaw_emd", "neck_emd", "cheek_emd", "lbp_uniformity", "seam_detected", "zone_scores", "anomaly_score"}
    actual_keys = set(result.keys())
    assert required_keys == actual_keys, f"Keys mismatch: expected {required_keys}, got {actual_keys}"
    
    # Verify zone_scores structure
    assert isinstance(result["zone_scores"], dict), "zone_scores must be dict"
    assert set(result["zone_scores"].keys()) == {"forehead", "cheek_L", "cheek_R", "jaw", "nose"}
    
    print(f"✓ Dict format correct")
    print(f"  Result keys: {list(result.keys())}")
    print(f"  zone_scores keys: {list(result['zone_scores'].keys())}")
    print()


def test_error_handling_missing_file():
    """Test error handling for missing image file."""
    print("=" * 60)
    print("TEST: Error handling - missing file")
    print("=" * 60)
    
    try:
        result = texture_agent("/nonexistent/path/image.jpg", [0, 0, 100, 100])
        print("✗ Should have raised FileNotFoundError")
        assert False
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError: {e}")
    print()


def test_error_handling_invalid_bbox():
    """Test error handling for invalid/too-small bbox."""
    print("=" * 60)
    print("TEST: Error handling - invalid bbox")
    print("=" * 60)
    
    test_image = Path(__file__).parent.parent / "test_images" / "suspect.jpg"
    if not test_image.exists():
        print(f"⚠ Skipping: test image not found")
        return
    
    try:
        # Very small bbox that will fail
        result = texture_agent(str(test_image), [0, 0, 10, 10])
        print("✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    print()


def test_repeated_calls():
    """Test that multiple calls work correctly."""
    print("=" * 60)
    print("TEST: Repeated calls")
    print("=" * 60)
    
    test_image = Path(__file__).parent.parent / "test_images" / "suspect.jpg"
    if not test_image.exists():
        print(f"⚠ Skipping: test image not found")
        return
    
    results = []
    for i in range(3):
        result = texture_agent(str(test_image), [50, 50, 350, 400])
        results.append(result)
        assert isinstance(result, dict)
        assert 0.0 <= result["anomaly_score"] <= 1.0
    
    print(f"✓ Made 3 repeated calls successfully")
    print(f"  Anomaly scores: {[r['anomaly_score'] for r in results]}")
    print()


if __name__ == "__main__":
    try:
        test_dict_return_format()
        test_error_handling_missing_file()
        test_error_handling_invalid_bbox()
        test_repeated_calls()
        
        print("=" * 60)
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
