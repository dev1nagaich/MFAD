#!/usr/bin/env python3
"""
Diagnostic tool to inspect texture feature values for real/fake images.

Shows detailed Gram distances, zone scores, and intermediate calculations
to help calibrate thresholds.

Usage:
    python diagnose_texture.py /path/to/image.jpg [x1 y1 x2 y2]
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from agents.texture import run_texture_agent


def diagnose(image_path: str, face_bbox: list = None):
    """Run texture agent and show detailed diagnostics."""
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        sys.exit(1)
    
    if face_bbox is None:
        # Default bbox for testing
        face_bbox = [50, 50, 350, 400]
    
    print("=" * 80)
    print("Texture Agent Diagnostics")
    print("=" * 80)
    print()
    print(f"Image:  {image_path}")
    print(f"BBox:   {face_bbox}")
    print()
    
    try:
        result = run_texture_agent(image_path, face_bbox)
        output = result.model_dump()
        
        print("-" * 80)
        print("MAIN METRICS")
        print("-" * 80)
        print(f"jaw_emd:           {output['jaw_emd']:.6f}  (cheek↔jaw boundary)")
        print(f"neck_emd:          {output['neck_emd']:.6f}  (jaw↔neck boundary)")
        print(f"cheek_emd:         {output['cheek_emd']:.6f}  (left↔right cheek)")
        print(f"lbp_uniformity:    {output['lbp_uniformity']:.6f}  (higher=more natural)")
        print(f"seam_detected:     {output['seam_detected']}")
        print(f"anomaly_score:     {output['anomaly_score']:.6f}")
        print()
        
        print("-" * 80)
        print("PER-ZONE ANOMALY SCORES")
        print("-" * 80)
        for zone, score in output['zone_scores'].items():
            print(f"  {zone:15s}: {score:.6f}")
        print()
        
        print("-" * 80)
        print("DIAGNOSTIC INFO")
        print("-" * 80)
        print(f"multi_scale_consistency: {output['multi_scale_consistency']:.6f}")
        print()
        
        if output.get('gram_distances'):
            print("Pairwise Gram distances:")
            for pair, dist in output['gram_distances'].items():
                print(f"  {pair:20s}: {dist:.6f}")
            print()
        
        # Interpretation
        print("-" * 80)
        print("INTERPRETATION")
        print("-" * 80)
        
        threshold = 0.65
        boundary_max = max(output['jaw_emd'], output['neck_emd'], output['cheek_emd'])
        print(f"Max boundary distance:  {boundary_max:.6f}")
        print(f"Seam threshold:         {threshold:.6f}")
        print(f"Exceeds threshold?      {'YES' if boundary_max > threshold else 'NO'}")
        print()
        
        mean_zone = sum(output['zone_scores'].values()) / len(output['zone_scores'])
        print(f"Mean zone score:        {mean_zone:.6f}")
        print(f"LBP uniformity:         {output['lbp_uniformity']:.6f}")
        print()
        
        if output['anomaly_score'] < 0.35:
            verdict = "🟢 LIKELY AUTHENTIC"
        elif output['anomaly_score'] > 0.70:
            verdict = "🔴 LIKELY DEEPFAKE"
        else:
            verdict = "🟡 UNCERTAIN"
        
        print(f"VERDICT: {verdict}")
        print(f"  anomaly_score = {output['anomaly_score']:.6f}")
        print()
        
        # Recommendations
        print("-" * 80)
        print("DEBUGGING NOTES")
        print("-" * 80)
        
        if output['anomaly_score'] > 0.70 and "likely authentic" in verdict.lower():
            print("⚠ Score is high but should be low (real image)")
            print("  → Zone scores are high. Check if Gabor extractor is too sensitive")
            print("  → Try increasing zone_score weight reduction")
            print("  → Or adjust per-zone baseline constants")
        
        elif output['anomaly_score'] < 0.35 and "deepfake" in verdict.lower():
            print("⚠ Score is low but should be high (fake image)")
            print("  → Boundary distances are below threshold")
            print("  → Try lowering GRAM_SEAM_THRESHOLD")
            print("  → Or increase boundary_score weight")
        
        else:
            print("✓ Classification seems reasonable")
        
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_texture.py <image_path> [x1 y1 x2 y2]")
        print()
        print("Example:")
        print("  python diagnose_texture.py test_images/real/real_045567.jpg")
        print("  python diagnose_texture.py test_images/fake/fake_001.jpg 60 70 350 400")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if len(sys.argv) >= 6:
        face_bbox = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
    else:
        face_bbox = None
    
    diagnose(image_path, face_bbox)
