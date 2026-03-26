"""
batch_run.py
============
Tests all images in a folder through the full pipeline.

Usage:
    python batch_run.py                          # default folder: test_images/
    python batch_run.py test_images\             # specify folder
    python batch_run.py test_images\ --verbose   # show per-image signal details

Output:
    outputs/final/<stem>_report.json   <- per-image full report
    outputs/batch_summary.json         <- all results combined
    outputs/batch_summary.csv          <- open in Excel
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from pipeline import run_pipeline

SUPPORTED       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
TEST_IMAGES_DIR = "test_images"

CSV_FIELDS = [
    "image_name",
    "anomaly_score",
    "pre_anomaly_score",
    "meta_anomaly_score",
    "pre_score",
    "meta_score",
    "face_detected",
    "face_count",
    "face_bboxes",
    "face_bbox",
    "image_dims",
    "ela_score",
    "ela_chi2",
    "prnu_score",
    "prnu_absent",
    "thumbnail_mismatch",
    "software_tag",
    "software_flagged",
    "exif_camera_present",
    "exif_camera_make",
    "exif_camera_model",
    "exif_gps_present",
    "exif_datetime_original",
    "hash_sha256",
    "hash_md5",
    "ela_map_path",
    "preprocessing_json",
    "metadata_json",
    "errors",
]


def run_batch(folder: str = TEST_IMAGES_DIR, verbose: bool = False):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"✗  Folder not found: {folder}")
        sys.exit(1)

    images = sorted(
        f for f in folder_path.iterdir()
        if f.suffix.lower() in SUPPORTED
    )
    if not images:
        print(f"No supported images found in '{folder}'")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED))}")
        sys.exit(1)

    Path("outputs/final").mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*72}")
    print(f"  BATCH RUN — {len(images)} image(s) in '{folder}'")
    print(f"{'═'*72}")
    print(f"  {'Image':<32} {'Verdict':<14} {'Score':>6}  "
          f"{'Faces':>5}  {'Time':>6}")
    print(f"  {'─'*32} {'─'*14} {'─'*6}  {'─'*5}  {'─'*6}")

    all_results = []
    start_total = time.time()

    for img_path in images:
        print(f"  {img_path.name:<32}", end=" ", flush=True)
        t0 = time.time()
        try:
            r       = run_pipeline(str(img_path))
            elapsed = time.time() - t0
            score = r.get("anomaly_score", 0) or 0
            icon  = "🔴" if score > 0.5 else "🟢"
            print(f"{icon} anomaly={score:.4f}  "
                  f"faces={r['face_count']:>2}  {elapsed:>5.1f}s")

            if verbose:
                _print_detail(r)

            all_results.append(_flatten(img_path.name, r))

        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗  ERROR  ({elapsed:.1f}s)\n   {e}")
            all_results.append({
                "image_name": img_path.name,
                "anomaly_score": None,
                "errors":     str(e),
            })

    total_time  = time.time() - start_total
    suspicious  = [r for r in all_results if (r.get("anomaly_score") or 0) > 0.5]
    clean       = [r for r in all_results if (r.get("anomaly_score") or 0) <= 0.5 and r.get("anomaly_score") is not None]
    errors_list = [r for r in all_results if r.get("anomaly_score") is None]

    _save_json(all_results)
    _save_csv(all_results)

    print(f"\n{'═'*72}")
    print(f"  SUMMARY")
    print(f"  {'─'*70}")
    print(f"  Total          : {len(images)}")
    print(f"  🔴 Suspicious (>0.5) : {len(suspicious)}")
    print(f"  🟢 Clean (<=0.5)     : {len(clean)}")
    print(f"  ✗  Errors       : {len(errors_list)}")
    print(f"  ⏱  Time         : {total_time:.1f}s  "
          f"({total_time/len(images):.1f}s avg)")
    print(f"  {'─'*70}")
    print(f"  outputs/batch_summary.csv   ← open in Excel")
    print(f"  outputs/batch_summary.json  ← full results")
    print(f"  outputs/final/              ← per-image reports")
    print(f"{'═'*72}\n")

    if suspicious:
        print("  🔴 Suspicious images (anomaly_score > 0.5):")
        for r in sorted(suspicious,
                        key=lambda x: float(x.get("anomaly_score") or 0),
                        reverse=True):
            print(f"     • {r['image_name']:<35}  anomaly={r.get('anomaly_score')}")
        print()

    return all_results


def _flatten(name: str, r: dict) -> dict:
    return {
        "image_name":             name,
        "anomaly_score":          r.get("anomaly_score"),
        "pre_anomaly_score":      r.get("pre_anomaly_score"),
        "meta_anomaly_score":     r.get("meta_anomaly_score"),
        "pre_score":              r.get("pre_score"),
        "meta_score":             r.get("meta_score"),
        "face_detected":          r.get("face_detected"),
        "face_count":             r.get("face_count", 0),
        "face_bboxes":            str(r.get("face_bboxes", [])),
        "face_bbox":              str(r.get("face_bbox", [])),
        "image_dims":             str(r.get("image_dims", [])),
        "ela_score":              r.get("ela_score"),
        "ela_chi2":               r.get("ela_chi2"),
        "prnu_score":             r.get("prnu_score"),
        "prnu_absent":            r.get("prnu_absent"),
        "thumbnail_mismatch":     r.get("thumbnail_mismatch"),
        "software_tag":           r.get("software_tag"),
        "software_flagged":       r.get("software_flagged"),
        "exif_camera_present":    r.get("exif_camera_present"),
        "exif_camera_make":       r.get("exif_camera_make"),
        "exif_camera_model":      r.get("exif_camera_model"),
        "exif_gps_present":       r.get("exif_gps_present"),
        "exif_datetime_original": r.get("exif_datetime_original"),
        "hash_sha256":            r.get("hash_sha256"),
        "hash_md5":               r.get("hash_md5"),
        "ela_map_path":           r.get("ela_map_path"),
        "preprocessing_json":     r.get("preprocessing_json"),
        "metadata_json":          r.get("metadata_json"),
        "errors":                 "; ".join(r.get("errors", [])),
    }


def _print_detail(r: dict):
    print(f"     ├─ ela_score        : {r.get('ela_score')}  →  pre_score={r.get('pre_score')}")
    print(f"     ├─ software_flagged : {r.get('software_flagged')}  ({r.get('software_tag')})")
    print(f"     ├─ thumbnail_miss   : {r.get('thumbnail_mismatch')}")
    print(f"     ├─ prnu_absent      : {r.get('prnu_absent')}  (score={r.get('prnu_score')})")
    print(f"     ├─ camera_present   : {r.get('exif_camera_present')}")
    print(f"     ├─ datetime         : {r.get('exif_datetime_original')}")
    print(f"     ├─ meta_score       : {r.get('meta_score')}")
    print(f"     └─ faces            : {r.get('face_count')}  {r.get('face_bboxes')}")


def _save_json(results: list):
    with open("outputs/batch_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


def _save_csv(results: list):
    with open("outputs/batch_summary.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image forensics runner")
    parser.add_argument("folder", nargs="?", default=TEST_IMAGES_DIR)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_batch(folder=args.folder, verbose=args.verbose)
