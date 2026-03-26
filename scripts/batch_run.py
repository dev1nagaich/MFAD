# MFAD_dev/scripts/batch_run.py
# Run frequency_agent on all images in a directory and save results as CSV + JSON
import json, sys, os, csv, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.frequency_agent import run, validate_output

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def batch_run(image_dir: str, output_dir: str = "reports"):
    image_dir = Path(image_dir)
    images = sorted([
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ])

    if not images:
        print(f"[ERROR] No images found in {image_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"batch_results_{ts}.csv"
    json_path = Path(output_dir) / f"batch_results_{ts}.json"

    all_results = []
    fields = [
        "filename", "fft_mid_anomaly_db", "fft_high_anomaly_db",
        "anomaly_score",
        "valid", "error"
    ]

    print(f"{'='*60}")
    print(f"Batch Frequency Analysis — {len(images)} images")
    print(f"{'='*60}")

    passed, failed = 0, 0
    start_time = time.time()

    for i, img_path in enumerate(images, 1):
        try:
            result = run({"input_type": "image", "path": str(img_path)})
            valid = validate_output(result)
            row = {
                "filename": img_path.name,
                **result,
                "valid": valid,
                "error": ""
            }
            if valid:
                passed += 1
            status = "FAKE" if result["anomaly_score"] >= 0.5 else "REAL"
            print(f"  [{i:3d}/{len(images)}] {img_path.name:40s}  anomaly={result['anomaly_score']:.4f}  -> {status}")
        except Exception as e:
            failed += 1
            row = {"filename": img_path.name, "valid": False, "error": str(e)}
            for f in fields:
                row.setdefault(f, 0.0)
            print(f"  [{i:3d}/{len(images)}] {img_path.name:40s}  ERROR: {e}")

        all_results.append(row)

    elapsed = time.time() - start_time

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    # Write JSON
    with open(json_path, "w") as f:
        json.dump({"timestamp": ts, "count": len(images),
                   "results": all_results}, f, indent=2, default=str)

    # Summary
    scores = [r["anomaly_score"] for r in all_results if isinstance(r.get("anomaly_score"), float) and r["valid"]]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total images : {len(images)}")
    print(f"  Processed OK : {passed}")
    print(f"  Errors       : {failed}")
    print(f"  Time         : {elapsed:.1f}s ({elapsed/len(images):.2f}s/image)")
    if scores:
        import numpy as np
        scores = np.array(scores)
        print(f"  Anomaly mean : {scores.mean():.4f}")
        print(f"  Anomaly std  : {scores.std():.4f}")
        print(f"  Anomaly min  : {scores.min():.4f}")
        print(f"  Anomaly max  : {scores.max():.4f}")
        flagged = (scores >= 0.5).sum()
        print(f"  Flagged FAKE : {flagged}/{len(scores)} ({100*flagged/len(scores):.1f}%)")
    print(f"\n  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")


if __name__ == "__main__":
    img_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "reports"
    batch_run(img_dir, out_dir)
