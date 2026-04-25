#!/usr/bin/env python
"""
train_texture.py — Standalone trainer for the MFAD texture agent.
═════════════════════════════════════════════════════════════════════════════

Strategy (texture-agent specific). The texture agent's UNIQUE signal in MFAD
is the **face-swap boundary seam** at jaw↔neck: a sharp SSIM dissimilarity and
EMD jump where the swapped face was blended onto the original frame. Other
agents own the rest of the texture-adjacent failure modes:

  • Frequency agent  → GAN spectral fingerprints / upsampling grids
                       (100KFake, TPDNE, GAN-zoo)
  • VLM agent        → diffusion semantic anomalies (Stable Diffusion family)
  • Biological agent → mouth-only reenactments (Face2Face, NeuralTextures)
  • Geometry agent   → attribute edits (AttGAN, STGAN, …)

Bayesian fusion downstream lets each agent specialise. So texture trains ONLY
on datasets where the boundary-seam signature is the dominant one:

  Real anchors (label = 0)
    • original              (FF++ real video frames — direct pair to FF++ fakes)
    • Flickr-Faces-HQ_10K   (FFHQ — high-quality real anchor for generalisation)
    • celebA-HQ_10K         (high-quality real anchor)

  Fake anchors (label = 1)
    • Deepfakes             (FF++ autoencoder face-swap — strong jaw seam)
    • FaceSwap              (FF++ CG face-swap — sharp boundary)
    • FaceShifter           (FF++ GAN face-swap — softer but visible seam)

  Skipped on purpose
    • 100KFake / TPDNE / GAN-zoo   → frequency agent's domain
    • Stable Diffusion family      → VLM agent's domain
    • Face2Face / NeuralTextures   → biological agent's domain
    • AttGAN / STGAN / etc         → geometry agent's domain

Output:
    checkpoints/texture_checkpoint/texture_rf.pkl   {classifier, scaler}
    checkpoints/texture_checkpoint/metrics.json

Usage:
    python train_texture.py                    # default 3000/2000 per source
    python train_texture.py --per-source-real 5000 --per-source-fake 3000
    python train_texture.py --no-face-detect   # treat whole image as face
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Make `agents/` importable when running from project root.
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

DATASET_ROOT = PROJECT_DIR / "dataset"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "texture_checkpoint"

REAL_SOURCES = {
    "FFpp_orig": "original",                # ← primary pair to FF++ fakes
    "FFHQ":      "Flickr-Faces-HQ_10K",     # ← high-quality real generaliser
    "celebAHQ":  "celebA-HQ_10K",           # ← high-quality real generaliser
}
FAKE_SOURCES = {
    "Deepfakes":   "Deepfakes",             # ← FF++ autoencoder face-swap
    "FaceSwap":    "FaceSwap",              # ← FF++ CG face-swap
    "FaceShifter": "FaceShifter",           # ← FF++ GAN face-swap
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# ─────────────────────────────────────────────────────────────────────────────
# Path collection
# ─────────────────────────────────────────────────────────────────────────────
def collect_image_paths(folder: Path, split: str, limit: int | None) -> list[Path]:
    """Walk {folder}/{split} recursively and return up to `limit` image paths.

    The repo's standard layout is `<folder>/{train,test}/<images-or-subdirs>`.
    """
    base = folder / split
    if not base.exists():
        return []
    found: list[Path] = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                found.append(Path(root) / f)
    found.sort()
    if limit and len(found) > limit:
        rng = random.Random(42 + (hash(folder.name) & 0xFFFF))
        rng.shuffle(found)
        found = found[:limit]
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Worker process state — initialised once per process via ProcessPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────
_AGENT = None
_FACE_DETECTOR = None
_USE_FACE_DETECT = False


def _worker_init(use_face_detect: bool, project_dir: str) -> None:
    """One-shot init: load the texture agent (and optional face detector)."""
    sys.path.insert(0, project_dir)
    global _AGENT, _FACE_DETECTOR, _USE_FACE_DETECT
    from agents.texture_agent import TextureAgent
    _AGENT = TextureAgent()
    _USE_FACE_DETECT = use_face_detect
    if use_face_detect:
        try:
            from texture_agent_evaluator import FaceDetector
            _FACE_DETECTOR = FaceDetector(backend="opencv")
        except Exception as e:
            # Fall back to whole-image bbox if the detector cannot be built.
            print(f"  [worker] face detector unavailable ({e!s}); using whole-image bbox.",
                  file=sys.stderr, flush=True)
            _FACE_DETECTOR = None
            _USE_FACE_DETECT = False


def _extract_features_for_image(item: tuple[str, int]) -> tuple[int, np.ndarray | None, str]:
    """Worker function: open image, get bbox, run feature extraction.

    Returns (label, 14-d feature vector or None, source path) so the parent can
    log failures by file.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from agents.texture_agent import BoundingBox

    path_str, label = item
    try:
        img = Image.open(path_str).convert("RGB")
        w, h = img.size
        bbox = None
        if _USE_FACE_DETECT and _FACE_DETECTOR is not None:
            arr_bgr = np.array(img)[:, :, ::-1]
            try:
                bboxes = _FACE_DETECTOR.detect(arr_bgr)
                if bboxes:
                    bbox = bboxes[0]
            except Exception:
                bbox = None
        if bbox is None:
            pad = int(min(w, h) * 0.05)
            bbox = BoundingBox(x1=pad, y1=pad, x2=w - pad, y2=h - pad)

        # Replicate the feature pipeline from TextureAgent.analyze without
        # building the full TextureAnalysisResult.
        img_rgb = np.array(img)
        expanded = bbox.expand(0.05)
        x1 = max(0, min(expanded.x1, w - 1))
        y1 = max(0, min(expanded.y1, h - 1))
        x2 = max(x1 + 1, min(expanded.x2, w))
        y2 = max(y1 + 1, min(expanded.y2, h))
        face_rgb = img_rgb[y1:y2, x1:x2]
        if face_rgb.size < 256:
            return label, None, path_str

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        zone_lbp      = _AGENT._compute_zone_lbp(face_gray)
        emd_scores    = _AGENT._compute_emd_matrix(zone_lbp)
        npr_residuals = _AGENT._compute_npr_residuals(face_gray)
        gabor_var     = _AGENT._compute_gabor_variance(face_gray)
        seam_score, _ = _AGENT._detect_boundary_seams(face_bgr)
        color_deltas  = _AGENT._compute_color_uniformity(face_rgb)

        feat = _AGENT._extract_features(
            emd_scores, npr_residuals, zone_lbp,
            seam_score, color_deltas, gabor_var,
        )
        return label, np.asarray(feat, dtype=np.float32).ravel(), path_str
    except Exception as exc:
        return label, None, f"{path_str} ({exc!s})"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset assembly + parallel feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def gather_features(items, workers, use_face_detect):
    feats: list[np.ndarray] = []
    labels: list[int] = []
    sources: list[str] = []
    failures: list[str] = []
    total = len(items)
    t0 = time.time()
    log_every = max(1, total // 20)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(use_face_detect, str(PROJECT_DIR)),
    ) as pool:
        futures = [pool.submit(_extract_features_for_image, it) for it in items]
        done = 0
        for fut in as_completed(futures):
            label, feat, info = fut.result()
            done += 1
            if feat is None:
                failures.append(info)
            else:
                feats.append(feat)
                labels.append(label)
                sources.append(info)
            if done % log_every == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f"  features: {done}/{total} ({100*done/total:5.1f}%) "
                      f"| {rate:5.1f} img/s | failures={len(failures)}", flush=True)

    if not feats:
        raise RuntimeError("No features extracted — every worker failed.")
    return (
        np.vstack(feats).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        sources,
        failures,
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--per-source-real", type=int, default=3000,
                    help="Cap images per real source (3 sources → ~9k real total).")
    ap.add_argument("--per-source-fake", type=int, default=3000,
                    help="Cap images per fake source (3 sources → ~9k fake total).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-face-detect", action="store_true",
                    help="Skip face detection (use whole image; fastest path for "
                         "pre-cropped FFHQ/100KFake/TPDNE).")
    ap.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "texture_rf.pkl")
    ap.add_argument("--metrics", type=Path, default=CHECKPOINT_DIR / "metrics.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print("MFAD Texture Agent — Trainer")
    print("=" * 72, flush=True)

    # ── 1. Collect paths ───────────────────────────────────────────────────────
    print("\n[1/4] Collecting image paths …", flush=True)
    items: list[tuple[str, int]] = []
    real_counts: dict[str, int] = {}
    for tag, folder_name in REAL_SOURCES.items():
        paths = collect_image_paths(DATASET_ROOT / folder_name, "train", args.per_source_real)
        real_counts[tag] = len(paths)
        items.extend((str(p), 0) for p in paths)
    fake_counts: dict[str, int] = {}
    for tag, folder_name in FAKE_SOURCES.items():
        paths = collect_image_paths(DATASET_ROOT / folder_name, "train", args.per_source_fake)
        fake_counts[tag] = len(paths)
        items.extend((str(p), 1) for p in paths)

    print(f"  real:  {real_counts}  → total {sum(real_counts.values())}")
    print(f"  fake:  {fake_counts}  → total {sum(fake_counts.values())}")
    print(f"  combined: {len(items)} images", flush=True)
    if not items:
        sys.exit("No data found under dataset/ — aborting.")

    random.shuffle(items)

    # ── 2. Feature extraction ─────────────────────────────────────────────────
    print(f"\n[2/4] Extracting features ({args.workers} workers, "
          f"face_detect={'OFF' if args.no_face_detect else 'ON'}) …", flush=True)
    X, y, _, failures = gather_features(items, args.workers, not args.no_face_detect)
    print(f"  collected: {X.shape[0]} vectors of dim {X.shape[1]}, "
          f"failures={len(failures)}", flush=True)

    # ── 3. Train classifier ───────────────────────────────────────────────────
    print("\n[3/4] Training HistGradientBoostingClassifier …", flush=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        class_weight="balanced",
        random_state=args.seed,
    )
    t_train = time.time()
    clf.fit(X_tr_s, y_tr)
    t_train = time.time() - t_train

    proba = clf.predict_proba(X_te_s)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc":   float(roc_auc_score(y_te, proba)),
        "f1":        float(f1_score(y_te, pred)),
        "accuracy":  float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall":    float(recall_score(y_te, pred)),
        "n_train":   int(X_tr.shape[0]),
        "n_test":    int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "real_counts": real_counts,
        "fake_counts": fake_counts,
        "failures":   len(failures),
        "train_seconds": round(t_train, 2),
        "model": "HistGradientBoostingClassifier(max_iter=500, lr=0.05, max_depth=8)",
    }
    print(f"  AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"  F1      : {metrics['f1']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision/Recall: {metrics['precision']:.4f} / {metrics['recall']:.4f}")
    print(f"  trained in {t_train:.1f}s on {X_tr.shape[0]} samples", flush=True)

    # ── 4. Save artefacts ─────────────────────────────────────────────────────
    print(f"\n[4/4] Saving artefacts → {args.out.parent}/", flush=True)
    with open(args.out, "wb") as f:
        pickle.dump({"classifier": clf, "scaler": scaler}, f)
    args.metrics.write_text(json.dumps(metrics, indent=2))
    print(f"  classifier → {args.out}")
    print(f"  metrics    → {args.metrics}")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
