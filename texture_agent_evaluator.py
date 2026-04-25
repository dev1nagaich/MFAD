"""
texture_agent_evaluator.py — NPR Texture Agent Evaluation
═══════════════════════════════════════════════════════════════════════════════

Runs the fine-tuned NPR detector on every dataset in `dataset/` and emits a
single JSON report with per-dataset and aggregated metrics:

  Per-dataset (when both classes present):
    accuracy, precision, recall, f1, roc_auc, pr_auc, fpr, fnr, eer

  Per-dataset (single-class):
    n, mean_prob, std_prob, fpr (real-only) or fnr (fake-only)

  Aggregated across all datasets:
    accuracy, precision, recall, f1, roc_auc, pr_auc, fpr, fnr, eer

Usage:
  python texture_agent_evaluator.py \
      --dataset_root /path/to/dataset \
      --weights checkpoints/texture_checkpoint/npr_finetuned.pth \
      --out outputs/texture_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, confusion_matrix,
)
from tqdm import tqdm

from agents.texture_agent import NPRDetector, load_npr_state_dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("texture_eval")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry — every dataset under dataset/
#   layout: "flat"   → images directly under <name>/<split>/
#           "split"  → <name>/<split>/0_real/  AND  <name>/<split>/1_fake/
#           "flat_no_split" → no train/test split, images under <name>/
# ─────────────────────────────────────────────────────────────────────────────
DATASETS: List[Dict] = [
    # FF++ real
    {"name": "original",                    "layout": "flat",         "label": 0},
    # FF++ fakes
    {"name": "Deepfakes",                   "layout": "flat",         "label": 1},
    {"name": "Face2Face",                   "layout": "flat",         "label": 1},
    {"name": "FaceSwap",                    "layout": "flat",         "label": 1},
    {"name": "FaceShifter",                 "layout": "flat",         "label": 1},
    {"name": "NeuralTextures",              "layout": "flat",         "label": 1},
    {"name": "DeepFakeDetection",           "layout": "flat",         "label": 1},
    # Faces-HQ real
    {"name": "Flickr-Faces-HQ_10K",         "layout": "flat",         "label": 0},
    {"name": "celebA-HQ_10K",               "layout": "flat",         "label": 0},
    # Faces-HQ fake
    {"name": "100KFake_10K",                "layout": "flat",         "label": 1},
    {"name": "thispersondoesntexists_10K",  "layout": "flat",         "label": 1},
    # CelebA (large real)
    {"name": "celeba",                      "layout": "flat_no_split", "label": 0},
    # Custom mixed
    {"name": "deepdetect25",                "layout": "split"},
    # Stable Diffusion (fake only)
    {"name": "stable_diffusion_512",        "layout": "flat",         "label": 1},
    {"name": "stable_diffusion_768",        "layout": "flat",         "label": 1},
    {"name": "stable_diffusion_1024",       "layout": "flat",         "label": 1},
    # GAN Zoo (real + fake subdirs)
    {"name": "AttGAN",                      "layout": "split"},
    {"name": "BEGAN",                       "layout": "split"},
    {"name": "CramerGAN",                   "layout": "split"},
    {"name": "InfoMaxGAN",                  "layout": "split"},
    {"name": "MMDGAN",                      "layout": "split"},
    {"name": "RelGAN",                      "layout": "split"},
    {"name": "SNGAN",                       "layout": "split"},
    {"name": "STGAN",                       "layout": "split"},
    {"name": "stargan",                     "layout": "split"},
    {"name": "progan",                      "layout": "split"},
    # Extras (real + fake)
    {"name": "whichfaceisreal",             "layout": "split"},
    {"name": "deepfake",                    "layout": "split"},
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def gather_dataset(dataset_root: Path, cfg: Dict, split: str
                   ) -> Tuple[List[Path], List[int]]:
    name   = cfg["name"]
    layout = cfg["layout"]
    ds     = dataset_root / name
    paths: List[Path] = []
    labels: List[int] = []

    if not ds.exists():
        return paths, labels

    if layout == "flat":
        sub = ds / split
        if not sub.exists():
            sub = ds
        files = collect(sub)
        paths.extend(files)
        labels.extend([int(cfg["label"])] * len(files))
    elif layout == "flat_no_split":
        files = collect(ds)
        paths.extend(files)
        labels.extend([int(cfg["label"])] * len(files))
    elif layout == "split":
        for cls_dir, cls_label in [("0_real", 0), ("1_fake", 1)]:
            sub = ds / split / cls_dir
            if not sub.exists():
                sub = ds / cls_dir
            if not sub.exists():
                continue
            files = collect(sub)
            paths.extend(files)
            labels.extend([cls_label] * len(files))

    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# Inference dataset
# ─────────────────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, paths: List[Path], labels: List[int]):
        self.paths = paths
        self.labels = labels
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            self.NORMALIZE,
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.tf(img), int(self.labels[idx]), 1
        except Exception:
            blank = Image.new("RGB", (256, 256))
            return self.tf(blank), int(self.labels[idx]), 0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def metrics_both_classes(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    return {
        "n":         int(len(y_true)),
        "n_real":    int((y_true == 0).sum()),
        "n_fake":    int((y_true == 1).sum()),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_score)),
        "pr_auc":    float(average_precision_score(y_true, y_score)),
        "fpr":       float(fpr),
        "fnr":       float(fnr),
        "eer":       compute_eer(y_true, y_score),
        "threshold": float(threshold),
    }


def metrics_single_class(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict:
    label = int(y_true[0])
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n":          int(len(y_true)),
        "label":      "real" if label == 0 else "fake",
        "mean_prob":  float(np.mean(y_score)),
        "std_prob":   float(np.std(y_score)),
        "median_prob": float(np.median(y_score)),
        "threshold":  float(threshold),
    }
    if label == 0:
        out["fpr"] = float(np.mean(y_pred == 1))
    else:
        out["fnr"] = float(np.mean(y_pred == 0))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_dataset(model, paths: List[Path], labels: List[int],
                  device: torch.device, batch_size: int, num_workers: int
                  ) -> Tuple[np.ndarray, np.ndarray]:
    ds = EvalDataset(paths, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    all_p = []
    all_y = []
    for x, y, ok in tqdm(loader, ncols=100, leave=False):
        x = x.to(device, non_blocking=True)
        prob = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
        ok = ok.numpy().astype(bool)
        all_p.append(prob[ok])
        all_y.append(y.numpy()[ok])
    if not all_p:
        return np.array([]), np.array([])
    return np.concatenate(all_p), np.concatenate(all_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--weights",      required=True, type=Path)
    ap.add_argument("--split",        default="test", choices=["train", "test"])
    ap.add_argument("--batch_size",   type=int, default=128)
    ap.add_argument("--num_workers",  type=int, default=8)
    ap.add_argument("--threshold",    type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=Path("outputs/texture_eval_report.json"))
    args = ap.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model = NPRDetector().to(device)
    load_npr_state_dict(model, str(args.weights))
    model.eval()
    log.info("Loaded weights from %s", args.weights)

    per_dataset: Dict[str, Dict] = {}
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    t0 = time.time()

    for cfg in DATASETS:
        name = cfg["name"]
        paths, labels = gather_dataset(args.dataset_root, cfg, args.split)
        if not paths:
            log.warning("[skip] %s — no images found", name)
            per_dataset[name] = {"n": 0, "skipped": True}
            continue

        log.info("[%s] %d images", name, len(paths))
        probs, y = score_dataset(model, paths, labels, device,
                                 args.batch_size, args.num_workers)
        if probs.size == 0:
            per_dataset[name] = {"n": 0, "skipped": True}
            continue

        all_probs.append(probs)
        all_labels.append(y)

        unique_classes = np.unique(y)
        if len(unique_classes) == 2:
            stats = metrics_both_classes(y, probs, args.threshold)
        else:
            stats = metrics_single_class(y, probs, args.threshold)
        per_dataset[name] = stats
        log.info("  → %s", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                        for k, v in stats.items()}))

    aggregated: Optional[Dict] = None
    if all_probs:
        all_p = np.concatenate(all_probs)
        all_y = np.concatenate(all_labels)
        if len(np.unique(all_y)) == 2:
            aggregated = metrics_both_classes(all_y, all_p, args.threshold)
        else:
            aggregated = metrics_single_class(all_y, all_p, args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "weights":      str(args.weights),
        "dataset_root": str(args.dataset_root),
        "split":        args.split,
        "threshold":    args.threshold,
        "elapsed_seconds": round(time.time() - t0, 2),
        "per_dataset":  per_dataset,
        "aggregated":   aggregated,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote report → %s", args.out)
    if aggregated:
        log.info("Aggregated: %s", json.dumps(
            {k: round(v, 4) if isinstance(v, float) else v
             for k, v in aggregated.items()}))


if __name__ == "__main__":
    main()
