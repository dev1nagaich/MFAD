"""
train_texture.py — Fine-tune NPR (CVPR 2024) on texture-relevant subsets.
═══════════════════════════════════════════════════════════════════════════════

Initialises ResNet50 stem from official NPR.pth (ProGAN-trained), then
fine-tunes on FF++ partial-swaps + StyleGAN + attribute-edit GANs.

GPU-only (A6000 target). Saves to checkpoints/texture_checkpoint/npr_finetuned.pth

Usage:
  python train_texture.py --dataset_root /path/to/dataset \
                          --epochs 10 --batch_size 64 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

from agents.texture_agent import NPRDetector, load_npr_state_dict
from dataset_config import (
    TRAIN_REAL_SOURCES, TRAIN_FAKE_SOURCES, collect_class,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("train_texture")


def build_split(dataset_root: Path, split: str) -> Tuple[List[Path], List[int]]:
    """Returns (paths, labels). label 1=fake, 0=real.

    Pulls ONLY the matching class subfolder for each source — no fallback.
    If a class folder is missing the source contributes 0 images and we log
    a warning. This prevents silent class contamination.
    """
    paths: List[Path] = []
    labels: List[int] = []

    log.info("REAL [%s]:", split)
    for name in TRAIN_REAL_SOURCES:
        files = collect_class(dataset_root, name, split, "real")
        if not files:
            log.warning("  %-30s real %-5s → 0 images (path missing!)", name, split)
        else:
            log.info("  %-30s real %-5s → %d images", name, split, len(files))
        paths.extend(files)
        labels.extend([0] * len(files))

    log.info("FAKE [%s]:", split)
    for name in TRAIN_FAKE_SOURCES:
        files = collect_class(dataset_root, name, split, "fake")
        if not files:
            log.warning("  %-30s fake %-5s → 0 images (path missing!)", name, split)
        else:
            log.info("  %-30s fake %-5s → %d images", name, split, len(files))
        paths.extend(files)
        labels.extend([1] * len(files))

    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class NPRDataset(Dataset):
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, paths: List[Path], labels: List[int], train: bool):
        self.paths = paths
        self.labels = labels
        if train:
            self.tf = transforms.Compose([
                transforms.Resize(288),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.NORMALIZE,
            ])
        else:
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
            return self.tf(img), float(self.labels[idx])
        except Exception as e:
            log.warning("Failed to load %s: %s", self.paths[idx], e)
            blank = Image.new("RGB", (256, 256))
            return self.tf(blank), float(self.labels[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val loops
# ─────────────────────────────────────────────────────────────────────────────

def atomic_save(obj, path: Path) -> None:
    """Write to <path>.tmp then rename. Survives mid-write crashes — the
    existing file is never half-overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load_resume(path: Path):
    """Load a resume checkpoint. Returns None if missing/corrupt."""
    if not path.exists() or path.stat().st_size < 1024:
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        log.warning("resume checkpoint %s unreadable (%s) — starting fresh", path, e)
        return None


def run_epoch(model, loader, optimizer, criterion, device, train: bool, desc: str):
    model.train(train)
    total_loss = 0.0
    all_y = []
    all_p = []
    pbar = tqdm(loader, desc=desc, ncols=100)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().unsqueeze(1)

        with torch.set_grad_enabled(train):
            logit = model(x)
            loss = criterion(logit, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(torch.sigmoid(logit).detach().cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    y_arr = np.concatenate(all_y).ravel()
    p_arr = np.concatenate(all_p).ravel()
    avg_loss = total_loss / max(1, len(loader.dataset))
    auc = roc_auc_score(y_arr, p_arr) if len(np.unique(y_arr)) > 1 else float("nan")
    f1 = f1_score(y_arr, (p_arr >= 0.5).astype(int), zero_division=0)
    acc = accuracy_score(y_arr, (p_arr >= 0.5).astype(int))
    return avg_loss, auc, f1, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True, type=Path)
    ap.add_argument("--init_weights", type=Path,
                    default=Path("checkpoints/texture_checkpoint/NPR.pth"))
    ap.add_argument("--out_weights", type=Path,
                    default=Path("checkpoints/texture_checkpoint/npr_finetuned.pth"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA device index (set CUDA_VISIBLE_DEVICES externally if you prefer).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Training is GPU-only. CUDA not available.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    log.info("Using device cuda:%d (%s)", args.gpu, torch.cuda.get_device_name(args.gpu))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log.info("Building train split from %s", args.dataset_root)
    paths, labels = build_split(args.dataset_root, "train")
    if not paths:
        raise RuntimeError("No training images found. Check --dataset_root.")

    paths = np.array(paths)
    labels = np.array(labels)
    rng = np.random.default_rng(args.seed)
    val_idx = []
    for c in (0, 1):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_val = int(len(idx_c) * args.val_frac)
        val_idx.extend(idx_c[:n_val].tolist())
    val_set = set(val_idx)
    train_idx = [i for i in range(len(paths)) if i not in val_set]
    val_idx = sorted(val_set)

    log.info("train=%d  val=%d  (real %d / fake %d in train)",
             len(train_idx), len(val_idx),
             int((labels[train_idx] == 0).sum()),
             int((labels[train_idx] == 1).sum()))

    train_labels = labels[train_idx]
    train_ds = NPRDataset(paths[train_idx].tolist(), train_labels.tolist(), train=True)
    val_ds   = NPRDataset(paths[val_idx].tolist(),   labels[val_idx].tolist(),   train=False)

    # ── Class-balanced sampler — 1:1 real:fake per batch ─────────────────────
    n_real = int((train_labels == 0).sum())
    n_fake = int((train_labels == 1).sum())
    class_w = {0: 1.0 / max(1, n_real), 1: 1.0 / max(1, n_fake)}
    sample_w = np.array([class_w[int(c)] for c in train_labels], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_w),
        num_samples=len(sample_w),
        replacement=True,
    )
    log.info("Class-balanced sampler enabled (real %d, fake %d → 1:1 batches)",
             n_real, n_fake)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = NPRDetector().to(device)
    if args.init_weights.exists():
        log.info("Loading init weights from %s", args.init_weights)
        load_npr_state_dict(model, str(args.init_weights))
    else:
        log.warning("Init weights %s not found — training from scratch.", args.init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_auc = -1.0
    start_epoch = 1

    # ── Resume from last checkpoint if present ───────────────────────────────
    resume_path = args.out_weights.with_name(args.out_weights.stem + ".last.pth")
    state = safe_load_resume(resume_path)
    if state is not None:
        try:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            history     = state.get("history", [])
            best_auc    = state.get("best_auc", -1.0)
            start_epoch = state.get("epoch", 0) + 1
            log.info(
                "▶ Resuming from %s (epoch %d, best_auc=%.4f)",
                resume_path, start_epoch - 1, best_auc,
            )
        except Exception as e:
            log.warning("resume load failed (%s) — starting fresh", e)
            start_epoch = 1
            best_auc = -1.0
            history = []

    if start_epoch > args.epochs:
        log.info("Already trained %d epochs (start_epoch=%d). Nothing to do.",
                 args.epochs, start_epoch)
        return

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc, tr_f1, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, True, f"epoch {epoch} train")
        vl_loss, vl_auc, vl_f1, vl_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, False, f"epoch {epoch} val  ")
        scheduler.step()
        dt = time.time() - t0

        log.info(
            "epoch %d  | train loss=%.4f auc=%.4f f1=%.4f acc=%.4f  "
            "| val loss=%.4f auc=%.4f f1=%.4f acc=%.4f  | %.1fs",
            epoch, tr_loss, tr_auc, tr_f1, tr_acc, vl_loss, vl_auc, vl_f1, vl_acc, dt,
        )
        history.append({
            "epoch": epoch,
            "train": {"loss": tr_loss, "auc": tr_auc, "f1": tr_f1, "acc": tr_acc},
            "val":   {"loss": vl_loss, "auc": vl_auc, "f1": vl_f1, "acc": vl_acc},
            "seconds": dt,
        })

        if vl_auc > best_auc:
            best_auc = vl_auc
            atomic_save(model.state_dict(), args.out_weights)
            log.info("  ✓ saved best (val AUC=%.4f) → %s", best_auc, args.out_weights)

        # Always save resume checkpoint (full state, atomic write)
        atomic_save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
            "history": history,
        }, resume_path)

    history_path = args.out_weights.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump({"best_val_auc": best_auc, "history": history,
                   "args": {k: str(v) for k, v in vars(args).items()}}, f, indent=2)
    log.info("history → %s", history_path)
    log.info("Done. Best val AUC = %.4f", best_auc)


if __name__ == "__main__":
    main()
