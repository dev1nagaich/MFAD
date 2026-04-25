"""
train_efficientnet.py — MFAD EfficientNet-B4 Fine-Tuning Script
================================================================
Fine-tunes EfficientNet-B4 (Noisy Student, pretrained on ImageNet + JFT-300M)
on the MFAD deepfake dataset using gradual unfreezing + early stopping.

PRETRAINED WEIGHTS SOURCE
--------------------------
Model  : tf_efficientnet_b4.ns_jft_in1k  (via timm)
Size   : ~74 MB, downloaded automatically on first run
Origin : Google Noisy Student training (arxiv:1911.04252)
         Trained on ImageNet-1k + 300M unlabeled JFT images via semi-supervised
         learning — strongest publicly available EfficientNet-B4 checkpoint.
         Same variant used in winning DFDC Kaggle solutions.

GRADUAL UNFREEZING SCHEDULE
-----------------------------
Phase 0 │ Epochs 1–2    │ Head only        │ lr = 1e-3
Phase 1 │ Epochs 3–4    │ + blocks.6       │ lr = 5e-4
Phase 2 │ Epochs 5–7    │ + blocks.5       │ lr = 2e-4
Phase 3 │ Epochs 8–10   │ + blocks.4       │ lr = 1e-4
Phase 4 │ Epochs 11+    │ Full model       │ lr = 5e-5

EARLY STOPPING
---------------
Monitors val AUC-ROC. Stops if AUC does not improve by MIN_DELTA
for EARLY_STOP_PATIENCE consecutive epochs. Best weights are
automatically restored from best_model.pt when training stops.
Early stopping state (patience counter, best AUC) is saved in every
checkpoint so resuming correctly continues tracking patience.

DATASET STRUCTURE EXPECTED
---------------------------
The script reads ONLY train/ splits — never touches test/ splits.
NO pre-processing step is needed. Raw images are loaded directly and
face detection / resizing happens on-the-fly inside the DataLoader.

  dataset_root/
  ├── FF++_C23/
  │   ├── original/train/       <- raw video frames (full-body OK)
  │   ├── Deepfakes/train/
  │   ├── Face2Face/train/
  │   ├── FaceSwap/train/
  │   ├── FaceShifter/train/
  │   ├── NeuralTextures/train/
  │   └── DeepFakeDetection/train/
  ├── Faces-HQ/
  │   ├── Flickr-Faces-HQ_10K/train/
  │   ├── celebA-HQ_10K/train/
  │   ├── 100KFake_10K/train/
  │   └── thispersondoesntexists_10K/train/
  ├── CelebA/celeba/train/
  └── deepdetect25/
      ├── real/train/
      └── fake/train/

ON-THE-FLY FACE PROCESSING
----------------------------
Each dataset entry in DATASET_CONFIGS has a "mode" field:
  "detect" — runs MediaPipe face detection + crop with padding.
             Use for full-body video frames (FF++, deepdetect25).
  "resize" — directly resizes to 512x512. Use for already face-cropped
             images (Faces-HQ, CelebA). MediaPipe is skipped entirely
             for these — no risk of silent drops on StyleGAN images.

No intermediate files are written. No disk space wasted.

USAGE
------
  # First run:
  python train_efficientnet.py

  # Override epochs and batch size:
  python train_efficientnet.py --epochs 30 --batch-size 16

  # Resume from a specific checkpoint:
  python train_efficientnet.py --resume checkpoints/vlm_checkpoint/checkpoint_epoch_007.pt

  # Resume and extend beyond original epoch count:
  python train_efficientnet.py --resume checkpoints/vlm_checkpoint/checkpoint_epoch_020.pt --epochs 35

OUTPUTS
--------
  checkpoints/
    checkpoint_epoch_001.pt   <- full state saved after every epoch
    checkpoint_epoch_002.pt
    ...
    best_model.pt             <- best val-AUC checkpoint (overwritten on improvement)

  models/
    efficientnet_b4_mfad_final.pth   <- weights-only export for VLMAgent

  training.log                <- full training log
  checkpoints/training_history.json  <- per-epoch metrics as JSON

INSTALL REQUIREMENTS
---------------------
  pip install timm scikit-learn Pillow tqdm mediapipe opencv-python
"""

import os
import sys
import json
import time
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2
cv2.setNumThreads(0)  # Prevent OpenCV multithreading from deadlocking with PyTorch DataLoader

try:
    import mediapipe as mp
    # Initialize MediaPipe per-worker to avoid multiprocessing deadlocks.
    _MP_DETECTOR = None
except ImportError:
    sys.exit("mediapipe not found. Run: pip install mediapipe")

try:
    import timm
except ImportError:
    sys.exit("timm not found. Run: pip install timm")

try:
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    sys.exit("scikit-learn not found. Run: pip install scikit-learn")


# =============================================================================
# CONFIGURATION — THE ONLY SECTION YOU NEED TO EDIT
# =============================================================================

# !! CHANGE THIS !! -- the folder that directly contains your dataset subfolders.
# Based on your screenshot, this is the "dataset" folder itself, e.g.:
#   DATASET_ROOT = Path("/home/yourname/MFAD/dataset")
# It must be the folder that contains: original/, Deepfakes/, Face2Face/, etc.
DATASET_ROOT = Path("dataset")   # <- CHANGE THIS

# HAS_TRAIN_SUBDIR controls whether images live inside a train/ subfolder.
# True  -> images at: dataset/Deepfakes/train/img.jpg
# False -> images at: dataset/Deepfakes/img.jpg
# Expand one folder in your file explorer to check, then set this.
HAS_TRAIN_SUBDIR = True   # <- change to False if no train/ subfolder exists

# Each entry is a dict with:
#   path  : subfolder name relative to DATASET_ROOT
#   label : 0 = real, 1 = fake
#   mode  : "detect" -> run MediaPipe face detection + crop (full-body frames)
#           "resize"  -> directly resize to 512x512 (already face-cropped images)
#
# Folder names below match your screenshot exactly.
# Missing directories are skipped with a warning -- script does NOT crash.
DATASET_CONFIGS: List[Dict] = [
    # FF++ C23 -- full-body video frames -> face detection needed
    {"path": "original",          "label": 0, "mode": "detect"},  # real YouTube frames
    {"path": "Deepfakes",         "label": 1, "mode": "detect"},  # autoencoder face swap
    {"path": "Face2Face",         "label": 1, "mode": "detect"},  # facial reenactment
    {"path": "FaceSwap",          "label": 1, "mode": "detect"},  # CG face swap
    {"path": "FaceShifter",       "label": 1, "mode": "detect"},  # GAN face swap
    {"path": "NeuralTextures",    "label": 1, "mode": "detect"},  # neural rendering
    {"path": "DeepFakeDetection", "label": 1, "mode": "detect"},  # Google DFD
    # Already face-cropped images -- skip face detection, just resize
    # (StyleGAN images risk silent drops if MediaPipe is run on them)
    {"path": "Flickr-Faces-HQ_10K",        "label": 0, "mode": "resize"},
    {"path": "celebA-HQ_10K",              "label": 0, "mode": "resize"},
    {"path": "100KFake_10K",               "label": 1, "mode": "resize"},
    {"path": "thispersondoesntexists_10K", "label": 1, "mode": "resize"},
    {"path": "celeba",                     "label": 0, "mode": "resize"},
    # deepdetect25 -- confirmed full photographs from sample images -> detect
    # These have separate real/ and fake/ subfolders inside deepdetect25/train
    {"path": "deepdetect25/train/real",    "label": 0, "mode": "detect", "no_train_subdir": True},
    {"path": "deepdetect25/train/fake",    "label": 1, "mode": "detect", "no_train_subdir": True},

    # --- NEWLY ADDED DATASETS ---

    # Stable Diffusion Face Dataset — mode: detect as requested
    {"path": "stable-diffusion-face-dataset_512/train/fake",  "label": 1, "mode": "detect", "no_train_subdir": True},
    {"path": "stable-diffusion-face-dataset_768/train/fake",  "label": 1, "mode": "detect", "no_train_subdir": True},
    {"path": "stable-diffusion-face-dataset_1024/train/fake", "label": 1, "mode": "detect", "no_train_subdir": True},

    # GAN Zoo — face-aligned synthetic outputs, resize is appropriate
    {"path": "AttGAN/train/real",     "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "AttGAN/train/fake",     "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "BEGAN/train/real",      "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "BEGAN/train/fake",      "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "CramerGAN/train/real",  "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "CramerGAN/train/fake",  "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "InfoMaxGAN/train/real", "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "InfoMaxGAN/train/fake", "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "MMDGAN/train/real",     "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "MMDGAN/train/fake",     "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "RelGAN/train/real",     "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "RelGAN/train/fake",     "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "SNGAN/train/real",      "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "SNGAN/train/fake",      "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "STGAN/train/real",      "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "STGAN/train/fake",      "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "progan/train/real",     "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "progan/train/fake",     "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "stargan/train/real",    "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "stargan/train/fake",    "label": 1, "mode": "resize", "no_train_subdir": True},

    # Extra Mixed Datasets
    {"path": "whichfaceisreal/train/real", "label": 0, "mode": "resize", "no_train_subdir": True},
    {"path": "whichfaceisreal/train/fake", "label": 1, "mode": "resize", "no_train_subdir": True},
    {"path": "deepfake/train/real",        "label": 0, "mode": "detect", "no_train_subdir": True},
    {"path": "deepfake/train/fake",        "label": 1, "mode": "detect", "no_train_subdir": True},
]
# Fraction of training images held out for validation (stratified by label)
VAL_SPLIT = 0.10

# Training hyper-parameters
BATCH_SIZE   = 32    # reduce to 16 if you get CUDA out-of-memory errors
NUM_WORKERS  = 8     # DataLoader worker processes; reduce to 4 on slower machines
NUM_EPOCHS   = 25    # maximum epochs; early stopping will usually stop before this
IMAGE_SIZE   = 380   # EfficientNet-B4 native resolution — do NOT change this
SEED         = 42

# Early stopping — monitors val AUC-ROC
EARLY_STOP_PATIENCE = 5      # stop after this many epochs with no improvement
MIN_DELTA           = 0.001  # minimum AUC gain that counts as "improvement"

# Output directories — VLM-side EfficientNet snapshots live in the vlm_checkpoint
# subfolder so the checkpoints/ root stays organised by agent.
CHECKPOINT_DIR   = Path("checkpoints") / "vlm_checkpoint"
MODEL_OUTPUT_DIR = Path("models")

# Gradual unfreezing schedule.
# Key   = first epoch (1-indexed) when this phase begins.
# Value = list of backbone submodule name prefixes to unfreeze at that epoch.
#         "_all_" is a special sentinel meaning "unfreeze the entire backbone".
# The head (self.head) is always trainable from epoch 1.
UNFREEZE_SCHEDULE: Dict[int, List[str]] = {
    1:  [],           # Phase 0: head only, backbone fully frozen
    3:  ["blocks.6"], # Phase 1: unfreeze last MBConv block
    5:  ["blocks.5"], # Phase 2: + second-to-last block
    8:  ["blocks.4"], # Phase 3: + block 4
    11: ["_all_"],    # Phase 4: full model
}

# Learning rate at the start of each phase (matches UNFREEZE_SCHEDULE keys)
LR_SCHEDULE: Dict[int, float] = {
    1:  1e-3,   # Phase 0 — head warm-up
    3:  5e-4,   # Phase 1
    5:  2e-4,   # Phase 2
    8:  1e-4,   # Phase 3
    11: 5e-5,   # Phase 4 — full model, low LR to avoid destroying pretrained features
}


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# DATASET UTILITIES
# =============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_samples(dataset_root: Path) -> List[Tuple[Path, int, str]]:
    """
    Walk DATASET_CONFIGS and collect (image_path, label, mode) triples.

    Respects HAS_TRAIN_SUBDIR:
      True  -> looks in  dataset_root / path / "train"
      False -> looks in  dataset_root / path   (images sit directly here)

    Only collects from the configured subdir — never touches test/ splits.
    Missing directories produce a warning and are skipped.
    """
    samples: List[Tuple[Path, int, str]] = []

    for cfg in DATASET_CONFIGS:
        base_dir  = dataset_root / cfg["path"]
        if cfg.get("no_train_subdir", False):
            scan_dir = base_dir
        else:
            scan_dir  = base_dir / "train" if HAS_TRAIN_SUBDIR else base_dir
        label     = cfg["label"]
        mode      = cfg["mode"]

        if not scan_dir.exists():
            logger.warning("Directory not found (skipping): %s", scan_dir)
            continue

        found = 0
        for img_path in scan_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((img_path, label, mode))
                found += 1

        label_name = "real" if label == 0 else "fake"
    if not samples:
        raise RuntimeError(
            f"No images found under {dataset_root}. "
            f"Check DATASET_ROOT and DATASET_CONFIGS at the top of the script."
        )
    return samples


def stratified_split(
    samples: List[Tuple[Path, int, str]],
    val_fraction: float,
    seed: int,
) -> Tuple[List, List]:
    """
    Stratified train/val split — real:fake ratio is identical in both halves.
    Preserves the (path, label, mode) triple through the split.
    """
    rng = random.Random(seed)

    real_samples = [s for s in samples if s[1] == 0]
    fake_samples = [s for s in samples if s[1] == 1]
    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    def split_list(lst: list, frac: float) -> Tuple[list, list]:
        n_val = max(1, int(len(lst) * frac))
        return lst[n_val:], lst[:n_val]   # returns (train_part, val_part)

    train_real, val_real = split_list(real_samples, val_fraction)
    train_fake, val_fake = split_list(fake_samples, val_fraction)

    train_samples = train_real + train_fake
    val_samples   = val_real   + val_fake
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def get_mp_detector():
    global _MP_DETECTOR
    if _MP_DETECTOR is None:
        _MP_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )
    return _MP_DETECTOR


def _detect_and_crop(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Run MediaPipe face detection on a BGR image.
    Returns a 512x512 face crop (BGR) with 25% padding, or None if no face found.

    Initializes MediaPipe lazily per worker to prevent PyTorch fork() deadlocks.

    Padding of 25% on each side ensures ears, hairline and upper neck are
    included. These regions carry important deepfake blending-seam signals.
    """
    h, w      = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    detector  = get_mp_detector()
    results   = detector.process(image_rgb)

    if not results.detections:
        return None

    # Pick the largest detected face (subject dominates in FF++ frames)
    best_bbox = None
    best_area = 0
    for det in results.detections:
        rb   = det.location_data.relative_bounding_box
        x1   = int(rb.xmin * w)
        y1   = int(rb.ymin * h)
        bw   = int(rb.width  * w)
        bh   = int(rb.height * h)
        area = bw * bh
        if area > best_area:
            best_area = area
            best_bbox = (x1, y1, x1 + bw, y1 + bh)

    if best_bbox is None:
        return None

    x1, y1, x2, y2 = best_bbox
    bw    = x2 - x1
    bh    = y2 - y1
    pad_x = int(bw * 0.25)
    pad_y = int(bh * 0.25)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)


def _resize_only(image_bgr: np.ndarray) -> np.ndarray:
    """
    Resize an already face-cropped image to 512x512.
    Used for Faces-HQ and CelebA — no face detection needed.
    """
    return cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LANCZOS4)


class DeepfakeDataset(Dataset):
    """
    Loads raw images on-the-fly and applies face detection or direct resize
    depending on the mode stored in each sample triple.

    No intermediate files are written to disk. Face detection runs inside
    the DataLoader worker processes in parallel, so it does not block training.

    Each sample is a (Path, label, mode) triple:
      Path  — path to the RAW image file (full-body frame or face crop)
      label — 0=real, 1=fake
      mode  — "detect" (run MediaPipe) or "resize" (direct 512x512 resize)

    If face detection fails on a "detect" image (no face found or corrupt),
    a random replacement sample is returned instead of crashing the epoch.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, int, str]],
        transform,
    ) -> None:
        self.samples   = samples
        self.transform = transform
        # Track detection failures per worker for logging — does not affect training
        self._no_face_count = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label, mode = self.samples[idx]
        try:
            # Read as BGR with OpenCV — faster than PIL for large batches
            image_bgr = cv2.imread(str(path))
            if image_bgr is None:
                raise ValueError(f"cv2.imread returned None for {path}")

            if mode == "detect":
                face_bgr = _detect_and_crop(image_bgr)
                if face_bgr is None:
                    # No face detected — return a random replacement silently
                    self._no_face_count += 1
                    return self.__getitem__(random.randint(0, len(self.samples) - 1))
            else:
                # mode == "resize": already a face image, just standardise size
                face_bgr = _resize_only(image_bgr)

            # Convert BGR -> RGB -> PIL for torchvision transforms
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            img_pil  = Image.fromarray(face_rgb)
            return self.transform(img_pil), label

        except Exception as exc:
            logger.debug(
                "Error on %s mode=%s (%s) — random replacement.", path.name, mode, exc
            )
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def build_transforms(image_size: int, split: str):
    """
    Training: strong augmentations to counter overfitting on near-duplicate FF++ frames.
    Validation: deterministic resize + center-crop only — no augmentation.
    """
    # Must match Noisy Student pretraining statistics exactly
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            # Geometric — deepfakes carry spatial artifacts near face boundaries
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=10),
            # Color — catches color-transfer and compression-shift deepfakes
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            # Blur — simulates JPEG compression artifacts common in real-world fakes
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # Erasing — simulates watermarks / partial occlusions
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # slight oversize then crop
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def make_weighted_sampler(samples: List[Tuple[Path, int, str]]) -> WeightedRandomSampler:
    """
    Over-samples the minority class so every batch sees a balanced distribution.
    Required because the dataset is ~70% real / ~30% fake. Without this,
    the model learns to predict "real" for everything and achieves 70% accuracy
    while being useless as a detector.
    """
    labels        = np.array([s[1] for s in samples])
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(np.float32)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(samples),
        replacement=True,
    )


def build_dataloaders(
    train_samples: List[Tuple[Path, int, str]],
    val_samples:   List[Tuple[Path, int, str]],
    batch_size:    int,
    num_workers:   int,
    device:        torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders."""
    use_pin_memory       = device.type == "cuda"
    use_persistent_workers = num_workers > 0

    train_ds = DeepfakeDataset(train_samples, build_transforms(IMAGE_SIZE, "train"))
    val_ds   = DeepfakeDataset(val_samples,   build_transforms(IMAGE_SIZE, "val"))

    sampler = make_weighted_sampler(train_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        prefetch_factor=2 if use_persistent_workers else None,
        persistent_workers=use_persistent_workers,
        drop_last=True,   # avoids BatchNorm issues on tiny last batches
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,  # no gradient — can fit a larger batch
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return train_loader, val_loader


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Stops training when val AUC fails to improve by MIN_DELTA for
    PATIENCE consecutive epochs.

    State is saved inside every checkpoint so resuming a run correctly
    continues tracking patience — the counter does NOT restart from zero.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0       # consecutive epochs without sufficient improvement
        self.best_auc  = 0.0     # best val AUC seen so far
        self.triggered = False   # set to True once patience is exhausted

    def step(self, val_auc: float) -> bool:
        """
        Call once per epoch with the current val AUC.
        Returns True if training should stop, False otherwise.
        """
        if val_auc >= self.best_auc + self.min_delta:
            # Genuine improvement — reset counter
            self.best_auc = val_auc
            self.counter  = 0
        else:
            self.counter += 1
            logger.info(
                "[EarlyStopping] No improvement for %d/%d epochs "
                "(best=%.4f  current=%.4f  required_delta=%.4f)",
                self.counter, self.patience,
                self.best_auc, val_auc, self.min_delta,
            )
            if self.counter >= self.patience:
                self.triggered = True
                logger.info(
                    "[EarlyStopping] Triggered — %d epochs without improvement. "
                    "Stopping. Best val-AUC=%.4f is in best_model.pt.",
                    self.patience, self.best_auc,
                )
                return True
        return False

    def state_dict(self) -> dict:
        return {
            "counter":   self.counter,
            "best_auc":  self.best_auc,
            "triggered": self.triggered,
        }

    def load_state_dict(self, state: dict) -> None:
        self.counter   = state.get("counter",   0)
        self.best_auc  = state.get("best_auc",  0.0)
        self.triggered = state.get("triggered", False)


# =============================================================================
# MODEL
# =============================================================================

class EfficientNetB4Detector(nn.Module):
    """
    EfficientNet-B4 Noisy Student backbone + custom binary classification head.

    backbone : tf_efficientnet_b4.ns_jft_in1k  (~19M params, ~74MB download)
    head     : Dropout(0.4) -> Linear(1792, 512) -> GELU -> Dropout(0.2) -> Linear(512, 2)

    1792 = EfficientNet-B4 penultimate feature dimension (after global avg pool).
    The two-layer head gives the model capacity to learn deepfake-specific
    decision boundaries from ImageNet features.
    """

    def __init__(self, num_classes: int = 2, drop_rate: float = 0.4) -> None:
        super().__init__()
        # num_classes=0 removes the ImageNet head; forward() returns (batch, 1792)
        self.backbone = timm.create_model(
            "tf_efficientnet_b4.ns_jft_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features  # 1792

        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — training head only.")

    def unfreeze_blocks(self, block_names: List[str]) -> None:
        """
        Unfreeze named backbone submodules.
        Pass ["_all_"] to unfreeze the entire backbone.
        """
        if "_all_" in block_names:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Full backbone unfrozen.")
            return

        for name in block_names:
            matched = False
            for module_name, module in self.backbone.named_modules():
                if module_name.startswith(name):
                    for param in module.parameters():
                        param.requires_grad = True
                    matched = True
            if matched:
                logger.info("Unfrozen backbone block: %s", name)
            else:
                logger.warning(
                    "Block name '%s' not found in backbone — check UNFREEZE_SCHEDULE.", name
                )

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PHASE MANAGEMENT
# =============================================================================

def get_phase_for_epoch(epoch: int) -> int:
    """Return the 0-indexed phase number for a given 1-indexed epoch."""
    sorted_starts = sorted(UNFREEZE_SCHEDULE.keys())
    phase = 0
    for i, start in enumerate(sorted_starts):
        if epoch >= start:
            phase = i
    return phase


def apply_phase_transition(
    epoch: int,
    model: EfficientNetB4Detector,
) -> Tuple[int, float]:
    """
    At the epoch where a phase begins, apply the required freeze/unfreeze.
    Returns (phase_index, target_lr_for_this_phase).
    Called once at the start of every epoch.
    """
    sorted_starts = sorted(UNFREEZE_SCHEDULE.keys())
    phase_idx = get_phase_for_epoch(epoch)

    if epoch in UNFREEZE_SCHEDULE:
        blocks    = UNFREEZE_SCHEDULE[epoch]
        target_lr = LR_SCHEDULE[epoch]

        if epoch == sorted_starts[0]:
            # Very first phase — freeze entire backbone before unfreezing anything
            model.freeze_backbone()

        if blocks:
            model.unfreeze_blocks(blocks)

        logger.info(
            "Epoch %d -> Phase %d | target_lr=%.1e | trainable params: %d",
            epoch, phase_idx, target_lr, model.count_trainable(),
        )
        return phase_idx, target_lr

    # Mid-phase epoch — no structural change; return the LR of the current phase
    current_phase_start = sorted_starts[phase_idx]
    return phase_idx, LR_SCHEDULE[current_phase_start]


def build_optimizer(model: EfficientNetB4Detector, lr: float) -> optim.AdamW:
    """
    Build a fresh AdamW over all currently trainable parameters.
    Called at every phase transition so newly unfrozen parameters are included
    with their own fresh gradient history (moment buffers start at zero).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check that freeze_backbone() was not called before the head was added."
        )
    return optim.AdamW(trainable, lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))


# =============================================================================
# CHECKPOINT SAVE / LOAD
# =============================================================================

def save_checkpoint(
    path:           Path,
    epoch:          int,
    model:          EfficientNetB4Detector,
    optimizer:      optim.Optimizer,
    scheduler,
    best_auc:       float,
    phase:          int,
    early_stopping: EarlyStopping,
) -> None:
    """
    Save complete training state.
    Includes early stopping counter so patience survives a restart.
    """
    torch.save(
        {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_auc":        best_auc,
            "phase":           phase,
            "early_stopping":  early_stopping.state_dict(),
            "timm_model_id":   "tf_efficientnet_b4.ns_jft_in1k",
        },
        path,
    )
    logger.info("Checkpoint saved -> %s", path)


def load_checkpoint(
    path:           Path,
    model:          EfficientNetB4Detector,
    early_stopping: EarlyStopping,
) -> Tuple[int, float, int, Optional[dict], Optional[dict]]:
    """
    Load training state from a checkpoint file.
    Returns (start_epoch, best_auc, phase, optimizer_state_dict, scheduler_state_dict).

    The optimizer is NOT rebuilt here — the caller rebuilds it after applying
    the correct freeze/unfreeze state (so param count matches), then loads
    the returned optimizer_state_dict.
    """
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Restore model weights first
    model.load_state_dict(ckpt["model_state"])

    # Restore early stopping state
    if "early_stopping" in ckpt:
        early_stopping.load_state_dict(ckpt["early_stopping"])
        logger.info(
            "Early stopping restored: counter=%d  best_auc=%.4f",
            early_stopping.counter, early_stopping.best_auc,
        )

    start_epoch = ckpt["epoch"] + 1          # resume from NEXT epoch
    best_auc    = ckpt.get("best_auc",  0.0)
    phase       = ckpt.get("phase",     0)
    opt_state   = ckpt.get("optimizer_state", None)
    sched_state = ckpt.get("scheduler_state", None)

    logger.info(
        "Checkpoint loaded: epoch=%d  best_auc=%.4f  phase=%d",
        ckpt["epoch"], best_auc, phase,
    )
    return start_epoch, best_auc, phase, opt_state, sched_state


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
    scaler:    torch.cuda.amp.GradScaler,
) -> Dict[str, float]:

    model.train()
    total_loss = 0.0
    all_preds:  List[float] = []
    all_labels: List[int]   = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        # Gradient clipping prevents instability when a new block is first unfrozen
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    try:
        auc = float(roc_auc_score(all_labels, all_preds))
    except Exception:
        auc = 0.0
    acc = float(accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int)))
    return {"loss": avg_loss, "auc": auc, "accuracy": acc}


@torch.no_grad()
def run_validation(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch:     int,
) -> Dict[str, float]:

    model.eval()
    total_loss = 0.0
    all_preds:  List[float] = []
    all_labels: List[int]   = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [ val ]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_preds.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    try:
        auc = float(roc_auc_score(all_labels, all_preds))
    except Exception:
        auc = 0.0
    acc = float(accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int)))
    return {"loss": avg_loss, "auc": auc, "accuracy": acc}


# =============================================================================
# MAIN
# =============================================================================

def main(resume_path: Optional[str], num_epochs: int, batch_size: int) -> None:
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s | VRAM: %.1f GB",
            torch.cuda.get_device_name(),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Dataset
    logger.info("Scanning dataset: %s", DATASET_ROOT)
    all_samples = collect_samples(DATASET_ROOT)
    n_real = sum(1 for s in all_samples if s[1] == 0)
    n_fake = sum(1 for s in all_samples if s[1] == 1)
    logger.info(
        "Total %d images | real=%d  fake=%d  ratio=1:%.2f",
        len(all_samples), n_real, n_fake, n_fake / max(n_real, 1),
    )

    train_samples, val_samples = stratified_split(all_samples, VAL_SPLIT, SEED)
    logger.info("Split: train=%d  val=%d", len(train_samples), len(val_samples))

    train_loader, val_loader = build_dataloaders(
        train_samples, val_samples, batch_size, NUM_WORKERS, device
    )

    # Model
    logger.info("Building EfficientNet-B4 Noisy Student detector ...")
    model = EfficientNetB4Detector(num_classes=2, drop_rate=0.4).to(device)
    logger.info(
        "Total params: %.2fM | Trainable (before phase 0): %.2fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
        model.count_trainable() / 1e6,
    )

    # Loss: CrossEntropy with label_smoothing=0.1 to prevent overconfident
    # predictions on near-duplicate FF++ frames.
    # (Not ArcFace/CosFace — those are for open-set identity verification, not
    #  binary classification.)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Mixed precision GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=MIN_DELTA)

    # State
    start_epoch   = 1
    best_auc      = 0.0
    current_phase = -1   # -1 forces Phase 0 setup on epoch 1
    optimizer:   Optional[optim.AdamW]                   = None
    scheduler:   Optional[optim.lr_scheduler.CosineAnnealingLR] = None
    history:     List[dict] = []

    # Resume
    if resume_path:
        start_epoch, best_auc, saved_phase, opt_state, sched_state = load_checkpoint(
            Path(resume_path), model, early_stopping
        )

        # Replay all phase transitions up to saved_phase so that the correct
        # backbone blocks are frozen/unfrozen before we rebuild the optimizer.
        # This ensures the param count in the optimizer matches what was saved.
        sorted_starts = sorted(UNFREEZE_SCHEDULE.keys())
        for i, phase_start in enumerate(sorted_starts):
            if i > saved_phase:
                break
            blocks = UNFREEZE_SCHEDULE[phase_start]
            if phase_start == sorted_starts[0]:
                model.freeze_backbone()
            if blocks:
                model.unfreeze_blocks(blocks)

        # Build optimizer with correct param set, then restore saved moments
        phase_lr  = LR_SCHEDULE[sorted_starts[saved_phase]]
        optimizer = build_optimizer(model, phase_lr)
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as e:
                logger.warning(
                    "Could not load optimizer state (%s). "
                    "Optimizer starts fresh (this is safe).", e
                )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(num_epochs - start_epoch + 1, 1),
            eta_min=1e-6,
        )
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
            except Exception as e:
                logger.warning("Could not load scheduler state (%s). Fresh scheduler.", e)

        current_phase = saved_phase
        logger.info(
            "Ready to resume: epoch=%d  phase=%d  es_counter=%d/%d",
            start_epoch, saved_phase,
            early_stopping.counter, early_stopping.patience,
        )

    # Training loop
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Phase transition
        phase_idx, target_lr = apply_phase_transition(epoch, model)

        if phase_idx != current_phase:
            # New phase — rebuild optimizer to include newly unfrozen params
            current_phase = phase_idx
            optimizer = build_optimizer(model, target_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - epoch + 1, 1),
                eta_min=1e-6,
            )
            logger.info(
                "Optimizer rebuilt: Phase %d  lr=%.1e  trainable=%d params",
                phase_idx, target_lr, model.count_trainable(),
            )
        elif optimizer is None:
            # First epoch of a fresh (non-resumed) run
            optimizer = build_optimizer(model, target_lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - epoch + 1, 1),
                eta_min=1e-6,
            )

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )

        # Validate
        val_metrics = run_validation(model, val_loader, criterion, device, epoch)

        # Scheduler step
        scheduler.step()

        elapsed    = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %03d/%03d  Phase=%d  lr=%.2e  "
            "train[loss=%.4f auc=%.4f acc=%.4f]  "
            "val[loss=%.4f auc=%.4f acc=%.4f]  "
            "time=%.1fs",
            epoch, num_epochs, phase_idx, current_lr,
            train_metrics["loss"], train_metrics["auc"], train_metrics["accuracy"],
            val_metrics["loss"],   val_metrics["auc"],   val_metrics["accuracy"],
            elapsed,
        )

        # Save epoch checkpoint (every epoch — allows resume from any point)
        ckpt_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt_path, epoch, model, optimizer, scheduler,
            best_auc, phase_idx, early_stopping,
        )

        # Save best model if val AUC improved beyond the minimum delta
        val_auc = val_metrics["auc"]
        if val_auc >= best_auc + MIN_DELTA:
            best_auc = val_auc
            save_checkpoint(
                CHECKPOINT_DIR / "best_model.pt",
                epoch, model, optimizer, scheduler,
                best_auc, phase_idx, early_stopping,
            )
            logger.info("* New best val-AUC=%.4f -> best_model.pt updated", best_auc)

        # History
        history.append({
            "epoch":      epoch,
            "phase":      phase_idx,
            "lr":         current_lr,
            "train":      train_metrics,
            "val":        val_metrics,
            "best_auc":   best_auc,
            "es_counter": early_stopping.counter,
        })

        # Early stopping — checked AFTER saving so the epoch is always persisted
        if early_stopping.step(val_auc):
            logger.info(
                "Early stopping at epoch %d. "
                "Best val-AUC=%.4f is stored in best_model.pt.",
                epoch, best_auc,
            )
            break

    # Save training history
    history_path = CHECKPOINT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Training history -> %s", history_path)

    # Export best weights for VLMAgent (weights only, no optimizer state)
    best_ckpt = CHECKPOINT_DIR / "best_model.pt"
    if best_ckpt.exists():
        logger.info("Loading best_model.pt for final export ...")
        best_state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(best_state["model_state"])
    else:
        logger.warning("best_model.pt not found — exporting current weights instead.")

    final_path = MODEL_OUTPUT_DIR / "efficientnet_b4_mfad_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Final weights -> %s", final_path)
    logger.info("Training complete. Best val-AUC = %.4f", best_auc)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B4 for MFAD deepfake detection"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_PATH",
        help=(
            "Path to a .pt checkpoint file to resume from. "
            "Example: checkpoints/vlm_checkpoint/checkpoint_epoch_007.pt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum training epochs (default: {NUM_EPOCHS}). "
             "Early stopping may stop before this limit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size per GPU (default: {BATCH_SIZE}). Reduce to 16 if CUDA OOM.",
    )
    args = parser.parse_args()

    main(
        resume_path=args.resume,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )