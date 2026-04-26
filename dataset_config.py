"""
dataset_config.py — Single source of truth for dataset layout.

Every dataset under `dataset/` follows: <name>/{train,test}/{fake,real}/
Some folders only have `real/`, some only `fake/`, some have both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# Verified from `find dataset -maxdepth 3 -type d` on 2026-04-26.
DATASETS: Dict[str, Dict[str, bool]] = {
    "100KFake_10K":                       {"real": False, "fake": True},
    "AttGAN":                             {"real": True,  "fake": True},
    "BEGAN":                              {"real": True,  "fake": True},
    "celeba":                             {"real": True,  "fake": False},
    "celebA-HQ_10K":                      {"real": True,  "fake": False},
    "CramerGAN":                          {"real": True,  "fake": True},
    "deepdetect25":                       {"real": True,  "fake": True},
    "deepfake":                           {"real": True,  "fake": True},
    "DeepFakeDetection":                  {"real": False, "fake": True},
    "Deepfakes":                          {"real": False, "fake": True},
    "Face2Face":                          {"real": False, "fake": True},
    "FaceShifter":                        {"real": False, "fake": True},
    "FaceSwap":                           {"real": False, "fake": True},
    "Flickr-Faces-HQ_10K":                {"real": True,  "fake": False},
    "InfoMaxGAN":                         {"real": True,  "fake": True},
    "MMDGAN":                             {"real": True,  "fake": True},
    "NeuralTextures":                     {"real": False, "fake": True},
    "original":                           {"real": True,  "fake": False},
    "progan":                             {"real": True,  "fake": True},
    "RelGAN":                             {"real": True,  "fake": True},
    "SNGAN":                              {"real": True,  "fake": True},
    "stable-diffusion-face-dataset_512":  {"real": False, "fake": True},
    "stable-diffusion-face-dataset_768":  {"real": False, "fake": True},
    "stable-diffusion-face-dataset_1024": {"real": False, "fake": True},
    "stargan":                            {"real": True,  "fake": True},
    "STGAN":                              {"real": True,  "fake": True},
    "thispersondoesntexists_10K":         {"real": False, "fake": True},
    "whichfaceisreal":                    {"real": True,  "fake": True},
}

# Texture-relevant training subsets — manipulation/generator artifacts.
# CelebA excluded (low-res 178×218, JPEG-compressed → out-of-distribution).
TRAIN_REAL_SOURCES: List[str] = [
    "original",
    "Flickr-Faces-HQ_10K",
    "celebA-HQ_10K",
]
TRAIN_FAKE_SOURCES: List[str] = [
    "Deepfakes",
    "FaceSwap",
    "FaceShifter",
    "NeuralTextures",
    "100KFake_10K",
    "thispersondoesntexists_10K",
    "AttGAN",
    "STGAN",
    "stargan",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_class(dataset_root: Path, name: str, split: str, cls: str) -> List[Path]:
    """Return image paths under `<dataset_root>/<name>/<split>/<cls>/`.

    Empty list if directory missing — caller decides whether that's an error.
    `cls` must be "real" or "fake".
    """
    if cls not in ("real", "fake"):
        raise ValueError(f"cls must be 'real' or 'fake', got {cls!r}")
    d = Path(dataset_root) / name / split / cls
    if not d.exists():
        return []
    return [p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
