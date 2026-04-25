"""
texture_agent.py — NPR (Neighboring Pixel Relationships) Deepfake Detector
═══════════════════════════════════════════════════════════════════════════════

Implementation of NPR (Tan et al., CVPR 2024 — arXiv:2312.10461):
  "Rethinking the Up-Sampling Operations in CNN-based Generative Network
   for Generalizable Deepfake Detection"

Architecture:
  • ResNet50 backbone (1 logit out)
  • Forward pass replaces input with the local NPR residual:
        NPR = x - up(down(x, 1/2))    (nearest-nearest)
        out = ResNet50(NPR * 2/3)
  • Official weights: chuangchuangtan/NPR-DeepfakeDetection
    Place at checkpoints/texture_checkpoint/npr_finetuned.pth (after fine-tune)
    or checkpoints/texture_checkpoint/NPR.pth (official ProGAN-trained init).

Inference contract:
  Master agent passes a 256×256 face crop (preprocessing handles resize).
  Agent runs single forward pass, returns calibrated P(fake).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models.resnet import Bottleneck

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("texture_agent")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL
# ═════════════════════════════════════════════════════════════════════════════

class NPRDetector(nn.Module):
    """Official NPR architecture (Tan et al., CVPR 2024) — 1.44M params.

    Truncated ResNet50: only layer1 (3 Bottleneck) + layer2 (4 Bottleneck, stride 2).
    Output of layer2 is 512 channels (Bottleneck expansion 4 × 128).

      conv1: Conv2d(3, 64, k=3, s=2, p=1)
      bn1, relu, maxpool(k=3, s=2, p=1)
      layer1: 3× Bottleneck(64)
      layer2: 4× Bottleneck(128, stride=2)
      avgpool: AdaptiveAvgPool2d(1)
      fc1:    Linear(512, 1)

    Forward:
      NPR = x - interpolate(x, 0.5)         # down-then-up nearest
      out = network(NPR * 2/3)
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.inplanes = 64
        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(Bottleneck, 64, 3)
        self.layer2  = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1     = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    @staticmethod
    def _interpolate(img: torch.Tensor, factor: float) -> torch.Tensor:
        down = F.interpolate(img, scale_factor=factor, mode="nearest",
                             recompute_scale_factor=True)
        return F.interpolate(down, scale_factor=1.0 / factor, mode="nearest",
                             recompute_scale_factor=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        npr = x - self._interpolate(x, 0.5)
        x = self.conv1(npr * 2.0 / 3.0)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)


def load_npr_state_dict(model: NPRDetector, weights_path: str) -> None:
    """Load NPR weights. Strips legacy `backbone.` and `module.` prefixes."""
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if unexpected:
        log.warning("Unexpected keys when loading NPR weights: %s", unexpected[:5])
    if missing:
        critical = [m for m in missing if "num_batches_tracked" not in m]
        if critical:
            log.warning("Missing keys when loading NPR weights: %s", critical[:5])


# ═════════════════════════════════════════════════════════════════════════════
# RESULT SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class TextureAnalysisResult:
    npr_fake_probability: float
    texture_fake_probability: float
    is_fake: bool
    anomaly_score: float
    model_name: str
    model_version: str
    inference_ms: float
    analyst_note: str = ""
    processing_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT
# ═════════════════════════════════════════════════════════════════════════════

class TextureAgent:
    """NPR-based deepfake texture detector (single forward pass)."""

    DEFAULT_WEIGHTS = (
        Path(__file__).resolve().parent.parent
        / "checkpoints" / "texture_checkpoint" / "npr_finetuned.pth"
    )
    FALLBACK_WEIGHTS = (
        Path(__file__).resolve().parent.parent
        / "checkpoints" / "texture_checkpoint" / "NPR.pth"
    )
    MODEL_VERSION = "NPR-ResNet50-finetuned-v1"
    DECISION_THRESHOLD = 0.50
    INPUT_SIZE = 256

    # NPR official preprocessing — ImageNet normalization
    _NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, device: Optional[str] = None, weights_path: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if weights_path is None:
            if self.DEFAULT_WEIGHTS.exists():
                weights_path = str(self.DEFAULT_WEIGHTS)
            elif self.FALLBACK_WEIGHTS.exists():
                weights_path = str(self.FALLBACK_WEIGHTS)
                log.warning(
                    "Fine-tuned weights not found, using official NPR.pth init. "
                    "Run train_texture.py to produce npr_finetuned.pth."
                )
            else:
                raise FileNotFoundError(
                    f"NPR weights not found at {self.DEFAULT_WEIGHTS} or "
                    f"{self.FALLBACK_WEIGHTS}. Download official weights to "
                    f"checkpoints/texture_checkpoint/NPR.pth then run train_texture.py."
                )
        self.weights_path = weights_path

        self.model = NPRDetector().to(self.device)
        load_npr_state_dict(self.model, weights_path)
        self.model.eval()

        self._transform = transforms.Compose([
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            self._NORMALIZE,
        ])

        log.info(
            "TextureAgent ready | device=%s | weights=%s",
            self.device, Path(weights_path).name,
        )

    @torch.no_grad()
    def analyze(
        self,
        image: Image.Image | np.ndarray,
        face_box: Optional[BoundingBox] = None,
    ) -> TextureAnalysisResult:
        """Run NPR forward pass on a face crop. `face_box` accepted for legacy
        master_agent compat — preprocessing already crops/resizes the input."""
        notes: List[str] = []
        t0 = time.perf_counter()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                pil = Image.fromarray(image[..., ::-1] if image.dtype == np.uint8 else image).convert("RGB")
            else:
                pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if face_box is not None:
            try:
                pil = pil.crop((face_box.x1, face_box.y1, face_box.x2, face_box.y2))
            except Exception as e:
                notes.append(f"face_box crop failed: {e}")

        x = self._transform(pil).unsqueeze(0).to(self.device)
        logit = self.model(x)
        prob = torch.sigmoid(logit).item()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_fake = prob >= self.DECISION_THRESHOLD

        if prob >= 0.85:
            note = f"HIGH confidence FAKE (NPR p={prob:.3f})"
        elif prob >= self.DECISION_THRESHOLD:
            note = f"FAKE (NPR p={prob:.3f})"
        elif prob >= 0.20:
            note = f"BORDERLINE — likely AUTHENTIC (NPR p={prob:.3f})"
        else:
            note = f"AUTHENTIC (NPR p={prob:.3f})"

        return TextureAnalysisResult(
            npr_fake_probability=float(prob),
            texture_fake_probability=float(prob),
            is_fake=bool(is_fake),
            anomaly_score=float(prob),
            model_name="NPR",
            model_version=self.MODEL_VERSION,
            inference_ms=float(elapsed_ms),
            analyst_note=note,
            processing_notes=notes,
        )

    @torch.no_grad()
    def predict_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Score a pre-transformed batch of [N, 3, 256, 256] tensors. Returns probs."""
        batch = batch.to(self.device, non_blocking=True)
        return torch.sigmoid(self.model(batch)).squeeze(-1).cpu()


# ═════════════════════════════════════════════════════════════════════════════
# LEGACY ENTRY POINT (master_agent compat)
# ═════════════════════════════════════════════════════════════════════════════

_AGENT_SINGLETON: Optional[TextureAgent] = None

def _get_agent() -> TextureAgent:
    global _AGENT_SINGLETON
    if _AGENT_SINGLETON is None:
        _AGENT_SINGLETON = TextureAgent()
    return _AGENT_SINGLETON


def run(
    image_path: Optional[str] = None,
    image_bgr: Optional[np.ndarray] = None,
    face_bbox: Optional[List[int]] = None,
    **kwargs,
) -> Dict:
    """Legacy interface for master_agent."""
    try:
        if image_path:
            img = Image.open(image_path).convert("RGB")
        elif image_bgr is not None:
            img = Image.fromarray(image_bgr[..., ::-1]).convert("RGB")
        else:
            raise ValueError("Must provide image_path or image_bgr")

        bbox = None
        if face_bbox and len(face_bbox) == 4:
            bbox = BoundingBox(
                x1=int(face_bbox[0]), y1=int(face_bbox[1]),
                x2=int(face_bbox[2]), y2=int(face_bbox[3]),
            )

        result = _get_agent().analyze(img, bbox)
        return result.to_dict()

    except Exception as e:
        log.error("TextureAgent error: %s", e, exc_info=True)
        return {
            "npr_fake_probability": 0.5,
            "texture_fake_probability": 0.5,
            "is_fake": False,
            "anomaly_score": 0.5,
            "model_name": "NPR",
            "model_version": TextureAgent.MODEL_VERSION,
            "inference_ms": 0.0,
            "analyst_note": f"Error: {e}",
            "processing_notes": [str(e)],
            "error": str(e),
        }


if __name__ == "__main__":
    print("texture_agent (NPR) loaded.")
