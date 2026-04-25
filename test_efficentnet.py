"""
test_efficientnet_detector.py — EfficientNet-B4 Deepfake Detector Test
=======================================================================
Tests the fine-tuned EfficientNet-B4 model in two modes:

  MODE 1 — Single image:
    Loads one image, runs the model, prints P(fake), verdict, and saves
    the Grad-CAM heatmap so you can visually inspect which regions the
    model focused on.

  MODE 2 — Dataset folder:
    Loads a folder of real images and a folder of fake images, runs
    the model on all of them (up to --n per class), and prints:
      - Per-image predictions
      - Accuracy, AUC-ROC, precision, recall, F1
      - Confusion matrix

USAGE
------
  # Test on a single image:
  python test_efficientnet_detector.py \
    --image /path/to/face.jpg \
    --weights models/efficientnet_b4_mfad_final.pth

  # Evaluate on real and fake folders:
  python test_efficientnet_detector.py \
    --real  /path/to/dataset/original \
    --fake  /path/to/dataset/Deepfakes \
    --weights models/efficientnet_b4_mfad_final.pth \
    --n 100

  # Test on your dataset/samples/ folder (auto-detect mode):
  python test_efficientnet_detector.py \
    --samples /path/to/dataset/samples \
    --weights models/efficientnet_b4_mfad_final.pth

  # Save Grad-CAM heatmaps for a folder (visual inspection):
  python test_efficientnet_detector.py \
    --real  /path/to/dataset/original \
    --fake  /path/to/dataset/Deepfakes \
    --weights models/efficientnet_b4_mfad_final.pth \
    --n 10 --save-heatmaps

OUTPUTS
--------
  Terminal: per-image results + aggregate metrics
  test_detector_output/                    (when --save-heatmaps)
    heatmap_REAL_imagename.png
    heatmap_FAKE_imagename.png
"""

import sys
import argparse
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms

try:
    import timm
except ImportError:
    sys.exit("timm not found. Run: pip install timm")

try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found — aggregate metrics will be skipped.")
    SKLEARN_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

EFFICIENTNET_SIZE = 380
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_DIR        = Path("test_detector_output")

# Decision threshold — matches BayesianFusion thresholds in contracts.py
FAKE_THRESHOLD    = 0.50   # P(fake) >= this -> predicted FAKE


# =============================================================================
# MODEL — identical to agents/vlm.py and train_efficientnet.py
# =============================================================================

class EfficientNetB4Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4.ns_jft_in1k",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features  # 1792
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def load_model(weights_path: str, device: str) -> EfficientNetB4Detector:
    path = Path(weights_path)
    if not path.exists():
        sys.exit(
            f"Weights not found: {path}\n"
            f"Run train_efficientnet.py first — it saves to models/efficientnet_b4_mfad_final.pth"
        )
    print(f"Loading weights from: {path}")
    model = EfficientNetB4Detector()
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded on {device} ({total:.1f}M params)\n")
    return model


# =============================================================================
# PREPROCESSING — matches validation transform from train_efficientnet.py
# =============================================================================

_transform = transforms.Compose([
    transforms.Resize(int(EFFICIENTNET_SIZE * 1.14)),
    transforms.CenterCrop(EFFICIENTNET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_image(image_path: Path) -> Optional[torch.Tensor]:
    """Load and preprocess an image. Returns None if the file is unreadable."""
    try:
        img = Image.open(image_path).convert("RGB")
        return _transform(img).unsqueeze(0)   # (1, 3, 380, 380)
    except Exception as exc:
        print(f"  [SKIP] Cannot load {image_path.name}: {exc}")
        return None


# =============================================================================
# GRAD-CAM — identical logic to agents/vlm.py
# =============================================================================

def compute_gradcam(model: EfficientNetB4Detector,
                    pixel_values: torch.Tensor,
                    device: str) -> np.ndarray:
    """
    Compute Grad-CAM for the fake class and return a 224x224 float32 map.
    Returns a uniform 0.5 map if computation fails.
    """
    pixel_values = pixel_values.to(device)

    _fmaps: list = []
    _grads: list = []

    def fwd_hook(m, i, o):
        _fmaps.append(o.detach())

    def bwd_hook(m, gi, go):
        _grads.append(go[0].detach())

    target_layer = model.backbone.blocks[-1]
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        pixel_values.requires_grad_(True)
        logits    = model(pixel_values)
        model.zero_grad()
        logits[0, 1].backward()   # backward from fake-class logit

        feature_maps = _fmaps[0].squeeze(0)
        gradients    = _grads[0].squeeze(0)
        weights      = gradients.mean(dim=(1, 2))

        cam = torch.zeros(feature_maps.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i]

        cam = torch.clamp(cam, min=0)
        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.cpu().numpy().astype(np.float32)
        return cv2.resize(cam_np, (224, 224), interpolation=cv2.INTER_LINEAR)

    except Exception as exc:
        print(f"    [Grad-CAM] Failed: {exc}")
        return np.full((224, 224), 0.5, dtype=np.float32)

    finally:
        fh.remove()
        bh.remove()
        pixel_values.requires_grad_(False)


# =============================================================================
# HEATMAP SAVE
# =============================================================================

def save_heatmap(image_path: Path,
                 cam_map: np.ndarray,
                 label: str,
                 fake_prob: float,
                 output_dir: Path) -> None:
    """
    Save a two-panel Grad-CAM heatmap PNG.
    Left: original image. Right: Grad-CAM overlay.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return

    img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w       = img_rgb.shape[:2]
    cam_sized  = cv2.resize(cam_map, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap    = (cm.get_cmap("jet")(cam_sized)[:, :, :3] * 255).astype(np.uint8)
    overlay    = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_rgb);  axes[0].set_title(f"{label} — original"); axes[0].axis("off")
    axes[1].imshow(overlay);  axes[1].set_title(f"Grad-CAM  P(fake)={fake_prob:.4f}"); axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("0=low fake signal  1=high fake signal", fontsize=8)

    out_name = f"heatmap_{label}_{image_path.stem}.png"
    out_path = output_dir / out_name
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Heatmap -> {out_path}")


# =============================================================================
# INFERENCE
# =============================================================================

def predict_image(model: EfficientNetB4Detector,
                  image_path: Path,
                  device: str,
                  save_heatmaps: bool = False,
                  true_label: Optional[str] = None,
                  output_dir: Optional[Path] = None) -> Optional[dict]:
    """
    Run model on one image. Returns a result dict or None if image unreadable.
    """
    tensor = load_image(image_path)
    if tensor is None:
        return None

    tensor_dev = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor_dev)
        probs  = torch.softmax(logits, dim=1)[0]

    fake_prob = float(probs[1])
    real_prob = float(probs[0])
    verdict   = "FAKE" if fake_prob >= FAKE_THRESHOLD else "REAL"

    result = {
        "path":       str(image_path),
        "name":       image_path.name,
        "fake_prob":  fake_prob,
        "real_prob":  real_prob,
        "verdict":    verdict,
        "true_label": true_label,
        "correct":    (verdict == true_label) if true_label else None,
    }

    if save_heatmaps and output_dir is not None:
        cam_map = compute_gradcam(model, tensor, device)
        display_label = true_label or "UNKNOWN"
        save_heatmap(image_path, cam_map, display_label, fake_prob, output_dir)

    return result


def collect_images(folder: Path, n: Optional[int], seed: int = 42) -> List[Path]:
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    random.Random(seed).shuffle(imgs)
    return imgs[:n] if n else imgs


# =============================================================================
# METRICS
# =============================================================================

def print_metrics(results: List[dict]) -> None:
    """Print aggregate metrics from a list of result dicts that have true_label."""
    labeled = [r for r in results if r["true_label"] is not None]
    if not labeled:
        print("No labeled results — skipping metrics.")
        return

    y_true  = [1 if r["true_label"] == "FAKE" else 0 for r in labeled]
    y_pred  = [1 if r["verdict"]    == "FAKE" else 0 for r in labeled]
    y_score = [r["fake_prob"] for r in labeled]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 55)
    print("AGGREGATE METRICS")
    print("=" * 55)
    print(f"  Total images evaluated : {len(labeled)}")
    print(f"  Accuracy               : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (fake class) : {prec:.4f}")
    print(f"  Recall    (fake class) : {rec:.4f}")
    print(f"  F1 score  (fake class) : {f1:.4f}")

    if SKLEARN_AVAILABLE and len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
        print(f"  AUC-ROC               : {auc:.4f}")
    else:
        print("  AUC-ROC               : (need both classes)")

    cm = confusion_matrix(y_true, y_pred)
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print(f"                  Pred REAL  Pred FAKE")
    print(f"  True REAL  :     {cm[0][0]:6d}     {cm[0][1]:6d}")
    print(f"  True FAKE  :     {cm[1][0]:6d}     {cm[1][1]:6d}")

    # False positive / false negative rates
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    print(f"\n  False positive rate (real misclassified as fake): {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  False negative rate (fake misclassified as real): {fnr:.4f} ({fnr*100:.2f}%)")
    print("=" * 55)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test EfficientNet-B4 deepfake detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image — shows P(fake) and verdict:
  python test_efficientnet_detector.py --image face.jpg --weights models/efficientnet_b4_mfad_final.pth

  # Evaluate on real and fake folders:
  python test_efficientnet_detector.py --real dataset/original --fake dataset/Deepfakes --n 100

  # Save Grad-CAM heatmaps for visual inspection:
  python test_efficientnet_detector.py --real dataset/original --fake dataset/Deepfakes --n 10 --save-heatmaps

  # Quick sanity check on samples folder:
  python test_efficientnet_detector.py --samples dataset/samples
        """,
    )
    parser.add_argument("--weights",       default="models/efficientnet_b4_mfad_final.pth",
                        help="Path to fine-tuned .pth weights file")
    parser.add_argument("--image",         help="Path to a single image to test")
    parser.add_argument("--real",          help="Folder of real images")
    parser.add_argument("--fake",          help="Folder of fake images")
    parser.add_argument("--samples",       help="A samples folder (runs detect mode, no labels)")
    parser.add_argument("--n",             type=int, default=None,
                        help="Max images per class to evaluate (default: all)")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save Grad-CAM heatmap PNGs to test_detector_output/")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--threshold",     type=float, default=FAKE_THRESHOLD,
                        help=f"P(fake) threshold for FAKE verdict (default: {FAKE_THRESHOLD})")
    args = parser.parse_args()

    global FAKE_THRESHOLD
    FAKE_THRESHOLD = args.threshold

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model = load_model(args.weights, device)

    output_dir = OUTPUT_DIR if args.save_heatmaps else None
    all_results: List[dict] = []

    # ── Single image mode ─────────────────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        print(f"\nTesting single image: {img_path}")
        t0 = time.time()
        result = predict_image(
            model, img_path, device,
            save_heatmaps=args.save_heatmaps,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0

        if result:
            print(f"\n{'='*45}")
            print(f"  Image    : {result['name']}")
            print(f"  P(fake)  : {result['fake_prob']:.4f}")
            print(f"  P(real)  : {result['real_prob']:.4f}")
            print(f"  Verdict  : {result['verdict']}")
            print(f"  Time     : {elapsed:.2f}s")
            print(f"{'='*45}")

    # ── Samples folder — no labels, just predictions ──────────────────────────
    elif args.samples:
        samples_dir = Path(args.samples)
        if not samples_dir.exists():
            sys.exit(f"Samples folder not found: {samples_dir}")

        images = collect_images(samples_dir, args.n, args.seed)
        print(f"\nTesting {len(images)} images from: {samples_dir}\n")
        print(f"  {'Filename':<40} {'P(fake)':>8}  {'P(real)':>8}  Verdict")
        print(f"  {'-'*40} {'-'*8}  {'-'*8}  -------")

        for img_path in images:
            result = predict_image(
                model, img_path, device,
                save_heatmaps=args.save_heatmaps,
                output_dir=output_dir,
            )
            if result:
                print(f"  {result['name']:<40} "
                      f"{result['fake_prob']:>8.4f}  "
                      f"{result['real_prob']:>8.4f}  "
                      f"{result['verdict']}")
                all_results.append(result)

    # ── Real + fake folder evaluation ─────────────────────────────────────────
    elif args.real or args.fake:
        if args.real:
            real_dir = Path(args.real)
            if not real_dir.exists():
                sys.exit(f"Real folder not found: {real_dir}")
            real_images = collect_images(real_dir, args.n, args.seed)
            print(f"Real images: {len(real_images)} from {real_dir}")
        else:
            real_images = []

        if args.fake:
            fake_dir = Path(args.fake)
            if not fake_dir.exists():
                sys.exit(f"Fake folder not found: {fake_dir}")
            fake_images = collect_images(fake_dir, args.n, args.seed)
            print(f"Fake images: {len(fake_images)} from {fake_dir}")
        else:
            fake_images = []

        total = len(real_images) + len(fake_images)
        print(f"\nEvaluating {total} images total...\n")
        print(f"  {'Label':<6} {'Filename':<40} {'P(fake)':>8}  {'P(real)':>8}  Verdict  Correct")
        print(f"  {'-'*6} {'-'*40} {'-'*8}  {'-'*8}  -------  -------")

        for img_path in real_images:
            result = predict_image(
                model, img_path, device,
                save_heatmaps=args.save_heatmaps,
                true_label="REAL",
                output_dir=output_dir,
            )
            if result:
                correct_str = "OK" if result["correct"] else "WRONG"
                print(f"  {'REAL':<6} {result['name']:<40} "
                      f"{result['fake_prob']:>8.4f}  {result['real_prob']:>8.4f}  "
                      f"{result['verdict']:<8} {correct_str}")
                all_results.append(result)

        for img_path in fake_images:
            result = predict_image(
                model, img_path, device,
                save_heatmaps=args.save_heatmaps,
                true_label="FAKE",
                output_dir=output_dir,
            )
            if result:
                correct_str = "OK" if result["correct"] else "WRONG"
                print(f"  {'FAKE':<6} {result['name']:<40} "
                      f"{result['fake_prob']:>8.4f}  {result['real_prob']:>8.4f}  "
                      f"{result['verdict']:<8} {correct_str}")
                all_results.append(result)

        if SKLEARN_AVAILABLE:
            print_metrics(all_results)

    else:
        parser.print_help()
        print("\nERROR: provide --image, --samples, or --real/--fake")
        sys.exit(1)


if __name__ == "__main__":
    main()