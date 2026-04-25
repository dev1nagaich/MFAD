"""
infer_texture.py — Single-image / batch CLI inference for the NPR texture agent.

Usage:
  python infer_texture.py --image path/to/face.jpg
  python infer_texture.py --image_dir path/to/folder/  --out outputs/preds.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from agents.texture_agent import NPRDetector, load_npr_state_dict, TextureAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("infer_texture")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class _Folder(Dataset):
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    def __init__(self, paths: List[Path]):
        self.paths = paths
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
            return self.tf(img), str(self.paths[idx]), 1
        except Exception as e:
            log.warning("load failed %s: %s", self.paths[idx], e)
            return self.tf(Image.new("RGB", (256, 256))), str(self.paths[idx]), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",     type=Path, help="Single image path.")
    ap.add_argument("--image_dir", type=Path, help="Folder with images (recursive).")
    ap.add_argument("--weights",   type=Path,
                    default=Path("checkpoints/texture_checkpoint/npr_finetuned.pth"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", type=Path, help="Output JSON (only used with --image_dir).")
    args = ap.parse_args()

    if not args.image and not args.image_dir:
        ap.error("Provide --image or --image_dir.")

    if args.image:
        agent = TextureAgent(weights_path=str(args.weights) if args.weights.exists() else None)
        result = agent.analyze(Image.open(args.image).convert("RGB"))
        d = result.to_dict()
        d["image_path"] = str(args.image)
        print(json.dumps(d, indent=2))
        return

    paths = [p for p in args.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    log.info("Found %d images under %s", len(paths), args.image_dir)
    if not paths:
        return

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NPRDetector().to(device)
    load_npr_state_dict(model, str(args.weights))
    model.eval()

    loader = DataLoader(_Folder(paths), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    results = []
    with torch.no_grad():
        for x, p, ok in tqdm(loader, ncols=100):
            x = x.to(device, non_blocking=True)
            probs = torch.sigmoid(model(x)).squeeze(-1).cpu().numpy()
            for prob, path, ok_flag in zip(probs, p, ok.tolist()):
                if not ok_flag:
                    continue
                results.append({
                    "image_path": path,
                    "npr_fake_probability": float(prob),
                    "is_fake": bool(prob >= args.threshold),
                })

    out_path = args.out or args.image_dir / "infer_texture.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"weights": str(args.weights),
                   "threshold": args.threshold,
                   "count": len(results),
                   "results": results}, f, indent=2)
    log.info("wrote %d predictions → %s", len(results), out_path)


if __name__ == "__main__":
    main()
