"""Preprocess full-resolution RGBD tensors for training the x4 SR stage.

Unlike preprocess_dataset.py (which downsizes every photo to 224 for the base
generator), this keeps the source photos at capped native resolution — the fine
detail the SR model learns to synthesize comes from here. Depth is estimated
with the same Depth-Anything-V2-Large model used by the base pipeline so the
two stages share a consistent notion of relative depth.

Output layout (matches what train.py's SRPairDataset expects):
    <output_dir>/sr_rgbd/<name>_rgbd.pt   float16 (4, H, W) in [0, 1]

Usage:
    python -m dreamsea.preprocess_sr_dataset -i raw_photos/ -o preprocessed_sr/
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def preprocess_sr_dataset(input_dir, output_dir, max_side=1792, min_side=224,
                          device='cuda' if torch.cuda.is_available() else 'cpu'):
    input_dir = Path(input_dir)
    out_dir = Path(output_dir) / "sr_rgbd"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    print(f"Loading Depth-Anything-V2-Large on {device}...")
    depth_estimator = pipeline(task="depth-estimation",
                               model="depth-anything/Depth-Anything-V2-Large-hf",
                               device=device)

    kept = 0
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        if min(W, H) < min_side:
            print(f"[{i + 1}/{len(image_paths)}] SKIP {img_path.name}: "
                  f"min side {min(W, H)} < {min_side}")
            continue

        # Cap the long side to bound file size/VRAM; Lanczos preserves the
        # edges the SR model is supposed to learn.
        if max(W, H) > max_side:
            scale = max_side / max(W, H)
            image = image.resize((round(W * scale), round(H * scale)),
                                 Image.Resampling.LANCZOS)
            W, H = image.size

        depth = depth_estimator(image)['depth']
        if depth.size != (W, H):
            depth = depth.resize((W, H), Image.Resampling.NEAREST)
        depth_np = np.array(depth, dtype=np.float32)
        dmin, dmax = depth_np.min(), depth_np.max()
        if dmax > dmin:
            depth_np = (depth_np - dmin) / (dmax - dmin)
        else:
            depth_np = np.zeros_like(depth_np)

        rgb_np = np.asarray(image, dtype=np.float32) / 255.0            # (H, W, 3)
        rgbd = np.concatenate([rgb_np, depth_np[..., None]], axis=-1)   # (H, W, 4)
        tensor = torch.from_numpy(rgbd).permute(2, 0, 1).contiguous().half()

        torch.save(tensor, out_dir / f"{img_path.stem}_rgbd.pt")
        kept += 1
        print(f"[{i + 1}/{len(image_paths)}] {img_path.name} -> {W}x{H}")

    print(f"\nDone. {kept}/{len(image_paths)} images saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess full-resolution RGBD tensors for SR-stage training.")
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory of raw source photos.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory (an 'sr_rgbd' subfolder is created).")
    parser.add_argument("-m", "--max_side", type=int, default=1792,
                        help="Cap on the photo's long side before depth estimation. Default 1792.")
    parser.add_argument("-d", "--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device.")
    args = parser.parse_args()

    preprocess_sr_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_side=args.max_side,
        device=args.device,
    )
