"""Randomly sample images from a folder, x4-SR each, and build one comparison
collage (RGB + depth) with optional color-corrected / uncorrected columns.

For eyeballing the SR stage on real data: each source image is downsampled to a
base-patch size (simulating the base generator's output), then super-resolved
x4 — so the original doubles as ground truth. The color-corrected column is
derived from the SAME diffusion output as the raw column (match_color is a cheap
post-process), so each image is only denoised once.

Usage:
    python -m dreamsea.sr_batch_collage -m checkpoints_sr/sr_epoch_175.pt \
        -f preprocessed_sr/sr_rgbd -n 4 -o out_batch
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from dreamsea.sr_upscale import (load_sr_model, make_sr_scheduler,
                                 sr_upscale_rgbd, match_color, _to_pil,
                                 load_rgbd_pt)


def save_grid(rows, col_labels, channel, output_dir, stem, pad=6, label_h=20):
    """Write a labelled grid image. `rows` is a list of rows, each a list of
    (4, H, W) [-1, 1] tensors aligned with `col_labels`. `channel` is 'rgb' or
    'depth'. All cells must share H, W."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def cell(rgbd):
        return _to_pil(rgbd[:3] if channel == 'rgb' else rgbd[3])

    cells = [[cell(c) for c in row] for row in rows]
    w, h = cells[0][0].size
    ncol, nrow = len(col_labels), len(rows)

    img = Image.new('RGB', (ncol * w, label_h + nrow * h), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    for j, label in enumerate(col_labels):
        draw.text((j * w + pad, pad), label, fill=(230, 230, 230))
    for i, row in enumerate(cells):
        for j, c in enumerate(row):
            img.paste(c, (j * w, label_h + i * h))

    path = output_dir / f"{stem}_{channel}.png"
    img.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample a folder of RGBD .pt files, x4-SR each, and "
                    "build one comparison collage (raw vs color-corrected vs GT).")
    parser.add_argument("-m", "--sr_ckpt", type=str, required=True,
                        help="Path to the trained SR checkpoint (.pt).")
    parser.add_argument("-f", "--folder", type=str, required=True,
                        help="Folder of RGBD .pt files to sample from (e.g. "
                             "preprocessed_sr/sr_rgbd).")
    parser.add_argument("-n", "--num", type=int, default=4,
                        help="How many files to randomly sample.")
    parser.add_argument("-o", "--output_dir", type=str, default="samples/sr_batch",
                        help="Directory for the collage PNGs.")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=100,
                        help="Denoising steps per image.")
    parser.add_argument("--factor", type=int, default=4, help="Upscale factor.")
    parser.add_argument("--patch_size", type=int, default=224,
                        help="Size the source image is downsampled to before SR "
                             "(simulates a base-model patch). Output is patch_size*factor.")
    parser.add_argument("-d", "--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device.")
    parser.add_argument("--color_strength", type=float, default=1.0,
                        help="Blend factor for the color-corrected column in [0, 1].")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the random sampling for a reproducible pick.")
    parser.add_argument("--no_raw", action="store_true",
                        help="Hide the uncorrected (raw SR) column.")
    parser.add_argument("--no_corrected", action="store_true",
                        help="Hide the color-corrected column.")
    parser.add_argument("--no_gt", action="store_true",
                        help="Hide the ground-truth column.")
    args = parser.parse_args()

    files = sorted(Path(args.folder).glob("*.pt"))
    if not files:
        raise ValueError(f"No .pt files found in {args.folder}")
    if args.seed is not None:
        random.seed(args.seed)
    picks = random.sample(files, min(args.num, len(files)))
    print(f"Sampled {len(picks)}/{len(files)} files from {args.folder}")

    print(f"Loading SR model from {args.sr_ckpt}...")
    model = load_sr_model(args.sr_ckpt, args.device)
    scheduler = make_sr_scheduler(args.sr_ckpt)

    col_labels = ["bilinear input"]
    if not args.no_raw:
        col_labels.append("SR raw")
    if not args.no_corrected:
        col_labels.append("SR corrected")
    if not args.no_gt:
        col_labels.append("ground truth")
    if len(col_labels) == 1:
        raise ValueError("All output columns disabled — nothing to show.")

    rows = []
    for i, p in enumerate(picks):
        print(f"  [{i + 1}/{len(picks)}] {p.name}")
        gt = load_rgbd_pt(p)
        base = F.interpolate(gt.unsqueeze(0), size=(args.patch_size, args.patch_size),
                             mode='area')[0]                     # simulated base patch
        sr_raw = sr_upscale_rgbd(base, model, scheduler,
                                 num_inference_steps=args.num_inference_steps,
                                 factor=args.factor, device=args.device,
                                 color_correct=False)
        out_h, out_w = sr_raw.shape[-2:]
        lr_up = F.interpolate(base.unsqueeze(0), size=(out_h, out_w),
                              mode='bilinear', align_corners=False)[0]

        row = [lr_up]
        if not args.no_raw:
            row.append(sr_raw)
        if not args.no_corrected:
            row.append(match_color(sr_raw, lr_up, strength=args.color_strength))
        if not args.no_gt:
            row.append(F.interpolate(gt.unsqueeze(0), size=(out_h, out_w),
                                     mode='area')[0])
        rows.append(row)

    rgb_path = save_grid(rows, col_labels, 'rgb', args.output_dir, 'batch_collage')
    depth_path = save_grid(rows, col_labels, 'depth', args.output_dir, 'batch_collage')
    print("\nSaved collages:")
    print(f" - {rgb_path}")
    print(f" - {depth_path}")
    print(f"   columns: {' | '.join(col_labels)}   ({len(rows)} rows)")


if __name__ == "__main__":
    main()
