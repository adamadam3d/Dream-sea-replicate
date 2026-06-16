"""
Measure conditioning strength of a conditional DreamSea checkpoint.
==================================================================
Answers "did the retrain actually improve type-conditioning?" with a single
number instead of eyeballing.

Idea: generate K different conditions x S random seeds. If the condition
controls the output, images sharing a condition look alike while images from
different conditions differ — i.e. BETWEEN-condition variation >> WITHIN-
condition (seed) variation. If the model ignores the condition, the two are
equal. We report eta^2 = SS_between / SS_total, the fraction of pixel variance
explained by the condition:

    ~0%   -> condition ignored (model collapsed to its marginal)
    small -> weak control
    large -> strong type control

Seeds are PAIRED across conditions (same initial noise for each seed index) so
the between-condition signal isn't drowned by seed noise.

Run it on the OLD and NEW checkpoints and compare the eta^2 values — a higher
number on the new one is the improvement.

Requires a GPU (it runs diffusion sampling).

Usage:
    python -m dreamsea.measure_conditioning -c checkpoints/cond.pt \
        --conditions_dir path/to/preprocess_out/conditions \
        -k 4 -s 4 --num_inference_steps 200 -o outputs/cond_test
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dreamsea.generation_inpainting import GeneratorInpainter

# Same built-in spread used by generate_3dgs, for the no-data fallback.
DEFAULT_STD = np.array([14.0204, 8.3740], dtype=np.float32)


def _farthest_point_sample(points, k):
    """Pick k indices that are maximally spread out (greedy farthest-point).

    Maximizing between-condition distance makes the conditioning signal — if any
    — as visible as possible.
    """
    n = points.shape[0]
    k = min(k, n)
    # Seed with the point farthest from the centroid.
    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(points - centroid, axis=1)))
    chosen = [first]
    min_d = np.linalg.norm(points - points[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(points - points[nxt], axis=1))
    return chosen


def _pick_conditions(args):
    """Return a (K, 2) array of conditions to probe, plus a label for each."""
    if args.conditions_dir and os.path.isdir(args.conditions_dir):
        pt_files = sorted(Path(args.conditions_dir).glob("*_cond.pt"))
        if not pt_files:
            pt_files = sorted(Path(args.conditions_dir).glob("*.pt"))
        vecs, names = [], []
        for f in pt_files:
            t = torch.load(f, map_location="cpu", weights_only=True)
            vecs.append(np.asarray(t.float().numpy(), dtype=np.float32).reshape(-1))
            names.append(f.stem.replace("_cond", ""))
        vecs = np.stack(vecs, axis=0)
        idx = _farthest_point_sample(vecs, args.num_conditions)
        print(f"Picked {len(idx)} far-apart real conditions from {len(vecs)} samples.")
        return vecs[idx], [names[i] for i in idx]

    # Fallback: synthesize spread conditions at +/- 1.5 std on each axis.
    std = DEFAULT_STD
    if args.latent_stats_path and os.path.exists(args.latent_stats_path):
        with open(args.latent_stats_path) as fh:
            stats = json.load(fh)
        if "std" in stats:
            std = np.array(stats["std"], dtype=np.float32)
    corners = np.array([[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5],
                        [0.0, 0.0]], dtype=np.float32)
    conds = corners[: args.num_conditions] * std
    names = [f"c{i}" for i in range(len(conds))]
    print(f"No conditions_dir; synthesized {len(conds)} spread conditions at +/-1.5 std.")
    return conds, names


def _to_rgb_uint8(img_chw):
    """(4,H,W) or (3,H,W) in [-1,1] -> (H,W,3) uint8 RGB."""
    rgb = img_chw[:3]
    rgb = (rgb + 1.0) / 2.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0))


def main():
    parser = argparse.ArgumentParser(description="Measure conditioning strength (eta^2) of a conditional checkpoint.")
    parser.add_argument("-c", "--cond_model_path", type=str, required=True,
                        help="Conditional DDPM checkpoint to evaluate.")
    parser.add_argument("-i", "--conditions_dir", type=str, default=None,
                        help="Preprocessed conditions/ dir. The probe conditions are the most "
                             "far-apart real samples from it. If omitted, spread conditions are synthesized.")
    parser.add_argument("-l", "--latent_stats_path", type=str, default=None,
                        help="latent_stats.json, used only for the synthesized-conditions fallback.")
    parser.add_argument("-k", "--num_conditions", type=int, default=4, help="Number of distinct conditions to probe.")
    parser.add_argument("-s", "--num_seeds", type=int, default=4, help="Number of random seeds per condition.")
    parser.add_argument("-n", "--num_inference_steps", type=int, default=200,
                        help="Denoising steps per sample (lower = faster, fine for a relative metric).")
    parser.add_argument("-b", "--seed_base", type=int, default=0, help="Base RNG seed (seed index s uses seed_base+s).")
    parser.add_argument("-o", "--output_dir", type=str, default="outputs/cond_test", help="Where to save the collage + report.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    conds, names = _pick_conditions(args)
    K, S = len(conds), args.num_seeds

    print(f"Loading checkpoint: {args.cond_model_path}")
    inpainter = GeneratorInpainter(cond_model_path=args.cond_model_path, uncond_model_path=None, device=args.device)

    # Generate K x S images with paired seeds (same noise per seed index).
    imgs = np.zeros((K, S, 4, 224, 224), dtype=np.float32)
    for s in range(S):
        for c in range(K):
            torch.manual_seed(args.seed_base + s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed_base + s)
            patch = inpainter.generate_patch(conds[c], num_inference_steps=args.num_inference_steps)
            imgs[c, s] = patch[0]
            print(f"  cond {c+1}/{K} ({names[c]}), seed {s+1}/{S} done")

    # eta^2 = SS_between / SS_total, on all channels and split RGB / depth.
    def eta2(x):  # x: (K, S, ...) -> scalar fraction
        x = x.reshape(K, S, -1)
        grand = x.mean(axis=(0, 1))
        mean_c = x.mean(axis=1)                      # (K, D)
        ss_between = S * ((mean_c - grand) ** 2).sum()
        ss_within = ((x - mean_c[:, None, :]) ** 2).sum()
        total = ss_between + ss_within
        return float(ss_between / total) if total > 0 else 0.0

    e_all = eta2(imgs)
    e_rgb = eta2(imgs[:, :, :3])
    e_depth = eta2(imgs[:, :, 3:4])

    print("\n=== Conditioning strength (fraction of output variance explained by the condition) ===")
    print(f"  RGBD : {e_all*100:6.2f}%")
    print(f"  RGB  : {e_rgb*100:6.2f}%")
    print(f"  Depth: {e_depth*100:6.2f}%")
    print("\n  Interpretation (RGB):")
    if e_rgb < 0.02:
        print("  ⚠ <2% — the model essentially IGNORES the condition (no type control).")
    elif e_rgb < 0.10:
        print("  ~ 2-10% — weak conditioning; some type signal but easily lost.")
    else:
        print("  ✓ >10% — clear type control: output tracks the condition.")
    print("  Compare this number between the OLD and NEW checkpoints — higher = improvement.")

    # Visual: rows = conditions, cols = seeds.
    cell = 224
    collage = Image.new("RGB", (S * cell, K * cell))
    for c in range(K):
        for s in range(S):
            collage.paste(Image.fromarray(_to_rgb_uint8(imgs[c, s])), (s * cell, c * cell))
    ckpt_tag = Path(args.cond_model_path).stem
    collage_path = os.path.join(args.output_dir, f"conditioning_{ckpt_tag}.png")
    collage.save(collage_path)
    print(f"\nCollage (rows=conditions, cols=seeds) saved to: {collage_path}")

    report = {
        "checkpoint": args.cond_model_path,
        "num_conditions": K, "num_seeds": S, "num_inference_steps": args.num_inference_steps,
        "conditions": [c.tolist() for c in conds], "condition_names": names,
        "eta2_rgbd": e_all, "eta2_rgb": e_rgb, "eta2_depth": e_depth,
    }
    report_path = os.path.join(args.output_dir, f"conditioning_{ckpt_tag}.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
