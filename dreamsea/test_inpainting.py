"""
Standalone visual + automated test for RePaint inpainting.

Unlike test_inference.py (which uses pure random noise as the "known" region and
therefore always *looks* like noise), this builds a STRUCTURED known RGBD image,
punches a rectangular hole, fills it with garbage, and asks repaint_inpaint to
restore it. It then:

  1. Saves side-by-side RGB and depth comparisons (original | hole | inpainted).
  2. Runs automated checks for the two classic failure modes:
       - the KNOWN region getting corrupted   (mask / algorithm bug)
       - the inpainted HOLE coming out as pure noise
         (e.g. checkpoint silently not loaded -> random UNet weights)

A trained unconditional checkpoint is required for a meaningful result; with no
checkpoint the UNet runs on random weights and the noise check is *expected* to
fail (the script says so).

Usage:
  PYTHONPATH=. python dreamsea/test_inpainting.py \
      --uncond_model_path checkpoints/unconditional_epoch_500.pt \
      [--cond_model_path checkpoints/conditional_epoch_500.pt] \
      [--num_inference_steps 250] [--jump_n_sample 5] \
      [--output_dir samples/inpaint_test]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dreamsea.generation_inpainting import GeneratorInpainter

SIZE = 224
HOLE = 80  # side length of the centred square hole


def to_uint8_rgb(chw):
    """(3, H, W) float in [-1, 1] -> (H, W, 3) uint8."""
    img = (np.transpose(chw[:3], (1, 2, 0)) + 1.0) / 2.0
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def to_uint8_depth(hw):
    """(H, W) float in [-1, 1] -> (H, W) uint8."""
    img = (hw + 1.0) / 2.0
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def synthetic_rgbd(size=SIZE):
    """A smooth, structured RGBD canvas in [-1, 1] (no model needed)."""
    ys, xs = np.mgrid[0:size, 0:size] / (size - 1)
    r = np.sin(xs * np.pi * 2.0) * 0.5 + 0.5
    g = ys
    b = (1.0 - ys) * xs
    d = np.sqrt((xs - 0.5) ** 2 + (ys - 0.5) ** 2)
    d = d / d.max()
    rgbd = np.stack([r, g, b, d], axis=0).astype(np.float32)  # [0, 1]
    return rgbd * 2.0 - 1.0  # -> [-1, 1]


def total_variation(chw):
    """Mean abs difference between adjacent pixels over RGB channels.

    Coherent images have low TV (~0.05-0.15); per-pixel i.i.d. noise has high
    TV (~0.7-1.1 in this [-1, 1] range). A good 'is it noise?' signal.
    """
    rgb = chw[:3]
    dx = np.abs(rgb[:, :, 1:] - rgb[:, :, :-1]).mean()
    dy = np.abs(rgb[:, 1:, :] - rgb[:, :-1, :]).mean()
    return float((dx + dy) / 2.0)


def save_comparison(path, tiles, depth=False):
    """tiles: list of (label, chw). Saves them in a horizontal strip."""
    imgs = []
    for _, chw in tiles:
        if depth:
            imgs.append(Image.fromarray(to_uint8_depth(chw[3]), mode="L"))
        else:
            imgs.append(Image.fromarray(to_uint8_rgb(chw), mode="RGB"))
    w, h = imgs[0].size
    mode = "L" if depth else "RGB"
    strip = Image.new(mode, (w * len(imgs), h))
    for i, im in enumerate(imgs):
        strip.paste(im, (i * w, 0))
    strip.save(path)


def main():
    parser = argparse.ArgumentParser(description="Test RePaint inpainting end to end.")
    parser.add_argument("-u", "--uncond_model_path", type=str, default=None,
                        help="Unconditional DDPM checkpoint (used to fill the hole).")
    parser.add_argument("-c", "--cond_model_path", type=str, default=None,
                        help="Optional conditional checkpoint. If given, a real generated "
                             "patch is used as the known canvas instead of a synthetic one.")
    parser.add_argument("-i", "--num_inference_steps", type=int, default=250)
    parser.add_argument("-j", "--jump_length", type=int, default=10)
    parser.add_argument("-n", "--jump_n_sample", type=int, default=5,
                        help="RePaint time-travel resamples per jump (1 disables time-travel).")
    parser.add_argument("-o", "--output_dir", type=str, default="samples/inpaint_test")
    parser.add_argument("-x", "--seed", type=int, default=0)
    parser.add_argument("-d", "--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.uncond_model_path:
        print("WARNING: no --uncond_model_path given. The hole will be filled by a UNet with "
              "RANDOM weights, so the noise check is EXPECTED to fail. Pass a checkpoint for a "
              "real test.\n")

    print(f"Initializing GeneratorInpainter on {args.device}...")
    gen = GeneratorInpainter(
        cond_model_path=args.cond_model_path,
        uncond_model_path=args.uncond_model_path,
        device=args.device,
    )

    # --- 1. Build a structured "known" canvas ---
    if args.cond_model_path:
        print("Generating a real RGBD patch with the conditional model as the known canvas...")
        latent = np.array([0.0, 0.0], dtype=np.float32)
        known = gen.generate_patch(latent, num_inference_steps=args.num_inference_steps)[0]
    else:
        print("Using a synthetic structured canvas (no conditional checkpoint).")
        known = synthetic_rgbd()

    # --- 2. Mask: 1 = known, 0 = hole (centred square) ---
    lo, hi = (SIZE - HOLE) // 2, (SIZE + HOLE) // 2
    mask = np.ones((1, SIZE, SIZE), dtype=np.float32)
    mask[:, lo:hi, lo:hi] = 0.0

    # Corrupt the hole in the INPUT with noise so a successful repaint cannot be a
    # trivial pass-through copy of the original.
    corrupted = known.copy()
    corrupted[:, lo:hi, lo:hi] = np.random.randn(4, HOLE, HOLE).astype(np.float32)

    image_input = corrupted[np.newaxis, ...]  # (1, 4, H, W)
    mask_in = mask[np.newaxis, ...]           # (1, 1, H, W)

    # --- 3. Run RePaint ---
    print(f"Running repaint_inpaint ({args.num_inference_steps} steps, "
          f"jump_n_sample={args.jump_n_sample})...")
    result = gen.repaint_inpaint(
        image_input, mask_in,
        num_inference_steps=args.num_inference_steps,
        jump_length=args.jump_length,
        jump_n_sample=args.jump_n_sample,
    )[0]  # (4, H, W)

    # --- 4. Save visuals ---
    save_comparison(out / "rgb_comparison.png",
                    [("original", known), ("hole", corrupted), ("inpainted", result)])
    save_comparison(out / "depth_comparison.png",
                    [("original", known), ("hole", corrupted), ("inpainted", result)],
                    depth=True)
    print(f"Saved comparisons to {out / 'rgb_comparison.png'} and {out / 'depth_comparison.png'}")

    # --- 5. Automated checks ---
    known_px = mask[0] > 0.5
    diff = np.abs(result - known)                      # (4, H, W)
    known_diff = float(diff[:, known_px].mean())       # should be ~0 (known is clamped)

    hole_out = result[:, lo:hi, lo:hi]
    hole_in = corrupted[:, lo:hi, lo:hi]
    hole_changed = float(np.abs(hole_out - hole_in).mean())  # repaint must alter the hole
    hole_tv = total_variation(hole_out)
    known_tv = total_variation(known)
    out_min, out_max = float(result.min()), float(result.max())

    print("\n================ RESULTS ================")
    print(f"Output value range:            [{out_min:.3f}, {out_max:.3f}]  (expected ~[-1, 1])")
    print(f"Known-region mean |diff|:      {known_diff:.4f}  (want < 0.05 -> region preserved)")
    print(f"Hole changed vs input:         {hole_changed:.4f}  (want > 0.10 -> repaint ran)")
    print(f"Hole total variation:          {hole_tv:.4f}  (want < 0.20; > 0.50 == noise)")
    print(f"Known total variation (ref):   {known_tv:.4f}")

    preserved_ok = known_diff < 0.05
    ran_ok = hole_changed > 0.10
    not_noise_ok = hole_tv < 0.50

    print("\nChecks:")
    print(f"  [{'PASS' if preserved_ok else 'FAIL'}] Known region preserved")
    print(f"  [{'PASS' if ran_ok else 'FAIL'}] RePaint modified the hole")
    print(f"  [{'PASS' if not_noise_ok else 'FAIL'}] Hole is not pure noise")

    if not not_noise_ok:
        print("\n  -> The hole is essentially random noise. With a checkpoint loaded this points\n"
              "     to a denoising failure (e.g. weights not actually loaded). Without one it is\n"
              "     expected (random UNet weights).")

    overall = preserved_ok and ran_ok and not_noise_ok
    print("\n" + ("OVERALL: PASS" if overall else "OVERALL: FAIL") + "\n")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
