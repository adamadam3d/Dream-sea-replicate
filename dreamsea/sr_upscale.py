"""Stage-2 inference of the cascade: x4-upscale a generated RGBD patch or map.

Takes any (4, H, W) or (1, 4, H, W) RGBD tensor in [-1, 1] — e.g. the
*_rgbd_map.pt written by generate_3dgs.py, or a raw base-model patch — and
produces a (4, 4H, 4W) RGBD tensor via the trained SR DDPM.

Usage:
    python -m dreamsea.sr_upscale -m checkpoints/sr_epoch_200.pt \
        -i outputs/run/run_rgbd_map.pt -o outputs/run_sr
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler

from dreamsea.models import SRDDPM
from dreamsea.generation_inpainting import _load_ddpm_checkpoint


def load_sr_model(ckpt_path, device):
    model = SRDDPM().to(device)
    _load_ddpm_checkpoint(model, ckpt_path, device)
    model.eval()
    return model


@torch.no_grad()
def sr_upscale_rgbd(rgbd, model, scheduler, num_inference_steps=100, factor=4,
                    device='cuda'):
    """rgbd: (4, H, W) tensor in [-1, 1]. Returns (4, factor*H, factor*W) in [-1, 1]."""
    lr = rgbd.unsqueeze(0).float().to(device)
    H, W = lr.shape[-2:]
    lr_up = F.interpolate(lr, size=(H * factor, W * factor),
                          mode='bilinear', align_corners=False)

    image = torch.randn(1, 4, H * factor, W * factor, device=device)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    for t in scheduler.timesteps:
        noise_pred = model(torch.cat([image, lr_up], dim=1), t)
        image = scheduler.step(noise_pred, t, image).prev_sample
    return image[0].cpu()


def save_rgbd_outputs(rgbd, output_dir, stem):
    """Save a (4, H, W) [-1, 1] RGBD tensor as RGB png, depth png and a .pt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arr = torch.clamp((rgbd.detach().cpu() + 1.0) / 2.0, 0.0, 1.0).numpy()
    rgb = (np.transpose(arr[:3], (1, 2, 0)) * 255).astype(np.uint8)
    depth = (arr[3] * 255).astype(np.uint8)

    rgb_path = output_dir / f"{stem}_rgb.png"
    depth_path = output_dir / f"{stem}_depth.png"
    pt_path = output_dir / f"{stem}_rgbd.pt"
    Image.fromarray(rgb, mode='RGB').save(rgb_path)
    Image.fromarray(depth, mode='L').save(depth_path)
    torch.save(rgbd.detach().cpu(), pt_path)
    return rgb_path, depth_path, pt_path


def main():
    parser = argparse.ArgumentParser(
        description="x4-upscale an RGBD tensor with the trained SR cascade stage.")
    parser.add_argument("-m", "--sr_ckpt", type=str, required=True,
                        help="Path to the trained SR checkpoint (.pt).")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to a .pt holding a (4, H, W) or (1, 4, H, W) RGBD "
                             "tensor in [-1, 1] (e.g. *_rgbd_map.pt from generate_3dgs.py).")
    parser.add_argument("-o", "--output_dir", type=str, default="samples/sr",
                        help="Directory for the upscaled outputs.")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=100,
                        help="Denoising steps for the SR stage. SR converges with far "
                             "fewer steps than base generation; 100 is plenty.")
    parser.add_argument("-f", "--factor", type=int, default=4,
                        help="Upscale factor. The model is trained for 4; other values "
                             "are off-distribution.")
    parser.add_argument("-d", "--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device.")
    args = parser.parse_args()

    rgbd = torch.load(args.input, map_location='cpu', weights_only=True).float()
    while rgbd.dim() > 3 and rgbd.shape[0] == 1:
        rgbd = rgbd.squeeze(0)
    if rgbd.dim() != 3 or rgbd.shape[0] != 4:
        raise ValueError(f"Expected a (4, H, W) RGBD tensor, got shape {tuple(rgbd.shape)}")

    print(f"Loading SR model from {args.sr_ckpt}...")
    model = load_sr_model(args.sr_ckpt, args.device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    H, W = rgbd.shape[-2:]
    print(f"Upscaling {W}x{H} -> {W * args.factor}x{H * args.factor} "
          f"({args.num_inference_steps} steps)...")
    sr = sr_upscale_rgbd(rgbd, model, scheduler,
                         num_inference_steps=args.num_inference_steps,
                         factor=args.factor, device=args.device)

    stem = Path(args.input).stem + f"_sr{args.factor}x"
    rgb_path, depth_path, pt_path = save_rgbd_outputs(sr, args.output_dir, stem)
    print("\nSuccess! Upscaled outputs saved to:")
    print(f" - {rgb_path}")
    print(f" - {depth_path}")
    print(f" - {pt_path}")


if __name__ == "__main__":
    main()
