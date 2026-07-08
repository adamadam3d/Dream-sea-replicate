import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import DDPMScheduler
from dreamsea.models import UnconditionalDDPM
from dreamsea.redilation import ReDilation

def normalize_tensor_to_image(tensor):
    """
    Converts a PyTorch tensor in range [-1, 1] into a valid [0, 255] uint8 numpy image.
    """
    image_np = tensor.detach().cpu().numpy()
    image_np = (image_np + 1.0) / 2.0
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def generate_unconditional(model_path, output_dir, num_inference_steps=1000, device='cuda', patch_size=224,
                           redilate=False, redilation_fraction=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Unconditional Checkpoint from: {model_path} onto {device}...")
    model = UnconditionalDDPM().to(device)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Prefer EMA weights when present — cleaner samples than raw weights
        if isinstance(checkpoint, dict) and 'ema_model_state_dict' in checkpoint:
            state_dict = checkpoint['ema_model_state_dict']
            print("Using EMA weights from checkpoint.")
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    print(f"Generating random RGBD patch (this takes ~{num_inference_steps} steps)...")
    
    # Start from pure noise
    image = torch.randn(1, 4, patch_size, patch_size, device=device)

    timesteps = scheduler.timesteps
    redila = None
    n_dilated_steps = 0
    if redilate and patch_size > 224:
        redila = ReDilation(model, patch_size / 224)
        n_dilated_steps = int(len(timesteps) * redilation_fraction)
        print(f"Re-dilation x{redila.scale} active for the first "
              f"{n_dilated_steps}/{len(timesteps)} steps.")

    try:
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if redila is not None:
                    if i < n_dilated_steps:
                        redila.enable()
                    else:
                        redila.disable()
                # Unconditional prediction (no text/conditions)
                noise_pred = model(image, t)
                # Denoise step
                image = scheduler.step(noise_pred, t, image).prev_sample
    finally:
        if redila is not None:
            redila.disable()

    # Extract RGB and Depth
    patch_tensor = image[0] # (4, 224, 224)
    rgb_tensor = patch_tensor[:3, :, :]
    depth_tensor = patch_tensor[3, :, :]

    # Convert to Images
    rgb_img_np = normalize_tensor_to_image(rgb_tensor)
    rgb_img_np = np.transpose(rgb_img_np, (1, 2, 0)) # CHW to HWC
    rgb_pil = Image.fromarray(rgb_img_np, mode='RGB')

    depth_img_np = normalize_tensor_to_image(depth_tensor)
    depth_pil = Image.fromarray(depth_img_np, mode='L')

    # Save
    rgb_path = output_dir / "uncond_generated_rgb.png"
    depth_path = output_dir / "uncond_generated_depth.png"
    
    rgb_pil.save(rgb_path)
    depth_pil.save(depth_path)

    print("\nSuccess! Visualizations saved to:")
    print(f" - {rgb_path} (Color Image)")
    print(f" - {depth_path} (Depth Map)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save an image using an UNCONDITIONAL trained checkpoint.")
    parser.add_argument("-u", "--uncond_model_path", type=str, required=True, help="Path to unconditional DDPM checkpoint (.pt file).")
    parser.add_argument("-o", "--output_dir", type=str, default="samples", help="Directory to save the generated images.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=1000, help="Number of denoising steps.")
    parser.add_argument("-p", "--patch_size", type=int, default=896,
                        help="Output resolution (square, must be divisible by 32). The model was "
                             "trained at 224 and is being sampled off-distribution at this size to "
                             "test whether the fully-convolutional UNet generalizes to a larger "
                             "single-patch output. Default 896.")
    parser.add_argument("--redilate", action="store_true",
                        help="Apply ScaleCrafter re-dilation when patch_size > 224: scales the convs' "
                             "receptive field to the larger canvas during the early denoising steps to "
                             "prevent repeated-tile artifacts. Off by default.")
    parser.add_argument("--redilation_fraction", type=float, default=0.5,
                        help="Fraction of denoising steps (from the noisiest) that run with dilated "
                             "convs. Higher = more coherent global layout, lower = sharper texture.")
    args = parser.parse_args()

    generate_unconditional(
        model_path=args.uncond_model_path,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        patch_size=args.patch_size,
        redilate=args.redilate,
        redilation_fraction=args.redilation_fraction
    )
