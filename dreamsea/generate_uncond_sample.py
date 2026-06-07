import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import DDPMScheduler
from dreamsea.models import UnconditionalDDPM

def normalize_tensor_to_image(tensor):
    """
    Converts a PyTorch tensor in range [-1, 1] into a valid [0, 255] uint8 numpy image.
    """
    image_np = tensor.detach().cpu().numpy()
    image_np = (image_np + 1.0) / 2.0
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def generate_unconditional(model_path, output_dir, num_inference_steps=1000, device='cuda'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Unconditional Checkpoint from: {model_path} onto {device}...")
    model = UnconditionalDDPM().to(device)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
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
    image = torch.randn(1, 4, 224, 224, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            # Unconditional prediction (no text/conditions)
            noise_pred = model(image, t)
            # Denoise step
            image = scheduler.step(noise_pred, t, image).prev_sample

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
    args = parser.parse_args()

    generate_unconditional(
        model_path=args.uncond_model_path,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        device=args.device
    )
