import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dreamsea.generation_inpainting import GeneratorInpainter

def normalize_tensor_to_image(tensor):
    """
    Converts a PyTorch tensor in range [-1, 1] or roughly Normal(0, 1) 
    into a valid [0, 255] uint8 numpy image.
    """
    # Detach and move to CPU
    image_np = tensor.detach().cpu().numpy()
    
    # Simple min-max normalization to [0, 1] for visualization
    img_min = image_np.min()
    img_max = image_np.max()
    
    if img_max > img_min:
        image_np = (image_np - img_min) / (img_max - img_min)
    else:
        image_np = np.zeros_like(image_np)
        
    # Scale to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def main():
    parser = argparse.ArgumentParser(description="Generate and save an image using trained checkpoints.")
    parser.add_argument("--cond_model_path", type=str, default=None, help="Path to conditional DDPM checkpoint (.pt file).")
    parser.add_argument("--output_dir", type=str, default="samples", help="Directory to save the generated images.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--num_inference_steps", type=int, default=250, help="Number of denoising steps (higher = better quality but slower).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing Generator on {args.device}...")
    try:
        inpainter = GeneratorInpainter(
            cond_model_path=args.cond_model_path,
            uncond_model_path=None, # We only need the conditional model to generate a raw patch
            device=args.device
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    print("\n--- Generating Sample Image ---")
    if args.cond_model_path:
        print(f"Using checkpoint: {args.cond_model_path}")
    else:
        print("WARNING: No checkpoint provided. Generating with RANDOM UNTRAINED weights.")

    # 1. Create a dummy latent condition (simulating DINOv2 feature)
    # Different numbers will theoretically produce different "styles" of underwater scenes
    latent_condition = np.array([0.5, -0.5], dtype=np.float32)
    print(f"Using latent condition vector: {latent_condition}")

    # 2. Generate the patch
    print(f"Running diffusion generation (this takes ~{args.num_inference_steps} steps)...")
    try:
        patch = inpainter.generate_patch(latent_condition, num_inference_steps=args.num_inference_steps) # Shape: (1, 4, 224, 224)
    except Exception as e:
         print(f"Error during generation: {e}")
         return

    # 3. Extract RGB and Depth
    # The output is (1, 4, 224, 224). 
    # Channels 0,1,2 are RGB. Channel 3 is Depth.
    patch_tensor = torch.from_numpy(patch[0]) # (4, 224, 224)
    
    rgb_tensor = patch_tensor[:3, :, :] # (3, 224, 224)
    depth_tensor = patch_tensor[3, :, :] # (224, 224)

    # 4. Normalize and Convert to PIL Images
    rgb_img_np = normalize_tensor_to_image(rgb_tensor) # (3, 224, 224)
    rgb_img_np = np.transpose(rgb_img_np, (1, 2, 0)) # CHW to HWC for PIL (224, 224, 3)
    rgb_pil = Image.fromarray(rgb_img_np, mode='RGB')

    depth_img_np = normalize_tensor_to_image(depth_tensor) # (224, 224)
    depth_pil = Image.fromarray(depth_img_np, mode='L') # 'L' mode is for grayscale

    # 5. Save the images
    rgb_path = output_dir / "generated_rgb.png"
    depth_path = output_dir / "generated_depth.png"
    
    rgb_pil.save(rgb_path)
    depth_pil.save(depth_path)

    print("\nSuccess! Visualizations saved to:")
    print(f" - {rgb_path} (Color Image)")
    print(f" - {depth_path} (Depth Map)")

if __name__ == "__main__":
    main()