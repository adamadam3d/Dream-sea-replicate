import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from dreamsea.generation_inpainting import GeneratorInpainter

def normalize_tensor_to_image(tensor):
    """
    Converts a PyTorch tensor in range [-1, 1] into a valid [0, 255] uint8 numpy image.
    """
    # Detach and move to CPU
    image_np = tensor.detach().cpu().numpy()
    
    # Map from [-1, 1] to [0, 1]
    image_np = (image_np + 1.0) / 2.0
    
    # Clip to ensure valid range [0, 1]
    image_np = np.clip(image_np, 0.0, 1.0)
        
    # Scale to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def main():
    parser = argparse.ArgumentParser(description="Generate and save an image using trained checkpoints.")
    parser.add_argument("-c", "--cond_model_path", type=str, default=None, help="Path to conditional DDPM checkpoint (.pt file).")
    parser.add_argument("-o", "--output_dir", type=str, default="samples", help="Directory to save the generated images.")
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("-l", "--latent_vector", type=str, default="0.5,-0.5", help="Comma-separated latent vector (e.g. '0.0,0.0').")
    parser.add_argument("-r", "--no_random", action="store_true", help="If set, do not add noise to the latent vector between samples.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("-s", "--num_inference_steps", type=int, default=1000, help="Number of denoising steps (higher = better quality but slower).")
    args = parser.parse_args()

    # Parse latent vector
    try:
        latent_base = np.array([float(x) for x in args.latent_vector.split(',')], dtype=np.float32)
        if latent_base.shape[0] != 2:
            raise ValueError("Latent vector must have exactly 2 components.")
    except Exception as e:
        print(f"Invalid latent vector format: {e}. Using default [0.5, -0.5]")
        latent_base = np.array([0.5, -0.5], dtype=np.float32)

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

    if args.cond_model_path:
        print(f"Using checkpoint: {args.cond_model_path}")
    else:
        print("WARNING: No checkpoint provided. Generating with RANDOM UNTRAINED weights.")

    all_rgb_images = []
    all_depth_images = []

    for i in range(args.num_samples):
        print(f"\n--- Generating Sample Image {i+1}/{args.num_samples} ---")
        # 1. Prepare latent condition
        latent_condition = latent_base.copy()
        if args.num_samples > 1 and not args.no_random:
             # Randomize latent slightly to see different variants
             latent_condition += np.random.normal(0, 0.2, size=2).astype(np.float32)
             
        print(f"Using latent condition vector: {latent_condition}")

        # 2. Generate the patch
        print(f"Running diffusion generation (this takes ~{args.num_inference_steps} steps)...")
        try:
            patch = inpainter.generate_patch(latent_condition, num_inference_steps=args.num_inference_steps) # Shape: (1, 4, 224, 224)
        except Exception as e:
             print(f"Error during generation: {e}")
             continue

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

        # Store for collage
        all_rgb_images.append(rgb_pil)
        all_depth_images.append(depth_pil)

        # 5. Save the images
        rgb_path = output_dir / f"generated_rgb_{i+1}.png"
        depth_path = output_dir / f"generated_depth_{i+1}.png"
        
        rgb_pil.save(rgb_path)
        depth_pil.save(depth_path)

        print(f"Success! Sample {i+1} saved to:")
        print(f" - {rgb_path}")
        print(f" - {depth_path}")

    # --- Create Collage ---
    if len(all_rgb_images) > 0:
        print("\n--- Creating Collages ---")
        cols = 5
        rows = (len(all_rgb_images) + cols - 1) // cols
        w, h = all_rgb_images[0].size
        
        collage_rgb = Image.new('RGB', (cols * w, rows * h))
        collage_depth = Image.new('L', (cols * w, rows * h))
        
        for idx, (rgb, depth) in enumerate(zip(all_rgb_images, all_depth_images)):
            r = idx // cols
            c = idx % cols
            collage_rgb.paste(rgb, (c * w, r * h))
            collage_depth.paste(depth, (c * w, r * h))
            
        rgb_collage_path = output_dir / "collage_rgb.png"
        depth_collage_path = output_dir / "collage_depth.png"
        
        collage_rgb.save(rgb_collage_path)
        collage_depth.save(depth_collage_path)
        
        print(f"Collages saved to:")
        print(f" - {rgb_collage_path}")
        print(f" - {depth_collage_path}")

    print(f"\nAll {args.num_samples} samples complete!")

if __name__ == "__main__":
    main()