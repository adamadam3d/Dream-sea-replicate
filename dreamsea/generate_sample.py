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
    parser.add_argument("--sr_ckpt", type=str, default=None,
                        help="Optional trained SR-stage checkpoint. When given, each generated patch "
                             "is additionally x4-upscaled through the SR DDPM (cascade: use -p 224 "
                             "for the base patch, giving 896 output).")
    parser.add_argument("--sr_steps", type=int, default=100,
                        help="Denoising steps for the SR stage (only with --sr_ckpt).")
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

    sr_model = None
    sr_scheduler = None
    if args.sr_ckpt:
        from diffusers import DDPMScheduler
        from dreamsea.sr_upscale import load_sr_model, sr_upscale_rgbd, save_rgbd_outputs
        print(f"Loading SR stage from: {args.sr_ckpt}")
        sr_model = load_sr_model(args.sr_ckpt, args.device)
        sr_scheduler = DDPMScheduler(num_train_timesteps=1000)

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
            patch = inpainter.generate_patch(latent_condition, num_inference_steps=args.num_inference_steps,
                                              patch_size=args.patch_size,
                                              redilate=args.redilate,
                                              redilation_fraction=args.redilation_fraction) # Shape: (1, 4, patch_size, patch_size)
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

        # Cascade stage 2: x4 SR of the freshly generated patch
        if sr_model is not None:
            print(f"Running x4 SR stage ({args.sr_steps} steps)...")
            sr = sr_upscale_rgbd(patch_tensor, sr_model, sr_scheduler,
                                 num_inference_steps=args.sr_steps, device=args.device)
            sr_rgb, sr_depth, _ = save_rgbd_outputs(sr, output_dir, f"generated_{i+1}_sr4x")
            print(f" - {sr_rgb}")
            print(f" - {sr_depth}")

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