"""
Standalone test script to verify and visualize image generation conditioned on
interpolated DINO embeddings, reproducing the concept in Figure 9 of the paper
("Examples of image generation conditioned on interpolated DINO embeddings. A smooth transition can be observed.").

This script allows:
1. Interpolating directly between two 2D PCA latent points (representing the PCA-reduced DINOv2 conditioning).
2. If two real images and a fitted PCA model are provided, extracting their DINOv2 features,
   interpolating them in high-dimensional or 2D PCA space, and generating the transition sequence.

To isolate the visual effect of the conditioning embedding and ensure a smooth layout transition,
we reset the random seed before generating each patch, ensuring the initial noise tensor is identical.

Usage:
  PYTHONPATH=. python dreamsea/test_dino_interpolation.py \
      --cond_model_path checkpoints/conditional_epoch_2000.pt \
      --steps 5 \
      --start_latent "1.0,-1.0" \
      --end_latent "-1.0,1.0"
"""

import argparse
import os
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import joblib

from dreamsea.generation_inpainting import GeneratorInpainter
from dreamsea.data_preprocessing import DataPreprocessor

def normalize_tensor_to_image(tensor):
    """
    Converts a PyTorch tensor in range [-1, 1] into a valid [0, 255] uint8 numpy image.
    """
    image_np = tensor.detach().cpu().numpy()
    # Map from [-1, 1] to [0, 1]
    image_np = (image_np + 1.0) / 2.0
    # Clip to ensure valid range [0, 1]
    image_np = np.clip(image_np, 0.0, 1.0)
    # Scale to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Generate images conditioned on interpolated DINO embeddings.")
    parser.add_argument("--cond_model_path", type=str, default=None,
                        help="Path to conditional DDPM checkpoint (.pt file). If None, runs with random weights.")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of interpolation steps (minimum 2).")
    parser.add_argument("--start_latent", type=str, default="1.0,-1.0",
                        help="Starting 2D latent condition (comma-separated, e.g. '1.0,-1.0'). Used if images are not provided.")
    parser.add_argument("--end_latent", type=str, default="-1.0,1.0",
                        help="Ending 2D latent condition (comma-separated, e.g. '-1.0,1.0'). Used if images are not provided.")
    parser.add_argument("--num_inference_steps", type=int, default=250,
                        help="Number of diffusion denoising steps.")
    parser.add_argument("--output_dir", type=str, default="samples/dino_interpolation",
                        help="Directory to save output images.")
    parser.add_argument("--image_a", type=str, default=None,
                        help="Optional path to starting real RGB image.")
    parser.add_argument("--image_b", type=str, default=None,
                        help="Optional path to ending real RGB image.")
    parser.add_argument("--pca_model_path", type=str, default=None,
                        help="Optional path to fitted PCA model (.pkl file) used when real images are provided.")
    parser.add_argument("--interpolate_high_dim", type=bool, default=True,
                        help="If True, interpolates in 1024-dim DINOv2 space before PCA. If False, interpolates in 2D PCA space.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the initial noise.")
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("Number of steps must be at least 2.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    print(f"Initializing GeneratorInpainter on {args.device}...")
    try:
        inpainter = GeneratorInpainter(
            cond_model_path=args.cond_model_path,
            uncond_model_path=None,
            device=args.device
        )
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    if args.cond_model_path:
        print(f"Loaded conditional checkpoint: {args.cond_model_path}")
    else:
        print("WARNING: No checkpoint provided. Running with RANDOM UNTRAINED weights.")

    # Determine endpoint latent vectors
    latent_steps = []
    
    if args.image_a and args.image_b:
        print("\n--- Processing Endpoint Images ---")
        if not Path(args.image_a).exists() or not Path(args.image_b).exists():
            print(f"Error: One or both of the specified images do not exist:\n  - {args.image_a}\n  - {args.image_b}")
            return
        if not args.pca_model_path or not Path(args.pca_model_path).exists():
            print(f"Error: --pca_model_path must be specified and exist when using real images. Provided: {args.pca_model_path}")
            return

        print("Loading preprocessor and extracting DINOv2 features...")
        preprocessor = DataPreprocessor(device=args.device)
        
        # Extract features for both images
        # extract_and_reduce_dino_features expects a list of paths
        # We need to temporarily mock/disable PCA fitting inside the method, or extract manually.
        # Let's extract manually to avoid fitting a new PCA on 2 images, and instead load the fitted PCA.
        inputs_a = preprocessor.dino_processor(images=Image.open(args.image_a).convert("RGB"), return_tensors="pt").to(args.device)
        inputs_b = preprocessor.dino_processor(images=Image.open(args.image_b).convert("RGB"), return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            feat_a = preprocessor.dino_model(**inputs_a).last_hidden_state[0, 0, :].cpu().numpy()
            feat_b = preprocessor.dino_model(**inputs_b).last_hidden_state[0, 0, :].cpu().numpy()

        print(f"Successfully extracted DINOv2 features of shape {feat_a.shape}")
        
        # Load PCA model
        print(f"Loading PCA model from: {args.pca_model_path}")
        pca = joblib.load(args.pca_model_path)
        
        if args.interpolate_high_dim:
            print("Interpolating features in high-dimensional DINOv2 space...")
            for i in range(args.steps):
                t = i / (args.steps - 1)
                # Linear interpolation in DINO space
                interp_feat = (1.0 - t) * feat_a + t * feat_b
                # Project to 2D
                latent_2d = pca.transform(interp_feat.reshape(1, -1))[0]
                latent_steps.append(latent_2d)
        else:
            print("Projecting endpoints to 2D PCA space and interpolating...")
            latent_a = pca.transform(feat_a.reshape(1, -1))[0]
            latent_b = pca.transform(feat_b.reshape(1, -1))[0]
            print(f"Image A 2D PCA: {latent_a}")
            print(f"Image B 2D PCA: {latent_b}")
            for i in range(args.steps):
                t = i / (args.steps - 1)
                latent_2d = (1.0 - t) * latent_a + t * latent_b
                latent_steps.append(latent_2d)
    else:
        print("\n--- Preparing Latent Steps from Coordinates ---")
        try:
            start_vec = np.array([float(x) for x in args.start_latent.split(',')], dtype=np.float32)
            end_vec = np.array([float(x) for x in args.end_latent.split(',')], dtype=np.float32)
            if start_vec.shape[0] != 2 or end_vec.shape[0] != 2:
                raise ValueError("Vectors must be 2D.")
        except Exception as e:
            print(f"Error parsing latent coordinates: {e}. Using defaults [1.0, -1.0] -> [-1.0, 1.0]")
            start_vec = np.array([1.0, -1.0], dtype=np.float32)
            end_vec = np.array([-1.0, 1.0], dtype=np.float32)

        print(f"Interpolating between: {start_vec} and {end_vec}")
        for i in range(args.steps):
            t = i / (args.steps - 1)
            latent_2d = (1.0 - t) * start_vec + t * end_vec
            latent_steps.append(latent_2d)

    # Convert to numpy arrays of float32
    latent_steps = [np.array(x, dtype=np.float32) for x in latent_steps]

    # --- Generate Patches ---
    all_rgb_images = []
    all_depth_images = []
    generated_patches = []

    print(f"\n--- Generating {args.steps} Interpolation Steps ---")
    for i, latent_cond in enumerate(latent_steps):
        print(f"Step {i+1}/{args.steps}: Latent condition = {latent_cond}")
        
        # Crucial: Reset seed before each generation to ensure starting noise is IDENTICAL.
        # This isolates the effect of the conditioning vector and guarantees a smooth visual morphing.
        set_seed(args.seed)
        
        try:
            patch = inpainter.generate_patch(latent_cond, num_inference_steps=args.num_inference_steps) # (1, 4, 224, 224)
            generated_patches.append(patch[0])
            
            # Extract RGB and Depth
            patch_tensor = torch.from_numpy(patch[0])
            rgb_tensor = patch_tensor[:3, :, :]
            depth_tensor = patch_tensor[3, :, :]
            
            # Normalize and convert to PIL
            rgb_np = normalize_tensor_to_image(rgb_tensor)
            rgb_np = np.transpose(rgb_np, (1, 2, 0)) # CHW to HWC
            rgb_pil = Image.fromarray(rgb_np, mode='RGB')
            
            depth_np = normalize_tensor_to_image(depth_tensor)
            depth_pil = Image.fromarray(depth_np, mode='L')
            
            all_rgb_images.append(rgb_pil)
            all_depth_images.append(depth_pil)
            
            # Save individual steps
            rgb_path = output_dir / f"step_{i}_rgb.png"
            depth_path = output_dir / f"step_{i}_depth.png"
            rgb_pil.save(rgb_path)
            depth_pil.save(depth_path)
            
        except Exception as e:
            print(f"Error generating step {i}: {e}")
            return

    # --- Save Collages ---
    print("\n--- Saving Collages ---")
    w, h = all_rgb_images[0].size

    # Create a combined collage: RGB on top, red-blue depth below, for each step
    collage = Image.new('RGB', (args.steps * w, h * 2))

    for i, (rgb, depth) in enumerate(zip(all_rgb_images, all_depth_images)):
        # Paste RGB on the top row
        collage.paste(rgb, (i * w, 0))

        # Convert grayscale depth to red-blue colormap:
        # Near (high depth value) = red, Far (low depth value) = blue
        depth_np = np.array(depth, dtype=np.float32) / 255.0  # normalize to [0, 1]
        r = (depth_np * 255).astype(np.uint8)
        g = np.zeros_like(r)
        b = ((1.0 - depth_np) * 255).astype(np.uint8)
        depth_colored = Image.fromarray(np.stack([r, g, b], axis=-1), mode='RGB')

        # Paste depth on the bottom row
        collage.paste(depth_colored, (i * w, h))

    collage_path = output_dir / "collage_rgb_depth.png"
    collage.save(collage_path)

    print(f"Combined collage saved to: {collage_path}")

    # --- Automated Evaluation Checks ---
    print("\n================ EVALUATION CHECKS ================")
    
    # 1. Output Range Check
    flat_patches = np.array(generated_patches)
    p_min, p_max = flat_patches.min(), flat_patches.max()
    print(f"Generated patch value range: [{p_min:.4f}, {p_max:.4f}] (Expected: close to [-1, 1])")
    
    # 2. Check for identical outputs (Conditioning responsiveness)
    # If the conditioning has no effect or is ignored, the outputs will be identical.
    differences = []
    for i in range(args.steps - 1):
        diff = np.abs(generated_patches[i+1] - generated_patches[i]).mean()
        differences.append(diff)
    
    mean_diff = np.mean(differences)
    print(f"Mean pixel difference between adjacent steps: {mean_diff:.6f}")
    
    # 3. Check for smooth transitions
    # If it is smooth, the step-to-step changes should be relatively small and uniform,
    # rather than jumping erratically.
    diff_variance = np.var(differences) if len(differences) > 1 else 0.0
    print(f"Variance of step differences: {diff_variance:.6f} (Low variance indicates a uniform/smooth transition)")

    cond_active = mean_diff > 1e-4
    if args.cond_model_path:
        if cond_active:
            print("\n[PASS] Model responds to different DINO/PCA conditioning vectors!")
        else:
            print("\n[FAIL] Model output is identical/nearly identical across different conditioning vectors.")
            print("       This means the cross-attention conditioning is ignored or not working.")
    else:
        print("\n[INFO] Running on untrained/random weights. The model architecture output differs between")
        print("       conditioning steps due to initial cross-attention projections, but a trained model")
        print("       is required for semantic shifts (e.g., from sand to coral reef).")
        
    print("\nTest completed!")

if __name__ == "__main__":
    main()
