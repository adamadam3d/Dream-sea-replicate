import torch
import numpy as np
from diffusers import DDPMScheduler

from dreamsea.data_preprocessing import DataPreprocessor
from dreamsea.fractal_latent import diamond_square_2d
from dreamsea.generation_inpainting import GeneratorInpainter
from dreamsea.models import UnconditionalDDPM

import dreamsea.gs_sds_optimization as gs_opt

def main():
    print("Running DreamSea End-to-End Pipeline Prototype...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 1: Preprocessing components (assuming mock inputs for pipeline test)
    print("\n--- 1. Testing Preprocessing Components ---")
    # For a real run we'd use process_rgb_to_rgbd with real images,
    # but here we'll just mock the output of preprocessing.
    preprocessor = DataPreprocessor(device=device)
    print("Foundation models loaded.")

    # Step 2: Generate Fractal Latent Grid
    print("\n--- 2. Generating Fractal Latent Grid ---")
    grid_size = 5 # 2^2 + 1
    latent_grid = diamond_square_2d(grid_size, roughness=0.5, seed=42)
    print(f"Latent grid shape: {latent_grid.shape}")

    # Step 3: Generation and Stitching
    print("\n--- 3. Generating and Inpainting RGBD Patches ---")
    generator = GeneratorInpainter(device=device)

    # We will generate a very small grid to save time in the test
    small_latent_grid = diamond_square_2d(3, roughness=0.5, seed=42)
    patch_grid = generator.generate_grid(small_latent_grid)
    print(f"Patch grid shape: {patch_grid.shape}")

    global_map = generator.stitch_and_inpaint(patch_grid, overlap_size=32)
    print(f"Global RGBD Map shape: {global_map.shape}")

    # Step 4: 3D Gaussian Splatting Initialization
    print("\n--- 4. Initializing 3D Gaussian Splatting ---")
    positions, colors, _ = gs_opt.create_point_cloud_from_rgbd(global_map)
    print(f"Extracted {positions.shape[0]} valid points from RGBD map.")

    # Only initialize if points were found
    if positions.shape[0] > 0:
        gs_model = gs_opt.GaussianSplattingModel(positions, colors)
        print("3DGS model initialized.")

        # Step 5: SDS Optimization
        print("\n--- 5. SDS Optimization (Dummy 5 iterations) ---")
        uncond_model = UnconditionalDDPM(in_channels=4, out_channels=4).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        gs_opt.optimize_3dgs_sds(gs_model, uncond_model, scheduler, iterations=5)

    print("\nDreamSea pipeline test complete!")

if __name__ == "__main__":
    main()
