import os
import argparse
import torch
import numpy as np
from diffusers import DDPMScheduler
from pathlib import Path

from dreamsea.fractal_latent import diamond_square_2d
from dreamsea.generation_inpainting import GeneratorInpainter
from dreamsea.models import UnconditionalDDPM
import dreamsea.gs_sds_optimization as gs_opt

def generate_3dgs(cond_ckpt, uncond_ckpt, grid_size=3, roughness=0.5, 
                  output_dir="outputs", sds_iterations=500, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Full pipeline to generate a 3DGS scene from trained checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Generator with both checkpoints
    print(f"\n--- 1. Initializing Generator ---")
    print(f"Loading Conditional Checkpoint: {cond_ckpt}")
    print(f"Loading Unconditional Checkpoint: {uncond_ckpt}")
    generator = GeneratorInpainter(
        cond_model_path=cond_ckpt,
        uncond_model_path=uncond_ckpt,
        device=device
    )

    # 2. Generate Fractal Latent Grid
    print(f"\n--- 2. Generating Fractal Latent Grid (size: {grid_size}x{grid_size}) ---")
    latent_grid = diamond_square_2d(grid_size, roughness=roughness)
    print(f"Latent grid generated.")

    # 3. Generate and Stitch RGBD Patches
    print("\n--- 3. Generating and Stitching RGBD Patches ---")
    # This uses the conditional model to generate patches and unconditional to inpaint seams
    patch_grid = generator.generate_grid(latent_grid)
    global_map = generator.stitch_and_inpaint(patch_grid, overlap_size=32)
    
    # Save the global RGBD map for inspection
    map_path = os.path.join(output_dir, "global_rgbd_map.pt")
    torch.save(torch.from_numpy(global_map), map_path)
    print(f"Global RGBD Map saved to: {map_path}")

    # 4. Initialize 3D Gaussian Splatting
    print("\n--- 4. Initializing 3D Gaussian Splatting ---")
    positions, colors = gs_opt.create_point_cloud_from_rgbd(global_map)
    print(f"Extracted {positions.shape[0]} points from map.")

    if positions.shape[0] == 0:
        print("Error: No valid points found in RGBD map. Aborting.")
        return

    gs_model = gs_opt.GaussianSplattingModel(positions, colors, device=device)
    print("3DGS model initialized.")

    # 5. SDS Optimization
    print(f"\n--- 5. Optimizing 3DGS via SDS ({sds_iterations} iterations) ---")
    # Note: SDS uses the Unconditional model as a prior to refine the appearance
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # We use the unconditional model loaded in the generator for SDS
    gs_opt.optimize_3dgs_sds(
        model=gs_model, 
        diffusion_model=generator.uncond_model, 
        scheduler=scheduler, 
        iterations=sds_iterations
    )

    # 6. Save Results
    # We save a dictionary that includes positions so the .pt can be converted to .ply
    final_path = os.path.join(output_dir, "final_gs_model.pt")
    save_dict = gs_model.state_dict()
    save_dict['positions'] = gs_model.positions.cpu() # Add positions to the save file
    
    torch.save(save_dict, final_path)
    print(f"\nSuccess! 3DGS generation complete.")
    print(f"Final model saved to: {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3DGS scene from trained DreamSea checkpoints.")
    parser.add_argument("--cond_ckpt", type=str, required=True, help="Path to the conditional DDPM checkpoint.")
    parser.add_argument("--uncond_ckpt", type=str, required=True, help="Path to the unconditional DDPM checkpoint.")
    parser.add_argument("--grid_size", type=int, default=3, help="Latent grid size (must be 2^n + 1, e.g., 3, 5, 9).")
    parser.add_argument("--roughness", type=float, default=0.5, help="Fractal roughness.")
    parser.add_argument("--sds_iterations", type=int, default=100, help="Number of SDS optimization steps.")
    parser.add_argument("--output_dir", type=str, default="outputs/3dgs_gen", help="Directory to save output files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")

    args = parser.parse_args()

    generate_3dgs(
        cond_ckpt=args.cond_ckpt,
        uncond_ckpt=args.uncond_ckpt,
        grid_size=args.grid_size,
        roughness=args.roughness,
        sds_iterations=args.sds_iterations,
        output_dir=args.output_dir,
        device=args.device
    )
