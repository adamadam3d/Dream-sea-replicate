import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path

from dreamsea.fractal_latent import diamond_square_2d, scale_latent_grid
from dreamsea.generation_inpainting import GeneratorInpainter
import dreamsea.gs_sds_optimization as gs_opt

# Default latent-space statistics: 2-component PCA of DINOv2 features from a
# representative preprocessed dataset. Used to calibrate the fractal grid into the
# training PCA range when no --latent_stats_path is given. Override by passing a
# latent_stats.json from your own preprocess_dataset.py run.
DEFAULT_LATENT_STATS = {
    "min":  [-25.790103912353516, -16.195302963256836],
    "max":  [ 26.72313690185547,   30.283246994018555],
    "mean": [-8.705721938895294e-07, 2.0599543404387077e-06],
    "std":  [ 14.020397186279297,   8.373991966247559],
}

def generate_3dgs(cond_ckpt, uncond_ckpt, grid_size=3, roughness=0.5,
                  output_dir="outputs", sds_iterations=500,
                  latent_stats_path=None, use_conditional_stitching=False,
                  rasterizer="gsplat",
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Full pipeline to generate a 3DGS scene from trained checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)

    # The unconditional model is only used for (a) seam inpainting when NOT using
    # conditional stitching, and (b) SDS optimization. Skip requiring/loading it
    # otherwise so a run that needs only the conditional model isn't blocked.
    uncond_needed = (not use_conditional_stitching) or (sds_iterations > 0)
    if uncond_needed and not uncond_ckpt:
        reasons = []
        if not use_conditional_stitching:
            reasons.append("seam inpainting uses the unconditional model "
                           "(add --use_conditional_stitching to inpaint seams with the conditional model)")
        if sds_iterations > 0:
            reasons.append("SDS optimization uses the unconditional model "
                           "(set --sds_iterations 0 to skip it)")
        raise ValueError("--uncond_ckpt is required here because " + "; and ".join(reasons) + ".")
    # Only load the unconditional model when it will actually be used; this avoids
    # blocking a conditional-only run (and avoids a spurious load failure if an
    # unused/incorrect path was passed).
    uncond_to_load = uncond_ckpt if uncond_needed else None
    if uncond_ckpt and not uncond_needed:
        print("Note: --uncond_ckpt is not needed for this run (conditional stitching + no SDS); "
              "skipping it.")

    # 1. Initialize Generator
    print(f"\n--- 1. Initializing Generator ---")
    print(f"Loading Conditional Checkpoint: {cond_ckpt}")
    print(f"Loading Unconditional Checkpoint: {uncond_to_load if uncond_to_load else '(none — not needed)'}")
    generator = GeneratorInpainter(
        cond_model_path=cond_ckpt,
        uncond_model_path=uncond_to_load,
        device=device
    )

    # 2. Generate Fractal Latent Grid
    print(f"\n--- 2. Generating Fractal Latent Grid (size: {grid_size}x{grid_size}) ---")
    latent_grid = diamond_square_2d(grid_size, roughness=roughness)

    # Rescale the grid into the PCA coordinate range seen during training so the
    # conditional model receives in-distribution latent vectors. Use an explicit
    # stats file if given, otherwise fall back to the built-in DEFAULT_LATENT_STATS.
    if latent_stats_path and os.path.exists(latent_stats_path):
        with open(latent_stats_path) as f:
            latent_stats = json.load(f)
        stats_source = latent_stats_path
    else:
        latent_stats = DEFAULT_LATENT_STATS
        stats_source = "built-in DEFAULT_LATENT_STATS"
        if latent_stats_path:
            print(f"WARNING: latent_stats_path '{latent_stats_path}' not found; "
                  f"falling back to built-in default latent stats.")

    if "mean" in latent_stats and "std" in latent_stats:
        latent_grid = scale_latent_grid(latent_grid, latent_stats["mean"], latent_stats["std"])
    else:
        # Older stats files stored only min/max — approximate a Gaussian fit
        # (mean = midpoint, std ~= range/4) so the fixed affine map still works.
        lo = np.array(latent_stats["min"], dtype=np.float32)
        hi = np.array(latent_stats["max"], dtype=np.float32)
        latent_grid = scale_latent_grid(latent_grid, (lo + hi) / 2.0, (hi - lo) / 4.0)
    print(f"Latent grid mapped into training PCA distribution using: {stats_source}")

    print(f"Latent grid generated.")

    # 3. Generate and Stitch RGBD Patches
    print("\n--- 3. Generating and Stitching RGBD Patches ---")
    # This uses the conditional model to generate patches and unconditional to inpaint seams
    patch_grid = generator.generate_grid(latent_grid)
    global_map = generator.stitch_and_inpaint(
        patch_grid, 
        overlap_size=32, 
        latent_grid=latent_grid, 
        use_conditional=use_conditional_stitching
    )
    
    # Save the global RGBD map for inspection
    map_path = os.path.join(output_dir, "global_rgbd_map.pt")
    torch.save(torch.from_numpy(global_map), map_path)
    print(f"Global RGBD Map saved to: {map_path}")

    # Save the global RGB map as a PNG image
    from PIL import Image
    rgb_map_path = os.path.join(output_dir, "global_rgb_map.png")
    rgb_map = global_map[:3, :, :] # Extract RGB channels
    rgb_img_np = (rgb_map + 1.0) / 2.0 # Normalize from [-1, 1] to [0, 1]
    rgb_img_np = np.clip(rgb_img_np, 0.0, 1.0)
    rgb_img_np = (rgb_img_np * 255.0).astype(np.uint8) # Convert to uint8
    rgb_img_np = np.transpose(rgb_img_np, (1, 2, 0)) # CHW -> HWC for PIL
    Image.fromarray(rgb_img_np).save(rgb_map_path)
    print(f"Global RGB Map image saved to: {rgb_map_path}")

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
    if sds_iterations > 0:
        print(f"\n--- 5. Optimizing 3DGS via SDS ({sds_iterations} iterations, "
              f"rasterizer={rasterizer}) ---")
        if rasterizer == "gsplat":
            # Faithful multi-view SDS with a real differentiable Gaussian rasterizer.
            from dreamsea.gs_sds_optimization_v2 import optimize_3dgs_sds_multiview
            optimize_3dgs_sds_multiview(
                model=gs_model,
                diffusion_model=generator.uncond_model,
                scheduler=generator.scheduler,
                iterations=sds_iterations,
            )
        else:
            # Simplified single-view scatter renderer (color/opacity only).
            gs_opt.optimize_3dgs_sds(
                model=gs_model,
                diffusion_model=generator.uncond_model,
                scheduler=generator.scheduler,
                iterations=sds_iterations
            )
    else:
        print("\n--- 5. Skipping SDS optimization (--sds_iterations 0) ---")

    # 6. Save Results
    # We save a dictionary that includes positions so the .pt can be converted to .ply
    final_path = os.path.join(output_dir, "final_gs_model.pt")
    save_dict = gs_model.state_dict()
    save_dict['positions'] = gs_model.positions.cpu() # Add positions to the save file
    
    torch.save(save_dict, final_path)
    print(f"\nSuccess! 3DGS generation complete.")
    print(f"Final model saved to: {final_path}")

    # 7. Auto-export to PLY
    print("\n--- 7. Exporting to PLY ---")
    from dreamsea.export_ply import export_to_ply
    ply_path = os.path.join(output_dir, "final_gs_model.ply")
    export_to_ply(final_path, ply_path)
    print(f"Auto-exported PLY for visualization to: {ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3DGS scene from trained DreamSea checkpoints.")
    parser.add_argument("--cond_ckpt", type=str, required=True, help="Path to the conditional DDPM checkpoint.")
    parser.add_argument("--uncond_ckpt", type=str, default=None,
                        help="Path to the unconditional DDPM checkpoint. Only required when NOT "
                             "using --use_conditional_stitching, or when --sds_iterations > 0. "
                             "Do not pass the conditional checkpoint here.")
    parser.add_argument("--grid_size", type=int, default=3, help="Latent grid size (must be 2^n + 1, e.g., 3, 5, 9, or 1 for a single patch).")
    parser.add_argument("--roughness", type=float, default=0.5, help="Fractal roughness.")
    parser.add_argument("--sds_iterations", type=int, default=100, help="Number of SDS optimization steps.")
    parser.add_argument("--output_dir", type=str, default="outputs/3dgs_gen", help="Directory to save output files.")
    parser.add_argument("--latent_stats_path", type=str, default=None,
                        help="Path to latent_stats.json from preprocessing. Rescales fractal "
                             "grid into the training PCA range for in-distribution generation. "
                             "If omitted, the built-in DEFAULT_LATENT_STATS are used.")
    parser.add_argument("--use_conditional_stitching", action="store_true",
                        help="Use the conditional DDPM for inpainting seams (averages latent vectors of adjacent patches).")
    parser.add_argument("--rasterizer", type=str, choices=["scatter", "gsplat"], default="gsplat",
                        help="SDS renderer. 'gsplat' = faithful multi-view differentiable Gaussian "
                             "rasterizer (default; requires `pip install gsplat` and a CUDA device). "
                             "'scatter' = simplified single-view top-down fallback (color/opacity only, "
                             "no extra deps).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")

    args = parser.parse_args()

    generate_3dgs(
        cond_ckpt=args.cond_ckpt,
        uncond_ckpt=args.uncond_ckpt,
        grid_size=args.grid_size,
        roughness=args.roughness,
        sds_iterations=args.sds_iterations,
        output_dir=args.output_dir,
        latent_stats_path=args.latent_stats_path,
        use_conditional_stitching=args.use_conditional_stitching,
        rasterizer=args.rasterizer,
        device=args.device
    )
