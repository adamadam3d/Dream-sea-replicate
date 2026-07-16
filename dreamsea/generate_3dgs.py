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

def _load_reference_condition(reference_cond):
    """Load a 2D condition vector from a preprocessed conditions/<name>_cond.pt file.

    These are the exact PCA conditions saved by preprocess_dataset.py, so the
    returned vector is a real, in-distribution training condition — no DINOv2/PCA
    recomputation is needed to reuse it.
    """
    if not os.path.exists(reference_cond):
        raise ValueError(
            f"--reference_cond '{reference_cond}' not found. Pass the path to a preprocessed "
            f"condition file, e.g. <preprocess_out>/conditions/<name>_cond.pt"
        )
    cond = torch.load(reference_cond, map_location='cpu', weights_only=True)
    cond = np.asarray(cond.float().numpy(), dtype=np.float32).reshape(-1)
    if cond.shape[0] != 2:
        raise ValueError(
            f"Reference condition must have 2 components, got {cond.shape[0]} from '{reference_cond}'."
        )
    return cond

def generate_3dgs(cond_ckpt, uncond_ckpt, grid_size=3, roughness=0.5,
                  output_dir="outputs", sds_iterations=500, sds_guidance=0.0,
                  sds_rgbd=False, sds_anchor=1.0,
                  latent_stats_path=None, use_conditional_stitching=False,
                  rasterizer="gsplat", save_init_ply=False, upscale_factor=1.0,
                  splat_scale=0.75, surfel_init=True, relief=10.0,
                  densify_views=30, align_depth=True,
                  reference_cond=None, reference_spread=1.0,
                  latent_vector=None, sr_ckpt=None, sr_factor=4, sr_steps=100,
                  sr_color_correct=True, sr_color_strength=1.0,
                  patch_size=224, redilate=False, redilation_fraction=0.5,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Full pipeline to generate a 3DGS scene from trained checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)
    run_name = os.path.basename(os.path.normpath(output_dir))

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

    # 2. Generate Latent Grid (Fractal or Constant)
    if latent_vector is not None:
        # User provided a fixed 2D condition: use it for all patches (uniform scene)
        print(f"\n--- 2. Using Constant Latent Vector ---")
        print(f"Latent condition: {latent_vector.tolist()}")
        latent_grid = np.full((grid_size, grid_size, 2), latent_vector, dtype=np.float32)
    else:
        # Generate a fractal grid with natural variation
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
        base_mean = np.array(latent_stats["mean"], dtype=np.float32)
        base_std = np.array(latent_stats["std"], dtype=np.float32)
    else:
        # Older stats files stored only min/max — approximate a Gaussian fit
        # (mean = midpoint, std ~= range/4) so the fixed affine map still works.
        lo = np.array(latent_stats["min"], dtype=np.float32)
        hi = np.array(latent_stats["max"], dtype=np.float32)
        base_mean = (lo + hi) / 2.0
        base_std = (hi - lo) / 4.0

    # Optionally re-center the fractal field on a chosen preprocessed sample's
    # condition so the whole scene is generated "of that type". The reference is
    # the 2D PCA vector already saved at preprocessing (conditions/<name>_cond.pt),
    # so it is exactly in-distribution. reference_spread scales how far patches
    # vary around it: 1.0 = natural terrain variation, 0.0 = every patch identical.
    if reference_cond:
        ref_latent = _load_reference_condition(reference_cond)
        grid_mean = ref_latent
        grid_std = base_std * reference_spread
        print(f"Centering generation on reference condition {ref_latent.tolist()} "
              f"from '{reference_cond}' (spread x{reference_spread}).")
    else:
        grid_mean = base_mean
        grid_std = base_std

    # An explicit --latent_vector is already a raw PCA coordinate (same space as
    # the saved conditions/*.pt and generate_sample.py's --latent_vector), so it
    # is fed to the model DIRECTLY. scale_latent_grid is only for mapping the
    # unit-variance fractal field into the PCA range — applying it here would
    # multiply the user's value by ~std and push it out of distribution.
    if latent_vector is not None:
        print(f"Using constant latent condition directly (no rescale): {latent_vector.tolist()}")
    else:
        latent_grid = scale_latent_grid(latent_grid, grid_mean, grid_std)
        print(f"Latent grid mapped into training PCA distribution using: {stats_source}")

    print(f"Latent grid generated.")

    # 3. Generate and Stitch RGBD Patches
    print("\n--- 3. Generating and Stitching RGBD Patches ---")
    # This uses the conditional model to generate patches and unconditional to inpaint seams
    patch_grid = generator.generate_grid(latent_grid, patch_size=patch_size,
                                         redilate=redilate,
                                         redilation_fraction=redilation_fraction)
    global_map = generator.stitch_and_inpaint(
        patch_grid,
        overlap_size=32,
        latent_grid=latent_grid,
        use_conditional=use_conditional_stitching,
        align_depth=align_depth
    )
    
    # Save the global RGBD map for inspection
    map_path = os.path.join(output_dir, f"{run_name}_rgbd_map.pt")
    torch.save(torch.from_numpy(global_map), map_path)
    print(f"Global RGBD Map saved to: {map_path}")

    # Increase 3DGS resolution by upscaling the stitched RGBD map. Two paths:
    #  - --sr_ckpt: real xN diffusion super-resolution (the trained SR cascade),
    #    run tiled over the seam-corrected map so the SR model stays at its
    #    trained ~224 resolution and large maps don't OOM. This adds genuine
    #    high-frequency detail (and color-corrects the diffusion hue drift).
    #  - otherwise --upscale_factor: plain bilinear resize (more points, no new
    #    detail) — the legacy behavior, kept as the no-SR fallback.
    if sr_ckpt:
        from diffusers import DDPMScheduler  # noqa: F401 (ensures diffusers import path)
        from dreamsea.sr_upscale import load_sr_model, make_sr_scheduler, sr_upscale_tiled
        print(f"\n--- Super-resolving global map x{sr_factor} (SR cascade) ---")
        print(f"Loading SR stage from: {sr_ckpt}")
        sr_model = load_sr_model(sr_ckpt, device)
        sr_scheduler = make_sr_scheduler(sr_ckpt)
        in_map = torch.from_numpy(global_map).float()  # (4, H, W) in [-1, 1]
        sr_map = sr_upscale_tiled(in_map, sr_model, sr_scheduler, factor=sr_factor,
                                  num_inference_steps=sr_steps, device=device,
                                  color_correct=sr_color_correct,
                                  color_strength=sr_color_strength)
        global_map = sr_map.numpy()
        # Downstream point-cloud tiling scales its patch/overlap by upscale_factor,
        # so make it match the resolution the SR pass actually produced.
        upscale_factor = float(sr_factor)
        print(f"SR map: {in_map.shape[2]}x{in_map.shape[1]} -> "
              f"{global_map.shape[2]}x{global_map.shape[1]} (factor: {sr_factor})")
    elif upscale_factor > 1.0:
        global_tensor = torch.from_numpy(global_map).unsqueeze(0).to(device)
        new_H = int(global_map.shape[1] * upscale_factor)
        new_W = int(global_map.shape[2] * upscale_factor)

        rgb_upscaled = torch.nn.functional.interpolate(
            global_tensor[:, :3], size=(new_H, new_W), mode='bilinear', align_corners=False
        )
        depth_upscaled = torch.nn.functional.interpolate(
            global_tensor[:, 3:4], size=(new_H, new_W), mode='bilinear', align_corners=False
        )

        global_tensor_upscaled = torch.cat([rgb_upscaled, depth_upscaled], dim=1)
        global_map = global_tensor_upscaled.squeeze(0).cpu().numpy()
        print(f"Upscaled global RGBD map from {global_tensor.shape[2]}x{global_tensor.shape[3]} "
              f"to {new_H}x{new_W} (factor: {upscale_factor})")

    # Save the global RGB and Depth map collage as a PNG image
    from PIL import Image
    rgb_map_path = os.path.join(output_dir, f"{run_name}_rgb_map.png")
    
    # 1. RGB Image
    rgb_map = global_map[:3, :, :] # Extract RGB channels
    rgb_img_np = (rgb_map + 1.0) / 2.0 # Normalize from [-1, 1] to [0, 1]
    rgb_img_np = np.clip(rgb_img_np, 0.0, 1.0)
    rgb_img_np = (rgb_img_np * 255.0).astype(np.uint8) # Convert to uint8
    rgb_img_np = np.transpose(rgb_img_np, (1, 2, 0)) # CHW -> HWC
    rgb_image = Image.fromarray(rgb_img_np)
    
    # 2. Depth Image (Heatmap: Red-Blue)
    depth_map = global_map[3, :, :] # Extract Depth channel (H, W)
    depth_img_np = (depth_map + 1.0) / 2.0 # Normalize from [-1, 1] to [0, 1]
    depth_img_np = np.clip(depth_img_np, 0.0, 1.0)
    
    # Red-Blue colormap
    r = (depth_img_np * 255.0).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1.0 - depth_img_np) * 255.0).astype(np.uint8)
    depth_colored_np = np.stack([r, g, b], axis=-1)
    depth_image = Image.fromarray(depth_colored_np, mode='RGB')
    
    # 3. Create Collage
    W, H = rgb_image.size
    collage = Image.new('RGB', (W, H * 2))
    collage.paste(rgb_image, (0, 0))
    collage.paste(depth_image, (0, H))
    collage.save(rgb_map_path)
    print(f"Global RGB and Depth map collage saved to: {rgb_map_path}")

    # 4. Initialize 3D Gaussian Splatting
    print("\n--- 4. Initializing 3D Gaussian Splatting ---")
    positions, colors, conds = gs_opt.create_point_cloud_from_rgbd(
        global_map,
        latent_grid=latent_grid,
        patch_size=int(patch_size * upscale_factor),
        overlap_size=int(32 * upscale_factor),
        relief=relief
    )
    print(f"Extracted {positions.shape[0]} points from map.")

    if positions.shape[0] == 0:
        print("Error: No valid points found in RGBD map. Aborting.")
        return

    # Per-point spacing of the unprojected pixel grid (z / focal, with the same
    # fov=60 used inside create_point_cloud_from_rgbd) so splat sizes match the
    # map's actual resolution instead of growing with map width.
    focal = 0.5 * global_map.shape[2] / np.tan(np.radians(60.0) / 2.0)
    point_spacing = positions[:, 2] / focal

    # Slope-aware anisotropic (surfel) init: disc Gaussians aligned to the local
    # surface, sized to the true 3D neighbor gaps — fills the pinholes that
    # isotropic splats leave on steep slopes without blurring flat terrain.
    init_scales, init_quats = (None, None)
    if surfel_init:
        init_scales, init_quats = gs_opt.compute_surfel_init(global_map, splat_scale=splat_scale,
                                                             relief=relief)
        print("Using surfel (slope-aware anisotropic) initialization.")

    gs_model = gs_opt.GaussianSplattingModel(positions, colors, point_cloud_conds=conds,
                                             upscale_factor=upscale_factor,
                                             point_spacing=point_spacing,
                                             splat_scale=splat_scale,
                                             init_scales=init_scales, init_quats=init_quats,
                                             device=device)
    print("3DGS model initialized.")

    # Alpha-driven hole densification (SplaTAM-style): render silhouettes from
    # random near-nadir views and insert Gaussians wherever interior coverage
    # drops out. Runs BEFORE the init PLY save and before SDS builds its
    # optimizer (add_gaussians re-creates the parameters).
    if densify_views > 0 and rasterizer == "gsplat":
        print(f"\n--- 4b. Hole densification ({densify_views} views) ---")
        from dreamsea.gs_sds_optimization_v2 import densify_holes
        patch_px = patch_size * upscale_factor
        frame_fraction = min(patch_px / global_map.shape[2], 1.0)
        scene_extent = (gs_model.positions.max(0).values - gs_model.positions.min(0).values).norm()
        densify_holes(gs_model, views=densify_views,
                      frame_extent=(scene_extent * frame_fraction).item())
    elif densify_views > 0:
        print("Skipping hole densification (requires --rasterizer gsplat).")

    # Save initial PLY before SDS optimization if requested
    if save_init_ply:
        init_pt_path = os.path.join(output_dir, f"{run_name}_init.pt")
        init_save_dict = gs_model.state_dict()
        init_save_dict['positions'] = gs_model.positions.cpu()
        torch.save(init_save_dict, init_pt_path)
        
        init_ply_path = os.path.join(output_dir, f"{run_name}_init.ply")
        from dreamsea.export_ply import export_to_ply
        export_to_ply(init_pt_path, init_ply_path)
        print(f"Saved initial pre-SDS model PLY to: {init_ply_path}")

    # 5. SDS Optimization
    if sds_iterations > 0:
        print(f"\n--- 5. Optimizing 3DGS via SDS ({sds_iterations} iterations, "
              f"rasterizer={rasterizer}) ---")
        if rasterizer == "gsplat":
            # Faithful multi-view SDS with a real differentiable Gaussian rasterizer.
            # Each view frames ~one training-patch footprint so the texture scale
            # matches the prior; the per-view condition is the latent at the
            # framed point, which is now coherent with what is in frame.
            from dreamsea.gs_sds_optimization_v2 import optimize_3dgs_sds_multiview
            patch_px = patch_size * upscale_factor
            frame_fraction = min(patch_px / global_map.shape[2], 1.0)
            optimize_3dgs_sds_multiview(
                model=gs_model,
                diffusion_model=generator.cond_model,
                scheduler=generator.scheduler,
                iterations=sds_iterations,
                guidance=sds_guidance,
                rgb_only=not sds_rgbd,
                frame_fraction=frame_fraction,
                anchor_weight=sds_anchor,
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
    final_path = os.path.join(output_dir, f"{run_name}.pt")
    save_dict = gs_model.state_dict()
    save_dict['positions'] = gs_model.positions.cpu() # Add positions to the save file
    
    torch.save(save_dict, final_path)
    print(f"\nSuccess! 3DGS generation complete.")
    print(f"Final model saved to: {final_path}")

    # 7. Auto-export to PLY
    print("\n--- 7. Exporting to PLY ---")
    from dreamsea.export_ply import export_to_ply
    ply_path = os.path.join(output_dir, f"{run_name}.ply")
    export_to_ply(final_path, ply_path)
    print(f"Auto-exported PLY for visualization to: {ply_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3DGS scene from trained DreamSea checkpoints.")
    parser.add_argument("-c", "--cond_ckpt", type=str, required=True, help="Path to the conditional DDPM checkpoint.")
    parser.add_argument("-u", "--uncond_ckpt", type=str, default=None,
                        help="Path to the unconditional DDPM checkpoint. Only required when NOT "
                             "using --use_conditional_stitching, or when --sds_iterations > 0. "
                             "Do not pass the conditional checkpoint here.")
    parser.add_argument("-g", "--grid_size", type=int, default=3, help="Latent grid size (must be 2^n + 1, e.g., 3, 5, 9, or 1 for a single patch).")
    parser.add_argument("-r", "--roughness", type=float, default=0.5, help="Fractal roughness.")
    parser.add_argument("-i", "--sds_iterations", type=int, default=100, help="Number of SDS optimization steps.")
    parser.add_argument("-s", "--sds_guidance", type=float, default=0.0,
                        help="Classifier-Free Guidance (CFG) scale for conditional SDS. Default 0 "
                             "(disabled): the conditional model was trained WITHOUT condition "
                             "dropout and the PCA latents are zero-mean, so a zero latent is the "
                             "MEAN condition rather than a null branch — CFG > 0 extrapolates "
                             "between two conditional predictions and produces saturated artifacts.")
    parser.add_argument("-q", "--sds_rgbd", action="store_true",
                        help="Backpropagate the SDS gradient through the depth channel as well. "
                             "Off by default: the rendered per-view min-max depth does not match "
                             "the per-image relative depth the prior was trained on, and its "
                             "gradient corrupts scaling/opacity (depth is still rendered as "
                             "context for the 4-channel UNet either way).")
    parser.add_argument("-w", "--sds_anchor", type=float, default=1.0,
                        help="Weight of the MSE anchor pulling color/opacity/scaling back toward "
                             "their initialization during SDS. 0 disables the anchor.")
    parser.add_argument("-o", "--output_dir", type=str, default="outputs/3dgs_gen", help="Directory to save output files.")
    parser.add_argument("-l", "--latent_stats_path", type=str, default=None,
                        help="Path to latent_stats.json from preprocessing. Rescales fractal "
                             "grid into the training PCA range for in-distribution generation. "
                             "If omitted, the built-in DEFAULT_LATENT_STATS are used.")
    parser.add_argument("-t", "--use_conditional_stitching", action="store_true",
                        help="Use the conditional DDPM for inpainting seams (averages latent vectors of adjacent patches).")
    parser.add_argument("-z", "--rasterizer", type=str, choices=["scatter", "gsplat"], default="gsplat",
                        help="SDS renderer. 'gsplat' = faithful multi-view differentiable Gaussian "
                             "rasterizer (default; requires `pip install gsplat` and a CUDA device). "
                             "'scatter' = simplified single-view top-down fallback (color/opacity only, "
                             "no extra deps).")
    parser.add_argument("-p", "--save_init_ply", action="store_true",
                        help="Save the initial point cloud/3DGS model as a .ply file before performing SDS optimization.")
    parser.add_argument("-f", "--upscale_factor", type=float, default=1.0,
                        help="Bilinear upscale factor for the stitched RGBD map before unprojecting "
                             "(more points, no new detail). Ignored when --sr_ckpt is given. Values > 1.0 "
                             "increase point density/3DGS resolution.")
    parser.add_argument("--sr_ckpt", type=str, default=None,
                        help="Trained SR-stage checkpoint. When set, the stitched RGBD map is "
                             "super-resolved with the diffusion SR cascade (tiled, so it stays at "
                             "the model's trained resolution and won't OOM) instead of bilinear "
                             "upscaling — adds real high-frequency detail for the 3DGS.")
    parser.add_argument("--sr_factor", type=int, default=4,
                        help="SR upscale factor (the model is trained for 4).")
    parser.add_argument("--sr_steps", type=int, default=100,
                        help="Denoising steps per SR tile.")
    parser.add_argument("--no_sr_color_correct", action="store_true",
                        help="Disable AdaIN color correction on the SR output (on by default; "
                             "removes the diffusion-SR hue drift).")
    parser.add_argument("--sr_color_strength", type=float, default=1.0,
                        help="Blend factor for SR color correction in [0, 1].")
    parser.add_argument("-P", "--patch_size", type=int, default=224,
                        help="Per-patch generation resolution (must be divisible by 32). "
                             "Default 224 (the trained size). Larger values give more detail "
                             "per patch before stitching/SR, at ~quadratic VRAM/time cost; "
                             "pair with --redilate above 224 to avoid repeated-tile artifacts.")
    parser.add_argument("--redilate", action="store_true",
                        help="Apply ScaleCrafter re-dilation during patch generation when "
                             "--patch_size > 224 (scales the convs' receptive field so the "
                             "larger patch keeps a coherent global layout). Off by default.")
    parser.add_argument("--redilation_fraction", type=float, default=0.5,
                        help="Fraction of denoising steps (from the noisiest) run with dilated "
                             "convs. Higher = more coherent layout, lower = sharper texture.")
    parser.add_argument("-k", "--splat_scale", type=float, default=0.75,
                        help="Initial Gaussian size as a multiple of the inter-point spacing. "
                             "Lower = sharper texture (risk of pinholes on steep slopes), "
                             "higher = smoother/safer coverage. 0.75 keeps the surface "
                             "hole-free without blurring the RGBD map.")
    parser.add_argument("--relief", type=float, default=10.0,
                        help="Vertical relief multiplier mapping the [0,1] depth channel to world "
                             "z (z = depth*relief + 1). The historical value is 10; lower values "
                             "(3-5) give gentler slopes and markedly fewer pinholes.")
    parser.add_argument("--densify_views", type=int, default=30,
                        help="Number of random near-nadir views for alpha-driven hole "
                             "densification (SplaTAM-style: insert Gaussians wherever interior "
                             "silhouette coverage drops out). 0 disables. Requires --rasterizer "
                             "gsplat.")
    parser.add_argument("--no_align_depth", action="store_true",
                        help="Disable the per-patch depth scale/offset alignment over overlap "
                             "bands before stitching (leaves seam depth steps in the map).")
    parser.add_argument("--no_surfel", action="store_true",
                        help="Disable the slope-aware anisotropic (surfel) initialization and "
                             "fall back to isotropic splats sized from the horizontal grid "
                             "spacing (the old behavior; pinholes on steep slopes).")
    parser.add_argument("-R", "--reference_cond", type=str, default=None,
                        help="Path to a preprocessed condition file (conditions/<name>_cond.pt) from "
                             "preprocess_dataset.py. Centers the whole generation on that sample's 2D "
                             "condition so the scene comes out 'of that type'. Guaranteed "
                             "in-distribution — it is a real training condition, not a fractal guess.")
    parser.add_argument("-X", "--reference_spread", type=float, default=1.0,
                        help="When --reference_cond is set, multiplier on the per-axis latent std used "
                             "to vary patches around the reference. 1.0 = natural terrain variation "
                             "around that type; 0.0 = every patch is exactly that type.")
    parser.add_argument("-L", "--latent_vector", type=str, default=None,
                        help="Fixed 2D latent condition as a RAW PCA coordinate (comma-separated, e.g. "
                             "'0,0' for the dataset mean, or values up to ~±std from latent_stats). Same "
                             "space as generate_sample.py's --latent_vector and the saved conditions/*.pt, "
                             "so it is fed to the model directly (NOT rescaled). If provided, ALL patches use "
                             "this constant condition (uniform scene type). Overrides --reference_cond. "
                             "Useful for testing a specific condition or reproducing a generate_sample.py result in 3D.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")

    args = parser.parse_args()

    # Parse latent vector if provided
    latent_vec = None
    if args.latent_vector:
        try:
            latent_vec = np.array([float(x) for x in args.latent_vector.split(',')], dtype=np.float32)
            if latent_vec.shape[0] != 2:
                raise ValueError("Latent vector must have exactly 2 components.")
        except Exception as e:
            print(f"Error parsing --latent_vector '{args.latent_vector}': {e}")
            import sys
            sys.exit(1)

    generate_3dgs(
        cond_ckpt=args.cond_ckpt,
        uncond_ckpt=args.uncond_ckpt,
        grid_size=args.grid_size,
        roughness=args.roughness,
        sds_iterations=args.sds_iterations,
        sds_guidance=args.sds_guidance,
        sds_rgbd=args.sds_rgbd,
        sds_anchor=args.sds_anchor,
        output_dir=args.output_dir,
        latent_stats_path=args.latent_stats_path,
        use_conditional_stitching=args.use_conditional_stitching,
        rasterizer=args.rasterizer,
        save_init_ply=args.save_init_ply,
        upscale_factor=args.upscale_factor,
        splat_scale=args.splat_scale,
        surfel_init=not args.no_surfel,
        relief=args.relief,
        densify_views=args.densify_views,
        align_depth=not args.no_align_depth,
        reference_cond=args.reference_cond,
        reference_spread=args.reference_spread,
        latent_vector=latent_vec,
        sr_ckpt=args.sr_ckpt,
        sr_factor=args.sr_factor,
        sr_steps=args.sr_steps,
        sr_color_correct=not args.no_sr_color_correct,
        sr_color_strength=args.sr_color_strength,
        patch_size=args.patch_size,
        redilate=args.redilate,
        redilation_fraction=args.redilation_fraction,
        device=args.device
    )
