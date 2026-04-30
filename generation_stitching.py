import torch
import numpy as np
from diffusers import DDPMScheduler

def generate_patch(latent_vector, conditional_ddpm, semantic_projector, noise_scheduler, device="cuda"):
    """
    Generates an RGBD patch from a 2D latent vector using the trained conditional DDPM.

    Args:
        latent_vector (np.ndarray): 2D latent vector (2,).
        conditional_ddpm: Trained UNet2DConditionModel.
        semantic_projector: Trained linear projector for semantics.
        noise_scheduler: DDPMScheduler used during training.

    Returns:
        torch.Tensor: Generated RGBD patch of shape (4, H, W).
    """
    conditional_ddpm.eval()
    semantic_projector.eval()

    # We pretend the single latent vector represents an entire homogeneous patch for conditioning
    # For a patch size of 64x64, let's create a sequence length of 16x16 (matching downsampled UNet context)
    seq_len = 16 * 16
    # Broadcast the single latent vector to the sequence length
    sem_flat = torch.tensor(latent_vector).float().unsqueeze(0).repeat(seq_len, 1).unsqueeze(0).to(device) # (1, SeqLen, 2)

    with torch.no_grad():
        encoder_hidden_states = semantic_projector(sem_flat) # (1, SeqLen, cross_attention_dim)

        # Start with random noise
        image = torch.randn((1, 4, 64, 64)).to(device)

        # Denoising loop
        noise_scheduler.set_timesteps(100) # Inference steps
        for t in noise_scheduler.timesteps:
            # Predict noise residual
            noise_pred = conditional_ddpm(image, t, encoder_hidden_states=encoder_hidden_states).sample

            # Compute previous image x_{t-1}
            image = noise_scheduler.step(noise_pred, t, image).prev_sample

    return image.squeeze(0).cpu() # (4, 64, 64)

from diffusers import RePaintScheduler

def repaint_stitch(patches_grid, unconditional_ddpm, noise_scheduler, overlap=16, device="cuda", num_inference_steps=100, jump_length=10, jump_n_sample=10):
    """
    Stitches a grid of generated RGBD patches using RePaint-style inpainting.

    Args:
        patches_grid (list of lists of torch.Tensor): Grid of generated RGBD patches.
        unconditional_ddpm: An unconditional DDPM trained on underwater patches to act as prior.
        noise_scheduler: DDPMScheduler or RePaintScheduler.
        overlap (int): Number of pixels overlapping between adjacent patches.

    Returns:
        torch.Tensor: Massive, seamless stitched RGBD map.
    """
    grid_h = len(patches_grid)
    grid_w = len(patches_grid[0])
    patch_size = patches_grid[0][0].shape[1] # Assumes square patches, e.g., 64
    channels = patches_grid[0][0].shape[0]

    stride = patch_size - overlap

    out_h = stride * (grid_h - 1) + patch_size
    out_w = stride * (grid_w - 1) + patch_size

    stitched_map = torch.zeros((1, channels, out_h, out_w)).to(device)
    count_map = torch.zeros((1, 1, out_h, out_w)).to(device)

    # Create mask indicating overlapping regions
    boundary_mask = torch.zeros((1, 1, out_h, out_w)).to(device)

    # 1. Naive placement to get initial image and compute overlaps
    for y in range(grid_h):
        for x in range(grid_w):
            patch = patches_grid[y][x].unsqueeze(0).to(device)
            y_start = y * stride
            x_start = x * stride

            stitched_map[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] += patch
            count_map[:, :, y_start:y_start+patch_size, x_start:x_start+patch_size] += 1

    # Normalize by count
    stitched_map = stitched_map / torch.clamp(count_map, min=1.0)

    # In diffusers RePaintScheduler, mask = 1 means "known" and mask = 0 means "unknown"
    # We want to keep the non-overlapping regions as known (1.0), and inpaint the overlaps (0.0)
    boundary_mask = torch.ones_like(count_map)
    boundary_mask[count_map > 1] = 0.0

    # Convert scheduler to RePaintScheduler if it isn't already
    if not isinstance(noise_scheduler, RePaintScheduler):
        repaint_scheduler = RePaintScheduler.from_config(noise_scheduler.config)
    else:
        repaint_scheduler = noise_scheduler

    repaint_scheduler.set_timesteps(num_inference_steps, jump_length=jump_length, jump_n_sample=jump_n_sample, device=device)

    # The known image is our naively stitched map
    original_image = stitched_map.clone()

    # Initialize random noise
    image = torch.randn_like(stitched_map)

    unconditional_ddpm.eval()

    print("Running RePaint inpainting for seamless boundaries...")
    with torch.no_grad():
        for t in repaint_scheduler.timesteps:
            # 1. Denoise step on current image
            model_output = unconditional_ddpm(image, t).sample

            # 2. RePaint scheduler step to merge known image with denoised output according to mask
            image = repaint_scheduler.step(
                model_output, t, image, original_image, boundary_mask
            ).prev_sample

    print(f"Stitched map dimensions: {image.squeeze(0).shape}")
    return image.squeeze(0).cpu()

def generate_full_map(latents_grid, conditional_ddpm, semantic_proj, noise_scheduler, unconditional_ddpm):
    """
    Iterates over the grid of latents, generates patches, and stitches them.
    """
    grid_size = latents_grid.shape[0]
    patches_grid = []

    print(f"Generating patches for a {grid_size}x{grid_size} grid...")
    for y in range(grid_size):
        row_patches = []
        for x in range(grid_size):
            latent = latents_grid[y, x]
            # Generate individual patch
            patch = generate_patch(latent, conditional_ddpm, semantic_proj, noise_scheduler)
            row_patches.append(patch)
        patches_grid.append(row_patches)

    print("Stitching patches using RePaint framework...")
    full_map = repaint_stitch(patches_grid, unconditional_ddpm, noise_scheduler)

    return full_map

if __name__ == "__main__":
    print("Generation and Stitching script initialized.")
    # Example usage:
    # Requires trained conditional_ddpm and unconditional_ddpm models
    # full_rgbd_map = generate_full_map(latents_grid, cond_model, sem_proj, scheduler, uncond_model)
