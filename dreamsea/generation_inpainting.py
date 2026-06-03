import torch
import numpy as np
from diffusers import DDPMScheduler
from .models import ConditionalDDPM, UnconditionalDDPM

class GeneratorInpainter:
    def __init__(self, cond_model_path=None, uncond_model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Load conditional DDPM
        self.cond_model = ConditionalDDPM().to(device)
        if cond_model_path:
            checkpoint = torch.load(cond_model_path, map_location=device, weights_only=False)
            # Support both new dict-based and old raw state_dict checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            # Safely remove 'module.' prefix ONLY if it is at the very beginning
            clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
            # Diffusers occasionally updates its internal block naming (e.g., from direct 'group_norm'
            # to nested 'transformer_blocks'). Using strict=False bypasses missing/unexpected key errors.
            self.cond_model.load_state_dict(clean_state_dict, strict=False)
        self.cond_model.eval()

        # Load unconditional DDPM
        self.uncond_model = UnconditionalDDPM().to(device)
        if uncond_model_path:
            checkpoint = torch.load(uncond_model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            self.uncond_model.load_state_dict(clean_state_dict, strict=False)
        self.uncond_model.eval()

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

    @torch.no_grad()
    def generate_patch(self, latent_condition, num_inference_steps=1000):
        """
        Generates a 4-channel RGBD patch (224x224) using the conditional DDPM based on the latent condition.
        """
        # latent_condition is shape (2,) numpy or tensor
        if not isinstance(latent_condition, torch.Tensor):
            condition = torch.from_numpy(latent_condition).float().to(self.device)
        else:
            condition = latent_condition.float().to(self.device)

        # Reshape to (1, 1, 2)
        condition = condition.view(1, 1, 2)

        # Start from random noise
        image = torch.randn(1, 4, 224, 224, device=self.device)

        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        for t in self.scheduler.timesteps:
            # Predict noise residual
            noise_pred = self.cond_model(image, t, encoder_hidden_states=condition)

            # Compute previous noisy sample x_t -> x_t-1
            image = self.scheduler.step(noise_pred, t, image).prev_sample

        return image.cpu().numpy() # (1, 4, 224, 224)

    def generate_grid(self, latent_grid):
        """
        Inference loop iterating over fractal latent grid.
        latent_grid is (N, N, 2).
        Returns a grid of generated patches.
        """
        N = latent_grid.shape[0]
        patch_grid = np.zeros((N, N, 4, 224, 224), dtype=np.float32)

        print(f"Generating {N}x{N} grid patches...")
        for y in range(N):
            for x in range(N):
                patch = self.generate_patch(latent_grid[y, x])
                patch_grid[y, x] = patch[0]

        return patch_grid

    @torch.no_grad()
    def repaint_inpaint(self, image_input, mask, num_inference_steps=1000, jump_length=10, jump_n_sample=10, latent_condition=None):
        """
        RePaint inpainting (Lugmayr et al. 2022) using either the conditional or unconditional DDPM.

        image_input: known image regions (1, 4, H, W) — unknown pixels can hold any value.
        mask:        1=known, 0=unknown (1, 1, H, W).
        jump_length: how many steps to jump back for time-travel resampling.
        jump_n_sample: how many extra resamplings per jump (1 = no time-travel).
        latent_condition: (Optional) If provided, uses conditional model for inpainting.

        At each denoising step t→t-1:
          - Known pixels are re-noised to level t-1 from the clean reference.
          - Unknown pixels are predicted by the DDPM.
          - Periodically the trajectory jumps back jump_length steps and is
            resampled jump_n_sample times so boundaries harmonise.
        """
        if not isinstance(image_input, torch.Tensor):
            image_input = torch.from_numpy(image_input).to(self.device)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(self.device)

        cond_tensor = None
        if latent_condition is not None:
            if not isinstance(latent_condition, torch.Tensor):
                cond_tensor = torch.from_numpy(latent_condition).float().to(self.device)
            else:
                cond_tensor = latent_condition.float().to(self.device)
            cond_tensor = cond_tensor.view(1, 1, -1)

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        timesteps = list(self.scheduler.timesteps)  # descending: T, T-1, ..., 0

        x_t = torch.randn_like(image_input)

        i = 0
        total_steps = len(timesteps)
        while i < total_steps:
            if i % 10 == 0:
                print(f"RePaint step {i}/{total_steps}...")
            
            t = timesteps[i]

            # Known region: add noise at the *next* (lower) noise level so known
            # pixels match the target noise level after this denoising step.
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            noise_known = torch.randn_like(image_input)
            x_known = self.scheduler.add_noise(
                image_input, noise_known,
                torch.tensor([t_prev], device=self.device)
            )

            # Unknown region: standard DDPM denoising step
            if cond_tensor is not None:
                noise_pred = self.cond_model(x_t, t, encoder_hidden_states=cond_tensor)
            else:
                noise_pred = self.uncond_model(x_t, t)
            
            x_unknown = self.scheduler.step(noise_pred, t, x_t).prev_sample

            x_t = mask * x_known + (1 - mask) * x_unknown

            # RePaint time-travel: every jump_length steps, re-noise and resample
            # jump_n_sample-1 extra times so the model can harmonise the boundary.
            if jump_n_sample > 1 and (i + 1) % jump_length == 0 and i + 1 < len(timesteps):
                jump_back_idx = max(0, i + 1 - jump_length)
                t_jump = timesteps[jump_back_idx]
                
                print(f"  [Time-Travel] Jumping back from step {i} to {jump_back_idx} ({jump_n_sample - 1} times)")

                for _ in range(jump_n_sample - 1):
                    # Forward-diffuse x_t back to t_jump noise level
                    noise_fwd = torch.randn_like(x_t)
                    x_t = self.scheduler.add_noise(
                        x_t, noise_fwd,
                        torch.tensor([t_jump], device=self.device)
                    )
                    # Denoise from t_jump back down to the current position
                    for j in range(jump_back_idx, i + 1):
                        t_j = timesteps[j]
                        t_j_prev = timesteps[j + 1] if j + 1 < len(timesteps) else 0
                        noise_k = torch.randn_like(image_input)
                        x_known_j = self.scheduler.add_noise(
                            image_input, noise_k,
                            torch.tensor([t_j_prev], device=self.device)
                        )
                        if cond_tensor is not None:
                            np_j = self.cond_model(x_t, t_j, encoder_hidden_states=cond_tensor)
                        else:
                            np_j = self.uncond_model(x_t, t_j)
                        
                        x_unknown_j = self.scheduler.step(np_j, t_j, x_t).prev_sample
                        x_t = mask * x_known_j + (1 - mask) * x_unknown_j

            i += 1

        return x_t.cpu().numpy()

    def stitch_and_inpaint(self, patch_grid, overlap_size=32, latent_grid=None, use_conditional=False):
        """
        Takes the generated patches and stitches them into a dense RGBD map.
        Uses a parallelizable inpainting pattern with RePaint to handle seams/overlaps
        to preserve latent control accuracy.
        """
        N = patch_grid.shape[0]
        patch_size = 224

        # Calculate final canvas size
        canvas_size = N * patch_size - (N - 1) * overlap_size
        canvas = np.zeros((4, canvas_size, canvas_size), dtype=np.float32)
        weight_map = np.zeros((1, canvas_size, canvas_size), dtype=np.float32)

        print("Stitching patches into global map...")
        # Simple blending first
        for y in range(N):
            for x in range(N):
                y_start = y * (patch_size - overlap_size)
                y_end = y_start + patch_size
                x_start = x * (patch_size - overlap_size)
                x_end = x_start + patch_size

                canvas[:, y_start:y_end, x_start:x_end] += patch_grid[y, x]
                weight_map[:, y_start:y_end, x_start:x_end] += 1.0

        canvas = canvas / np.maximum(weight_map, 1e-5)

        # Mask: 1=known (single-patch coverage), 0=unknown (overlap to inpaint)
        mask = (weight_map <= 1.0).astype(np.float32)

        # Collect all seam tile coordinates first (horizontal then vertical seams).
        # All tiles are then inpainted from a frozen snapshot of the canvas so no
        # seam patch depends on another — matching the parallelizable inpainting
        # pattern described in the paper.
        print(f"Inpainting seams using parallelizable RePaint (Conditional: {use_conditional})...")
        seam_regions = []

        for y in range(1, N):
            y_seam_center = y * (patch_size - overlap_size) + overlap_size // 2
            y_s = y_seam_center - patch_size // 2
            y_e = y_seam_center + patch_size // 2
            for x in range(N):
                x_s = x * (patch_size - overlap_size)
                x_e = x_s + patch_size
                if y_s >= 0 and y_e <= canvas_size and x_s >= 0 and x_e <= canvas_size:
                    cond = None
                    if use_conditional and latent_grid is not None:
                        cond = (latent_grid[y - 1, x] + latent_grid[y, x]) / 2.0
                    seam_regions.append((y_s, y_e, x_s, x_e, cond))

        for x in range(1, N):
            x_seam_center = x * (patch_size - overlap_size) + overlap_size // 2
            x_s = x_seam_center - patch_size // 2
            x_e = x_seam_center + patch_size // 2
            for y in range(N):
                y_s = y * (patch_size - overlap_size)
                y_e = y_s + patch_size
                if y_s >= 0 and y_e <= canvas_size and x_s >= 0 and x_e <= canvas_size:
                    cond = None
                    if use_conditional and latent_grid is not None:
                        cond = (latent_grid[y, x - 1] + latent_grid[y, x]) / 2.0
                    seam_regions.append((y_s, y_e, x_s, x_e, cond))

        # Freeze the canvas state so every seam tile reads from the same unmodified
        # blended image — writes go to canvas only after all reads are determined.
        canvas_snapshot = canvas.copy()
        mask_snapshot = mask.copy()

        seam_count = len(seam_regions)
        for idx, (y_s, y_e, x_s, x_e, cond) in enumerate(seam_regions):
            print(f"  Inpainting seam {idx + 1}/{seam_count}...")
            local_canvas = np.expand_dims(canvas_snapshot[:, y_s:y_e, x_s:x_e], 0)
            local_mask = np.expand_dims(mask_snapshot[:, y_s:y_e, x_s:x_e], 0)

            inpainted = self.repaint_inpaint(local_canvas, local_mask, latent_condition=cond)

            canvas[:, y_s:y_e, x_s:x_e] = (
                local_mask[0] * local_canvas[0] + (1 - local_mask[0]) * inpainted[0]
            )

        return canvas
