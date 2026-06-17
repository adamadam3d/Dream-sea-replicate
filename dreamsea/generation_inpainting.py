import torch
import numpy as np
from diffusers import DDPMScheduler
from .models import ConditionalDDPM, UnconditionalDDPM


def _load_ddpm_checkpoint(model, ckpt_path, device):
    """Load a DDPM checkpoint into `model`, failing loudly on any key mismatch.

    A silent partial load (strict=False) leaves the unmatched layers at their
    random initialization. A UNet with random weights emits garbage "noise"
    predictions, so the reverse diffusion never denoises and the output collapses
    to rainbow static. We therefore load with strict=True and, on failure, raise
    a clear, actionable error instead of running on a half-initialized model.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Support both new dict-based and old raw state_dict checkpoint formats.
    # Prefer EMA weights when the checkpoint carries them — the averaged copy
    # produces noticeably cleaner samples than the raw training weights.
    if isinstance(checkpoint, dict) and 'ema_model_state_dict' in checkpoint:
        state_dict = checkpoint['ema_model_state_dict']
        print(f"Using EMA weights from '{ckpt_path}'.")
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # Safely remove 'module.' prefix ONLY if it is at the very beginning
    clean_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    try:
        model.load_state_dict(clean_state_dict, strict=True)
    except RuntimeError as e:
        # Re-run non-strict purely to enumerate exactly what didn't line up.
        result = model.load_state_dict(clean_state_dict, strict=False)

        # Cross-attention transformer blocks only exist in the CONDITIONAL model.
        # If they show up where the target model didn't expect them (or are absent
        # where it did), the wrong *kind* of checkpoint was passed — e.g. handing a
        # conditional .pt to the unconditional slot. That's a model-type mix-up,
        # not a diffusers version drift.
        ckpt_is_conditional = any("transformer_blocks" in k for k in result.unexpected_keys)
        model_is_conditional = any("transformer_blocks" in k for k in result.missing_keys)
        if ckpt_is_conditional or model_is_conditional:
            kind = "conditional" if ckpt_is_conditional else "unconditional"
            hint = (
                f"Model-type mismatch: the checkpoint is a {kind.upper()} model but you are "
                f"loading it into {type(model).__name__}. Pass a conditional checkpoint to "
                f"--cond_ckpt and an unconditional checkpoint to --uncond_ckpt — do NOT reuse "
                f"the same file for both."
            )
        else:
            hint = (
                "This usually means the installed 'diffusers' version differs from the one used "
                "to train this checkpoint (diffusers renames internal UNet submodules between "
                "releases). Pin diffusers to the training version in requirements.txt, or remap "
                "the keys. Do NOT silence this with strict=False — that runs the model on random "
                "weights and produces pure-noise output."
            )
        raise RuntimeError(
            f"Failed to load checkpoint '{ckpt_path}' into {type(model).__name__}.\n"
            f"  Missing keys (in model, absent from checkpoint): {len(result.missing_keys)}\n"
            f"  Unexpected keys (in checkpoint, absent from model): {len(result.unexpected_keys)}\n"
            f"{hint}\n"
            f"  First missing keys:    {result.missing_keys[:5]}\n"
            f"  First unexpected keys: {result.unexpected_keys[:5]}\n"
            f"Original error: {e}"
        ) from e


def _detect_cond_embed_dim(ckpt_path, device, default=256):
    """Infer the ConditionalDDPM architecture a checkpoint needs.

    Returns the cond_embed_dim to construct the model with so that BOTH legacy
    bare-2D checkpoints (no cond_embed MLP, cross_attention_dim=2) and new
    embed-MLP checkpoints load without manual flags:
      - 0  -> legacy checkpoint (no cond_embed.* params)
      - N  -> embed-MLP checkpoint; N is read from the first Linear's out_features
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'ema_model_state_dict' in checkpoint:
        state_dict = checkpoint['ema_model_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    embed_w = None
    for k, v in state_dict.items():
        if (k[7:] if k.startswith('module.') else k) == 'cond_embed.0.weight':
            embed_w = v
            break
    if embed_w is None:
        return 0  # legacy: no embedding MLP
    return int(embed_w.shape[0])


class GeneratorInpainter:
    def __init__(self, cond_model_path=None, uncond_model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Load conditional DDPM. Detect the checkpoint's architecture first so a
        # legacy bare-2D checkpoint and a new embed-MLP checkpoint both load
        # cleanly into a matching model (default to the new arch when none given).
        cond_embed_dim = 256
        if cond_model_path:
            cond_embed_dim = _detect_cond_embed_dim(cond_model_path, device)
            if cond_embed_dim == 0:
                print(f"Note: '{cond_model_path}' is a legacy bare-2D conditional checkpoint "
                      f"(no embed MLP). Loading it with the legacy architecture.")
        self.cond_model = ConditionalDDPM(cond_embed_dim=cond_embed_dim).to(device)
        if cond_model_path:
            _load_ddpm_checkpoint(self.cond_model, cond_model_path, device)
        self.cond_model.eval()

        # Load unconditional DDPM
        self.uncond_model = UnconditionalDDPM().to(device)
        if uncond_model_path:
            _load_ddpm_checkpoint(self.uncond_model, uncond_model_path, device)
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

    def _forward_to_noisier(self, x, from_t, to_t):
        """Forward-diffuse an already-noised sample from noise level `from_t` up
        to a noisier level `to_t` (to_t > from_t) using the closed-form
        q(x_{to_t} | x_{from_t}) of the diffusion forward process.

        This is the correct step for RePaint time-travel resampling. It must NOT
        be confused with scheduler.add_noise, which assumes its input is the
        clean x_0 and would therefore re-noise an already-noised sample all the
        way to near-pure noise.
        """
        alphas_cumprod = self.scheduler.alphas_cumprod.to(x.device)
        a_from = alphas_cumprod[from_t]
        a_to = alphas_cumprod[to_t]
        ratio = a_to / a_from  # in (0, 1] since to_t is the noisier level
        noise = torch.randn_like(x)
        return torch.sqrt(ratio) * x + torch.sqrt(1.0 - ratio) * noise

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

            # RePaint time-travel: every jump_length steps, jump back up the noise
            # schedule and resample jump_n_sample-1 extra times so the model can
            # harmonise the boundary.
            if jump_n_sample > 1 and (i + 1) % jump_length == 0 and i + 1 < len(timesteps):
                jump_back_idx = max(0, i + 1 - jump_length)
                t_jump = timesteps[jump_back_idx]
                # x_t currently sits at the noise level we just stepped down to.
                cur_level = timesteps[i + 1]

                print(f"  [Time-Travel] Jumping back from step {i} to {jump_back_idx} ({jump_n_sample - 1} times)")

                for _ in range(jump_n_sample - 1):
                    # Forward-diffuse x_t from its CURRENT noise level up to t_jump.
                    # Using scheduler.add_noise here is wrong: it treats x_t as the
                    # clean x_0 and would blast it to near-pure noise every jump.
                    x_t = self._forward_to_noisier(x_t, cur_level, t_jump)
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
