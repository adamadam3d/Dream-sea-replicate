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
            self.cond_model.load_state_dict(torch.load(cond_model_path, map_location=device))
        self.cond_model.eval()

        # Load unconditional DDPM
        self.uncond_model = UnconditionalDDPM().to(device)
        if uncond_model_path:
            self.uncond_model.load_state_dict(torch.load(uncond_model_path, map_location=device))
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

        # Move entire latent grid to device once before looping
        if not isinstance(latent_grid, torch.Tensor):
            latent_grid_tensor = torch.from_numpy(latent_grid).float().to(self.device)
        else:
            latent_grid_tensor = latent_grid.float().to(self.device)

        print(f"Generating {N}x{N} grid patches...")
        for y in range(N):
            for x in range(N):
                patch = self.generate_patch(latent_grid_tensor[y, x])
                patch_grid[y, x] = patch[0]

        return patch_grid

    @torch.no_grad()
    def repaint_inpaint(self, image_input, mask, num_inference_steps=1000, jump_length=10, jump_n_sample=10):
        """
        Implementation of RePaint using the unconditional DDPM.
        image_input: The image with known and unknown regions (1, 4, 224, 224)
        mask: 1 for known regions, 0 for unknown regions (1, 1, 224, 224)
        """
        image_input = torch.from_numpy(image_input).to(self.device) if not isinstance(image_input, torch.Tensor) else image_input
        mask = torch.from_numpy(mask).to(self.device) if not isinstance(mask, torch.Tensor) else mask

        # Start from random noise
        x_t = torch.randn_like(image_input)

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        timesteps = self.scheduler.timesteps

        # RePaint requires time travel, but for simplicity in this basic loop
        # we do a standard masked DDPM sampling step to simulate inpainting.
        # A full RePaint implementation involves loops of forward and backward steps.
        # Here we do a simplified version: at each step, mix known image + noise with predicted unknown.
        for i, t in enumerate(timesteps):
            # Known region
            if i < len(timesteps) - 1:
                noise = torch.randn_like(image_input)
                t_next = timesteps[i+1]
                x_known_t = self.scheduler.add_noise(image_input, noise, torch.tensor([t_next], device=self.device))
            else:
                x_known_t = image_input

            # Unknown region prediction
            noise_pred = self.uncond_model(x_t, t)
            x_unknown_t_minus_1 = self.scheduler.step(noise_pred, t, x_t).prev_sample

            # Combine
            x_t = mask * x_known_t + (1 - mask) * x_unknown_t_minus_1

        return x_t.cpu().numpy()

    def stitch_and_inpaint(self, patch_grid, overlap_size=32):
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

        # Now, identifying seams to inpaint. We mask out the overlaps.
        print("Inpainting seams using parallelizable RePaint...")
        # Create a seam mask: where weight_map > 1, it's an overlap. We want to inpaint these regions.
        # We will mask out the seams (set to 0) and keep the rest as known (set to 1).
        mask = (weight_map <= 1.0).astype(np.float32)

        # To avoid massive memory usage for a full large canvas, we would typically do this in tiles.
        # For this prototype, we'll extract tiles along the seams, inpaint them, and put them back.
        # Parallelizable pattern: we can inpaint all horizontal seams in parallel, then vertical.

        # As an example, we inpaint patches over the horizontal seams
        for y in range(1, N):
            y_seam_center = y * (patch_size - overlap_size) + overlap_size // 2
            y_start = y_seam_center - patch_size // 2
            y_end = y_seam_center + patch_size // 2

            for x in range(N):
                # Sample a patch around the horizontal seam
                x_start = x * (patch_size - overlap_size)
                x_end = x_start + patch_size

                # Check boundaries
                if y_start < 0 or y_end > canvas_size or x_start < 0 or x_end > canvas_size:
                    continue

                local_canvas = canvas[:, y_start:y_end, x_start:x_end]
                local_mask = mask[:, y_start:y_end, x_start:x_end]

                # Expand dims for batch
                local_canvas = np.expand_dims(local_canvas, 0)
                local_mask = np.expand_dims(local_mask, 0)

                # Inpaint
                inpainted = self.repaint_inpaint(local_canvas, local_mask)

                # Update canvas with inpainted region only where mask is 0
                canvas[:, y_start:y_end, x_start:x_end] = \
                    local_mask[0] * local_canvas[0] + (1 - local_mask[0]) * inpainted[0]

                # Update mask so it's considered known for vertical passes
                mask[:, y_start:y_end, x_start:x_end] = 1.0

        # Repeat for vertical seams
        for x in range(1, N):
            x_seam_center = x * (patch_size - overlap_size) + overlap_size // 2
            x_start = x_seam_center - patch_size // 2
            x_end = x_seam_center + patch_size // 2

            for y in range(N):
                y_start = y * (patch_size - overlap_size)
                y_end = y_start + patch_size

                if y_start < 0 or y_end > canvas_size or x_start < 0 or x_end > canvas_size:
                    continue

                local_canvas = canvas[:, y_start:y_end, x_start:x_end]
                local_mask = mask[:, y_start:y_end, x_start:x_end]

                local_canvas = np.expand_dims(local_canvas, 0)
                local_mask = np.expand_dims(local_mask, 0)

                inpainted = self.repaint_inpaint(local_canvas, local_mask)

                canvas[:, y_start:y_end, x_start:x_end] = \
                    local_mask[0] * local_canvas[0] + (1 - local_mask[0]) * inpainted[0]
                mask[:, y_start:y_end, x_start:x_end] = 1.0

        return canvas
