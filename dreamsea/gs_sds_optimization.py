import torch
import torch.nn as nn
import numpy as np

class GaussianSplattingModel(nn.Module):
    def __init__(self, point_cloud_positions, point_cloud_colors, point_cloud_conds=None, upscale_factor=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        # Freeze 3D positions to prevent memory overflow
        self.positions = torch.tensor(point_cloud_positions, dtype=torch.float32, requires_grad=False).to(self.device)

        N = self.positions.shape[0]

        # Optimize appearance only (covariance, opacity, radiance)
        # Covariance represented via scaling and rotation (quaternions)
        # Scale proportionally to depth to ensure Gaussians overlap to form a solid surface
        # Log-space scaling is used here for stability in optimization
        z_vals = self.positions[:, 2:3].detach()
        # Scale proportionally to depth; use a larger multiplier and a floor to prevent gaps/invisibility
        # Adjust base scale size by upscale_factor to match the increased point density
        base_scale = torch.clamp(z_vals / (80.0 * upscale_factor), min=0.01 / upscale_factor)
        self.scaling = nn.Parameter(torch.log(base_scale.repeat(1, 3)))
        
        self.rotation = nn.Parameter(torch.zeros((N, 4), dtype=torch.float32).to(self.device))
        self.rotation.data[:, 0] = 1.0 # Initialize real part of quaternion to 1

        # Opacity
        # Initialize to 5.0 so that sigmoid(5.0) = 0.993 (fully opaque for solid reconstruction)
        self.opacity = nn.Parameter(torch.ones((N, 1), dtype=torch.float32).to(self.device) * 5.0)

        # Radiance/Colors (Spherical Harmonics, but for simplicity we use RGB features here)
        # Inverse transform colors to pre-tanh space so the initial render matches the input image
        target_colors = point_cloud_colors * 2.0 - 1.0
        target_colors = np.clip(target_colors, -0.999, 0.999)
        init_features_dc = np.arctanh(target_colors)
        self.features_dc = nn.Parameter(torch.tensor(init_features_dc, dtype=torch.float32).to(self.device))

        # Store per-point conditioning vectors for conditional SDS
        if point_cloud_conds is not None:
            self.point_conds = torch.tensor(point_cloud_conds, dtype=torch.float32, requires_grad=False).to(self.device)
        else:
            self.point_conds = torch.zeros((N, 2), dtype=torch.float32, requires_grad=False).to(self.device)

    def forward(self, camera=None, canvas_size=224):
        """
        Differentiable top-down orthographic projection onto a (1, 4, H, W) RGBD canvas.
        Gradients flow through features_dc and opacity so SDS can optimize appearance.
        (In a full implementation this would call the CUDA Gaussian rasterizer.)
        """
        opacities = torch.sigmoid(self.opacity)   # (N, 1)
        colors = torch.tanh(self.features_dc)     # (N, 3), bounded [-1, 1]

        x_pos = self.positions[:, 0]
        y_pos = self.positions[:, 1]
        z_pos = self.positions[:, 2]

        x_norm = (x_pos - x_pos.min()) / (x_pos.max() - x_pos.min() + 1e-8)
        y_norm = (y_pos - y_pos.min()) / (y_pos.max() - y_pos.min() + 1e-8)

        px = (x_norm * (canvas_size - 1)).long().clamp(0, canvas_size - 1)
        py = (y_norm * (canvas_size - 1)).long().clamp(0, canvas_size - 1)
        flat_idx = py * canvas_size + px  # (N,)

        # Depth channel: normalize z to [-1, 1]
        z_norm = (z_pos - z_pos.min()) / (z_pos.max() - z_pos.min() + 1e-8) * 2.0 - 1.0

        # Opacity-weighted RGBD values — this is the differentiable part
        rgba = torch.cat([colors, z_norm.unsqueeze(1)], dim=1)  # (N, 4)
        weighted = rgba * opacities                              # (N, 4)

        # Scatter onto canvas with index_put (accumulate=True supports autograd)
        flat_size = canvas_size * canvas_size
        channels = []
        for c in range(4):
            ch = torch.zeros(flat_size, device=self.device)
            ch = ch.index_put((flat_idx,), weighted[:, c], accumulate=True)
            channels.append(ch)

        den = torch.zeros(flat_size, device=self.device)
        den = den.index_put((flat_idx,), opacities[:, 0], accumulate=True)

        # Mask pixels with no projecting Gaussians (den == 0)
        mask = (den > 1e-8).unsqueeze(0)  # (1, flat_size)
        raw_rendered = torch.stack(channels, dim=0) / (den.unsqueeze(0) + 1e-8)

        # Background color: -1.0 (black), Background depth: 1.0 (furthest depth)
        bg_values = torch.tensor([-1.0, -1.0, -1.0, 1.0], device=self.device).view(4, 1)
        rendered = torch.where(mask, raw_rendered, bg_values)

        return rendered.view(4, canvas_size, canvas_size).unsqueeze(0)  # (1, 4, H, W)

def create_point_cloud_from_rgbd(rgbd_map, fov=60.0, latent_grid=None, patch_size=224, overlap_size=32):
    """
    Converts a stitched RGBD map into a 3D point cloud and optionally retrieves the DINO/PCA conditioning vectors per-point.
    rgbd_map: (4, H, W) numpy array, channels are RGB + Depth.
    Accepts values in either [0, 1] or [-1, 1] range (auto-detected).
    """
    C, H, W = rgbd_map.shape

    # Auto-detect and denormalize from [-1, 1] to [0, 1] if needed
    if rgbd_map.min() < -0.5:
        rgbd_map = (rgbd_map + 1.0) / 2.0
        rgbd_map = np.clip(rgbd_map, 0.0, 1.0)

    rgb = rgbd_map[:3, :, :].transpose(1, 2, 0) # (H, W, 3)
    depth = rgbd_map[3, :, :] # (H, W)

    # Intrinsics
    focal = 0.5 * W / np.tan(0.5 * np.radians(fov))
    cx, cy = W / 2.0, H / 2.0

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # Unproject
    z = depth * 10.0 + 1.0 # arbitrary scaling to move away from camera
    x_3d = (x - cx) * z / focal
    y_3d = (y - cy) * z / focal

    positions = np.stack([x_3d, y_3d, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    # Filter invalid depths
    valid = z.flatten() > 0.1
    
    positions_valid = positions[valid]
    colors_valid = colors[valid]
    
    # Extract conditioning vectors per-point via bilinear interpolation from latent_grid
    conds_valid = None
    if latent_grid is not None:
        N_y, N_x, _ = latent_grid.shape
        stride = patch_size - overlap_size
        
        # Calculate continuous grid coordinates for each pixel (y, x)
        gy = y / stride
        gx = x / stride
        
        # Clamp to bounds to prevent out-of-index
        gy = np.clip(gy, 0.0, max(N_y - 1.0 - 1e-5, 0.0))
        gx = np.clip(gx, 0.0, max(N_x - 1.0 - 1e-5, 0.0))
        
        gy0 = gy.astype(np.int32)
        gy1 = np.minimum(gy0 + 1, N_y - 1)
        gx0 = gx.astype(np.int32)
        gx1 = np.minimum(gx0 + 1, N_x - 1)
        
        wy = gy - gy0
        wx = gx - gx0
        
        # Retrieve endpoints
        c00 = latent_grid[gy0, gx0]  # (H, W, 2)
        c01 = latent_grid[gy0, gx1]
        c10 = latent_grid[gy1, gx0]
        c11 = latent_grid[gy1, gx1]
        
        # Interpolate
        wy = np.expand_dims(wy, -1)
        wx = np.expand_dims(wx, -1)
        
        c0 = c00 * (1 - wx) + c01 * wx
        c1 = c10 * (1 - wx) + c11 * wx
        conds = c0 * (1 - wy) + c1 * wy  # (H, W, 2)
        
        conds_flat = conds.reshape(-1, 2)
        conds_valid = conds_flat[valid]
        
    return positions_valid, colors_valid, conds_valid


def compute_sds_loss(diffusion_model, scheduler, rendered_image, text_embeddings=None):
    """
    Score Distillation Sampling loss (DreamFusion).
    rendered_image: (1, 4, H, W) — must be a differentiable function of the 3DGS parameters.
    """
    # Low timestep range: the Gaussians are initialized from a complete RGBD
    # map, so SDS only refines. High-t scores point toward the dataset mean
    # image and erase the initialization.
    t = torch.randint(
        int(scheduler.config.num_train_timesteps * 0.02),
        int(scheduler.config.num_train_timesteps * 0.30),
        (1,), device=rendered_image.device
    ).long()

    noise = torch.randn_like(rendered_image)
    # Detach before add_noise: the SDS gradient flows through rendered_image
    # directly (see loss below), not through the forward diffusion path.
    noisy_image = scheduler.add_noise(rendered_image.detach(), noise, t)

    with torch.no_grad():
        if text_embeddings is not None:
            noise_pred = diffusion_model(noisy_image, t, encoder_hidden_states=text_embeddings)
        elif hasattr(diffusion_model, 'unet') and hasattr(diffusion_model.unet, 'config') and 'cross_attention_dim' in diffusion_model.unet.config and diffusion_model.unet.config.cross_attention_dim is not None:
            # Fallback to zero embedding for conditional model to prevent missing argument crashes
            dummy_cond = torch.zeros((noisy_image.shape[0], 1, 2), device=noisy_image.device)
            noise_pred = diffusion_model(noisy_image, t, encoder_hidden_states=dummy_cond)
        else:
            # Fallback for models without unet wrapper, e.g. mock or test classes
            try:
                noise_pred = diffusion_model(noisy_image, t)
            except TypeError:
                dummy_cond = torch.zeros((noisy_image.shape[0], 1, 2), device=noisy_image.device)
                noise_pred = diffusion_model(noisy_image, t, encoder_hidden_states=dummy_cond)

    # Use uniform weighting to prevent vanishing gradients at small timesteps
    w = 1.0
    grad = w * (noise_pred - noise)

    # Use mean instead of sum to stabilize gradients and prevent opacity/color saturation
    loss = torch.mean(grad.detach() * rendered_image)
    return loss


def optimize_3dgs_sds(model, diffusion_model, scheduler, iterations=1000):
    """
    Optimization loop for 3DGS using SDS loss.
    """
    # The simplified scatter renderer in GaussianSplattingModel.forward only uses
    # color (features_dc) and opacity; it does NOT use the Gaussian covariance
    # (scaling/rotation), so those receive no gradient. Including them here was
    # misleading — we only optimize the appearance terms the renderer actually
    # differentiates through. scaling/rotation keep their depth-based init (which
    # export_ply.py reads). A faithful covariance optimization would require a real
    # differentiable Gaussian rasterizer and multi-view rendering (paper Eq. 4).
    optimizer = torch.optim.Adam([
        {'params': [model.opacity], 'lr': 0.05},
        {'params': [model.features_dc], 'lr': 0.01}
    ])

    # Note: positions are frozen by design (requires_grad=False); scaling/rotation
    # are not optimized because the simplified renderer doesn't use them.

    print("Starting 3DGS SDS optimization...")
    for i in range(iterations):
        # 1. Random camera pose (simplified)
        camera_pose = None

        # 2. Render
        rendered_image = model(camera_pose)

        # 3. Compute SDS Loss
        loss = compute_sds_loss(diffusion_model, scheduler, rendered_image)

        # 4. Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{iterations} | SDS Loss: {loss.item():.4f}")

    print("Optimization complete.")
