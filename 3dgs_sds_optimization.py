import torch
import torch.nn as nn
import torch.nn.functional as F

class DreamSea3DGS(nn.Module):
    def __init__(self, rgbd_map, num_points_threshold=100000):
        super().__init__()

        print("Unprojecting RGBD map to 3D point cloud...")
        points, colors = self.unproject_rgbd(rgbd_map)

        # Subsample if too many points
        if points.shape[0] > num_points_threshold:
            indices = torch.randperm(points.shape[0])[:num_points_threshold]
            points = points[indices]
            colors = colors[indices]

        print(f"Initialized {points.shape[0]} 3D Gaussians.")

        # 1. Freeze 3D positions (p_i)
        self.register_buffer('positions', points)

        # 2. Learnable properties: covariance (scaling & rotation), opacity, radiance (color/SH)
        # Scaling
        self.log_scales = nn.Parameter(torch.zeros(points.shape[0], 3))
        # Rotation (Quaternions)
        self.rotations = nn.Parameter(torch.zeros(points.shape[0], 4))
        self.rotations.data[:, 0] = 1.0 # Initialize to identity
        # Opacity (inverse sigmoid)
        self.opacities = nn.Parameter(torch.zeros(points.shape[0], 1))
        # Radiance (Spherical Harmonics or just base color for simplicity)
        self.base_colors = nn.Parameter(colors)

    def unproject_rgbd(self, rgbd_map):
        """
        Unprojects the 2D RGBD map into a 3D point cloud.
        Assuming simple orthographic or pinhole projection depending on use case.
        Here we map (u, v, depth) -> (x, y, z).
        """
        # rgbd_map shape: (4, H, W)
        rgb = rgbd_map[:3, :, :]
        depth = rgbd_map[3, :, :]

        C, H, W = rgb.shape

        # Create meshgrid
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

        # Scale coordinates to [-1, 1]
        x_norm = (x.float() / W) * 2.0 - 1.0
        y_norm = (y.float() / H) * 2.0 - 1.0

        # Unproject: (x_norm, y_norm, depth)
        # Adjust depth scaling as necessary
        z = depth * 5.0 # Scale depth

        # Valid depth mask
        valid_mask = depth > 0

        points = torch.stack([x_norm[valid_mask], y_norm[valid_mask], z[valid_mask]], dim=-1)
        colors = rgb[:, valid_mask].permute(1, 0)

        return points, colors

    def forward(self, camera_pose):
        """
        Dummy forward pass representing 3DGS rendering.
        In reality, this would project the 3D Gaussians onto the 2D plane
        given the camera_pose and composite them.
        """
        # Rendered image shape: (1, 3, H, W)
        rendered_image = torch.rand(1, 3, 256, 256).to(self.positions.device)
        return rendered_image

def score_distillation_sampling_loss(rendered_image, diffusion_model, noise_scheduler, text_embeddings=None):
    """
    Computes the Score Distillation Sampling (SDS) loss.
    ∇_θ L_SDS = E_{t, ε} [ w(t) * (ε_θ(z_t; y, t) - ε) * ∂z_t/∂θ ]

    In practice, we don't backprop through the diffusion model.
    We compute the noise prediction and treat it as a constant target.

    Args:
        rendered_image (torch.Tensor): Image rendered from 3DGS (B, C, H, W).
        diffusion_model: Pre-trained 2D diffusion model acting as prior.
        noise_scheduler: DDPMScheduler.
    """
    # 1. Add random noise to the rendered image
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=rendered_image.device).long()
    noise = torch.randn_like(rendered_image)
    noisy_image = noise_scheduler.add_noise(rendered_image, noise, t)

    # 2. Predict noise using the diffusion model (with condition if any)
    with torch.no_grad():
        if text_embeddings is not None:
            noise_pred = diffusion_model(noisy_image, t, encoder_hidden_states=text_embeddings).sample
        else:
            noise_pred = diffusion_model(noisy_image, t).sample

    # 3. Compute gradient w.r.t rendered image
    # w(t) is often set to alpha_t or just 1.0 for simplicity
    w_t = 1.0
    grad = w_t * (noise_pred - noise)

    # 4. We want to push the rendered image such that the noise prediction matches the actual noise.
    # We apply this gradient to the rendered image.
    # The PyTorch way to do this without writing a custom autograd function
    # is to construct a loss term where taking the gradient w.r.t rendered_image yields `grad`.

    # detach grad to treat it as constant target
    grad = grad.detach()

    # The loss is (rendered_image * grad).sum(), so that d(loss)/d(rendered_image) = grad
    sds_loss = torch.sum(rendered_image * grad)

    return sds_loss

def optimize_3dgs(rgbd_map, diffusion_model, noise_scheduler, num_iters=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize 3DGS Model
    gs_model = DreamSea3DGS(rgbd_map).to(device)

    # Optimizer (only optimizing learnable params, position is frozen)
    optimizer = torch.optim.Adam(gs_model.parameters(), lr=0.01)

    diffusion_model.to(device)
    diffusion_model.eval() # Diffusion model is frozen

    print("Starting 3DGS Optimization via SDS...")
    for i in range(num_iters):
        # 1. Sample a random camera pose
        random_pose = None # Dummy

        # 2. Render the scene from this pose
        rendered_image = gs_model(random_pose)

        # 3. Compute SDS Loss using 2D Diffusion prior
        loss = score_distillation_sampling_loss(rendered_image, diffusion_model, noise_scheduler)

        # 4. Backpropagate and optimize Gaussian parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iters} - SDS Loss: {loss.item():.4f}")

    print("3DGS Optimization Complete.")
    return gs_model

if __name__ == "__main__":
    print("3DGS SDS Optimization script initialized.")
    # Example usage:
    # dummy_rgbd = torch.rand(4, 256, 256)
    # trained_gs = optimize_3dgs(dummy_rgbd, dummy_diffusion_model, dummy_scheduler)
