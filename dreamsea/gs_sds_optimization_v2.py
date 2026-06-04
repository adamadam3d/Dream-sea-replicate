"""
Multi-view SDS optimization with a real differentiable Gaussian rasterizer.

This is the faithful version of the SDS stage (paper Eq. 4-5). Unlike the
simplified scatter renderer in gs_sds_optimization.py, it:
  - renders with a true differentiable 3DGS rasterizer (gsplat), so the Gaussian
    covariance (scaling/rotation) and positions all affect the image;
  - renders RANDOM novel viewpoints each step (within a near-nadir cone, since
    the diffusion prior only ever saw top-down seafloor), which is what lets SDS
    actually refine 3D appearance/consistency;
  - anneals the diffusion timestep from coarse to fine over training.

Positions are kept frozen (paper §3.4); scaling, rotation, opacity and colour
are optimized.

Requires gsplat (CUDA only):  pip install gsplat
It is imported lazily so the rest of the package works without it.

Reuses the GaussianSplattingModel from gs_sds_optimization.py unchanged, so
export_ply.py stays compatible.
"""

import math

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Camera helpers (OpenCV convention: x-right, y-down, z-forward into the scene) #
# --------------------------------------------------------------------------- #
def intrinsics(fov_deg, W, H, device):
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2.0)
    return torch.tensor([[f, 0, W / 2.0],
                         [0, f, H / 2.0],
                         [0, 0, 1.0]], dtype=torch.float32, device=device)


def look_at(cam_pos, target, up=(0.0, 0.0, 1.0), device='cuda'):
    """Returns a 4x4 world-to-camera matrix (OpenCV convention)."""
    cam_pos = torch.as_tensor(cam_pos, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=device)
    up = torch.as_tensor(up, dtype=torch.float32, device=device)

    f = target - cam_pos
    f = f / (f.norm() + 1e-8)                 # forward  (+z)
    r = torch.linalg.cross(f, up)
    r = r / (r.norm() + 1e-8)                 # right    (+x)
    d = torch.linalg.cross(f, r)              # down     (+y)

    R = torch.stack([r, d, f], dim=0)         # rows are camera axes in world frame
    vm = torch.eye(4, device=device)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ cam_pos
    return vm


def sample_topdown_camera(positions, fov_deg=60.0, device='cuda',
                          elev_min=60.0, elev_max=90.0):
    """Sample a camera within a near-nadir cone looking at the terrain centre.

    elev is measured from horizontal: 90 deg = straight down. Keep elev_min high
    (~60) so views stay close to top-down, matching the diffusion prior. Widening
    this cone pushes the prior out of distribution and reintroduces artifacts.
    """
    center = positions.mean(0)
    extent = (positions.max(0).values - positions.min(0).values).norm()
    dist = 0.6 * extent / math.tan(math.radians(fov_deg) / 2.0)

    elev = math.radians(np.random.uniform(elev_min, elev_max))
    azim = math.radians(np.random.uniform(0.0, 360.0))
    offset = torch.tensor([
        dist * math.cos(elev) * math.cos(azim),
        dist * math.cos(elev) * math.sin(azim),
        dist * math.sin(elev),
    ], dtype=torch.float32, device=device)
    return look_at(center + offset, center, up=(0.0, 0.0, 1.0), device=device)


# --------------------------------------------------------------------------- #
# Differentiable RGBD render                                                    #
# --------------------------------------------------------------------------- #
def render_rgbd(rasterization, model, viewmat, K, H=224, W=224, rgb_only=False):
    """Render a (4, H, W) RGBD image in [-1, 1] (or (3, H, W) if rgb_only).

    `rasterization` is gsplat.rasterization, passed in so this module imports
    cleanly without gsplat installed.
    """
    means = model.positions                                  # (N,3) frozen
    quats = model.rotation                                   # (N,4) live
    scales = model.scaling.clamp_min(1e-6)                   # (N,3) live, positive
    opacities = torch.sigmoid(model.opacity).squeeze(-1)     # (N,)
    colors01 = (torch.tanh(model.features_dc) + 1.0) / 2.0   # (N,3) -> [0,1]

    mode = "RGB" if rgb_only else "RGB+ED"
    out, _alphas, _meta = rasterization(
        means, quats, scales, opacities, colors01,
        viewmat[None], K[None], W, H, render_mode=mode,
    )                                                        # out: (1, H, W, 3) or (1, H, W, 4)

    rgb = out[0, ..., :3].permute(2, 0, 1)                   # (3,H,W) in [0,1]
    if rgb_only:
        return rgb * 2.0 - 1.0

    depth = out[0, ..., 3]                                   # (H,W) expected depth
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)  # per-view -> [0,1]
    rgbd = torch.cat([rgb, depth[None]], dim=0)             # (4,H,W) in [0,1]
    return rgbd * 2.0 - 1.0                                  # -> [-1,1] to match training


# --------------------------------------------------------------------------- #
# SDS loss                                                                      #
# --------------------------------------------------------------------------- #
def compute_sds_loss(diffusion_model, scheduler, rendered,
                     t_lo=0.02, t_hi=0.98, cond=None, guidance=0.0):
    """DreamFusion SDS surrogate. `rendered` is (1, C, H, W), differentiable."""
    T = scheduler.config.num_train_timesteps
    t = torch.randint(int(T * t_lo), max(int(T * t_hi), int(T * t_lo) + 1),
                      (1,), device=rendered.device).long()

    noise = torch.randn_like(rendered)
    x_t = scheduler.add_noise(rendered.detach(), noise, t)

    with torch.no_grad():
        if cond is not None and guidance > 0.0:
            # Guided SDS (conditional model only). NB: the conditional UNet was not
            # trained with condition dropout, so a zero latent is only an approximate
            # null branch — treat `guidance` as a heuristic sharpener.
            eps_c = diffusion_model(x_t, t, encoder_hidden_states=cond)
            eps_u = diffusion_model(x_t, t, encoder_hidden_states=torch.zeros_like(cond))
            eps = eps_u + guidance * (eps_c - eps_u)
        else:
            eps = diffusion_model(x_t, t)

    w = 1.0 - scheduler.alphas_cumprod.to(t.device)[t]
    grad = w * (eps - noise)
    # Stop-grad on the score; differentiate only through the render.
    return (grad.detach() * rendered).sum()


# --------------------------------------------------------------------------- #
# Optimization loop                                                             #
# --------------------------------------------------------------------------- #
def optimize_3dgs_sds_multiview(model, diffusion_model, scheduler, iterations=1000,
                                views_per_iter=4, fov_deg=60.0, H=224, W=224,
                                elev_min=60.0, cond=None, guidance=0.0, rgb_only=False):
    """Multi-view SDS with a real Gaussian rasterizer.

    cond / guidance: pass the conditional model + its latent (shape (1,1,2)) and
    guidance>0 to do guided SDS; otherwise plain unconditional SDS.
    rgb_only: skip the depth channel (more robust — avoids the relative-vs-metric
    depth domain mismatch; geometry is carried by the frozen-position init).
    """
    try:
        from gsplat import rasterization
    except ImportError as e:
        raise ImportError(
            "gsplat is required for --rasterizer gsplat (multi-view SDS). "
            "Install it with `pip install gsplat` (CUDA toolkit required)."
        ) from e

    device = model.positions.device
    if device.type != "cuda":
        raise RuntimeError("gsplat rasterization requires a CUDA device; "
                           f"the model is on '{device}'. Use --device cuda.")

    # Positions stay frozen (paper §3.4). With a real rasterizer, scaling/rotation
    # finally receive gradients, so they are optimized here.
    optimizer = torch.optim.Adam([
        {'params': [model.scaling],     'lr': 5e-3},
        {'params': [model.rotation],    'lr': 1e-3},
        {'params': [model.opacity],     'lr': 5e-2},
        {'params': [model.features_dc], 'lr': 1e-2},
    ])

    K = intrinsics(fov_deg, W, H, device)

    print(f"Starting multi-view SDS ({iterations} iters, {views_per_iter} views/iter, "
          f"rasterizer=gsplat, rgb_only={rgb_only}, guidance={guidance})...")
    for i in range(iterations):
        # Anneal the upper timestep from coarse (0.98) to fine (0.50).
        frac = i / max(iterations - 1, 1)
        t_hi = 0.98 - 0.48 * frac

        loss = 0.0
        for _ in range(views_per_iter):
            vm = sample_topdown_camera(model.positions, fov_deg, device, elev_min=elev_min)
            rgbd = render_rgbd(rasterization, model, vm, K, H, W, rgb_only=rgb_only)
            loss = loss + compute_sds_loss(
                diffusion_model, scheduler, rgbd.unsqueeze(0),
                t_hi=t_hi, cond=cond, guidance=guidance,
            )
        loss = loss / views_per_iter

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Keep quaternions unit-norm.
        with torch.no_grad():
            model.rotation.data = torch.nn.functional.normalize(model.rotation.data, dim=-1)

        if (i + 1) % 50 == 0:
            # NB: this is the SDS *surrogate* value, not a real loss — it can be
            # negative and need not decrease monotonically.
            print(f"  Iter {i + 1}/{iterations} | SDS surrogate: {loss.item():.4f}")

    print("Multi-view SDS optimization complete.")
