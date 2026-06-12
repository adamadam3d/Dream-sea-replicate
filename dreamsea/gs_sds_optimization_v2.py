"""
Multi-view SDS optimization with a real differentiable Gaussian rasterizer.

This is the faithful version of the SDS stage (paper Eq. 4-5). Unlike the
simplified scatter renderer in gs_sds_optimization.py, it:
  - renders with a true differentiable 3DGS rasterizer (gsplat), so the Gaussian
    covariance (scaling/rotation) and positions all affect the image;
  - renders RANDOM novel viewpoints each step (within a near-nadir cone, since
    the diffusion prior only ever saw top-down seafloor), which is what lets SDS
    actually refine 3D appearance/consistency;
  - frames each view at roughly one training-patch footprint (not the whole
    scene), so the texture scale the prior sees matches what it was trained on;
  - restricts the diffusion timestep to a LOW range (refinement). The Gaussians
    are initialized from a complete stitched RGBD map, so high-t SDS would pull
    the scene toward the dataset mean and erase the init;
  - anchors the appearance parameters to their initialization so a mismatched
    score cannot drift the scene arbitrarily far.

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


def sample_topdown_camera(positions, target=None, fov_deg=60.0, device='cuda',
                          elev_min=75.0, elev_max=90.0, frame_extent=None):
    """Sample a camera within a near-nadir cone looking at the target (or terrain centre if None).

    elev is measured from horizontal: 90 deg = straight down. Keep elev_min high
    (~60) so views stay close to top-down, matching the diffusion prior. Widening
    this cone pushes the prior out of distribution and reintroduces artifacts.

    frame_extent: world-space extent the view should roughly cover. Pass the
    extent of ONE training patch so the rendered texture scale matches what the
    diffusion prior saw. If None, frames the whole scene (out of distribution
    for multi-patch scenes — the prior then smooths fine detail away).
    """
    if target is None:
        center = positions.mean(0)
    else:
        center = torch.as_tensor(target, dtype=torch.float32, device=device)

    extent = (positions.max(0).values - positions.min(0).values).norm()
    if frame_extent is not None:
        extent = torch.as_tensor(frame_extent, dtype=torch.float32, device=device)
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
    scales = torch.exp(model.scaling)                        # (N,3) live, positive (log-space parameterization)
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
    alphas = _alphas[0, ..., 0]                              # (H,W) accumulated opacity
    fg_mask = alphas > 1e-4

    if fg_mask.any():
        fg_depth = depth[fg_mask]
        d_min = fg_depth.min()
        d_max = fg_depth.max()
        if d_max > d_min:
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
        else:
            depth_norm = torch.zeros_like(depth)
    else:
        depth_norm = torch.zeros_like(depth)

    # Set background pixels to 1.0 (furthest depth) in [0, 1] range
    depth_norm = torch.where(fg_mask, depth_norm, torch.ones_like(depth_norm))

    rgbd = torch.cat([rgb, depth_norm[None]], dim=0)         # (4,H,W) in [0,1]
    return rgbd * 2.0 - 1.0                                  # -> [-1,1] to match training


# --------------------------------------------------------------------------- #
# SDS loss                                                                      #
# --------------------------------------------------------------------------- #
def compute_sds_loss(diffusion_model, scheduler, rendered,
                     t_lo=0.02, t_hi=0.30, cond=None, guidance=0.0):
    """DreamFusion SDS surrogate. `rendered` is (1, C, H, W), differentiable.

    The default timestep range is LOW ([0.02, 0.30]) because this stage refines
    an already-initialized scene. At high t the score points toward the dataset
    mean image, which overwrites a good initialization instead of sharpening it.
    """
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
        elif cond is not None:
            # Plain conditional SDS: score with the actual local condition.
            # (Previously this fell through to the zero-condition branch below, so
            # with the default guidance=0 every view was scored against the
            # dataset-MEAN latent — SDS then pulled all regions toward the mean
            # appearance instead of refining toward the local one.)
            eps = diffusion_model(x_t, t, encoder_hidden_states=cond)
        else:
            # Check if conditional model. If so, pass dummy zero condition to prevent signature crash
            if hasattr(diffusion_model, 'unet') and hasattr(diffusion_model.unet, 'config') and 'cross_attention_dim' in diffusion_model.unet.config and diffusion_model.unet.config.cross_attention_dim is not None:
                dummy_cond = torch.zeros((x_t.shape[0], 1, 2), device=x_t.device)
                eps = diffusion_model(x_t, t, encoder_hidden_states=dummy_cond)
            else:
                try:
                    eps = diffusion_model(x_t, t)
                except TypeError:
                    dummy_cond = torch.zeros((x_t.shape[0], 1, 2), device=x_t.device)
                    eps = diffusion_model(x_t, t, encoder_hidden_states=dummy_cond)

    # Use uniform weighting to prevent vanishing gradients at small timesteps
    w = 1.0
    grad = w * (eps - noise)
    # Stop-grad on the score; differentiate only through the render using mean to stabilize gradients
    return (grad.detach() * rendered).mean()


# --------------------------------------------------------------------------- #
# Optimization loop                                                             #
# --------------------------------------------------------------------------- #
def optimize_3dgs_sds_multiview(model, diffusion_model, scheduler, iterations=1000,
                                views_per_iter=4, fov_deg=60.0, H=224, W=224,
                                elev_min=60.0, cond=None, guidance=0.0, rgb_only=True,
                                frame_fraction=None, anchor_weight=1.0):
    """Multi-view SDS with a real Gaussian rasterizer.

    cond / guidance: with the conditional model, the per-view local latent is
    always used to score the render; guidance>0 additionally applies CFG against
    a zero-latent branch. NB: guided SDS is unreliable here — the conditional UNet was trained WITHOUT condition
    dropout and the PCA latents have mean ~0, so a zero latent is the MEAN
    condition, not a null branch. CFG then extrapolates between two conditional
    predictions and produces saturated artifacts. Leave guidance at 0 unless the
    model is retrained with condition dropout.
    rgb_only: detach the depth channel before the SDS loss (default). The
    rendered per-view min-max depth does not match the per-image relative depth
    the prior was trained on, and its gradient wrecks scaling/opacity; the
    depth is still rendered as context for the 4-channel UNet, and geometry is
    carried by the frozen-position init.
    frame_fraction: fraction of the scene extent each rendered view should
    cover. Pass patch_size / map_size so one view ~ one training patch; None
    frames the whole scene (only in-distribution for single-patch scenes).
    anchor_weight: weight of an MSE anchor pulling color/opacity/scaling back
    toward their initialization, bounding how far SDS can drift the scene.
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

    # World-space extent one rendered view should frame (patch footprint).
    frame_extent = None
    if frame_fraction is not None:
        scene_extent = (model.positions.max(0).values - model.positions.min(0).values).norm()
        frame_extent = (scene_extent * min(frame_fraction, 1.0)).item()

    # Snapshot the initialization (in activated space) for the anchor loss.
    init_colors = torch.tanh(model.features_dc).detach().clone()
    init_opacity = torch.sigmoid(model.opacity).detach().clone()
    init_scaling = model.scaling.detach().clone()

    print(f"Starting multi-view SDS ({iterations} iters, {views_per_iter} views/iter, "
          f"rasterizer=gsplat, rgb_only={rgb_only}, guidance={guidance}, "
          f"frame_fraction={frame_fraction}, anchor_weight={anchor_weight})...")
    for i in range(iterations):
        # Anneal the upper timestep limit from 0.30 down to 0.10 (cosine). The
        # whole range stays LOW: the scene is initialized from a complete RGBD
        # map, so SDS only refines — high-t scores would pull the appearance
        # toward the dataset mean and erase the init.
        frac = i / max(iterations - 1, 1)
        cos_val = math.cos(math.pi * frac)  # +1 down to -1
        t_hi = 0.10 + 0.20 * (cos_val + 1.0) / 2.0

        loss = 0.0
        for _ in range(views_per_iter):
            # Sample a random target point on the terrain surface to serve as camera look-at target
            num_points = model.positions.shape[0]
            target_idx = np.random.randint(0, num_points)
            target_pos = model.positions[target_idx]
            
            # Retrieve per-point DINO/PCA conditioning coordinate
            local_cond = None
            if hasattr(model, 'point_conds') and model.point_conds is not None:
                # Shape expected by model: (1, 1, 2)
                local_cond = model.point_conds[target_idx].view(1, 1, -1)
            elif cond is not None:
                local_cond = cond

            vm = sample_topdown_camera(model.positions, target=target_pos, fov_deg=fov_deg,
                                       device=device, elev_min=elev_min, frame_extent=frame_extent)
            rgbd = render_rgbd(rasterization, model, vm, K, H, W)
            if rgb_only:
                # The 4-channel UNet still needs a depth channel as context, but
                # detaching it blocks the mismatched depth gradient (per-view
                # min-max depth vs. the prior's per-image relative depth) from
                # corrupting scaling/opacity.
                rgbd = torch.cat([rgbd[:3], rgbd[3:].detach()], dim=0)
            loss = loss + compute_sds_loss(
                diffusion_model, scheduler, rgbd.unsqueeze(0),
                t_hi=t_hi, cond=local_cond, guidance=guidance,
            )
        loss = loss / views_per_iter

        # Anchor to the initialization so a mismatched score can't drift the
        # scene arbitrarily far from the (already correct) unprojected map.
        if anchor_weight > 0.0:
            anchor = (
                torch.nn.functional.mse_loss(torch.tanh(model.features_dc), init_colors)
                + torch.nn.functional.mse_loss(torch.sigmoid(model.opacity), init_opacity)
                + torch.nn.functional.mse_loss(model.scaling, init_scaling)
            )
            loss = loss + anchor_weight * anchor

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
