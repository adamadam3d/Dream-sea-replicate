# DreamSea Architecture Overview

DreamSea is a generative model pipeline designed to create photorealistic 3D underwater terrains from unannotated 2D RGB imagery. This document outlines the inner workings of the pipeline, which is loosely based on the concepts from the paper *Infinite Leagues Under the Sea* by Zhang et al.

## 1. Data Preprocessing (`data_preprocessing.py`)

The pipeline relies on generating a robust 2D representation before lifting to 3D. Since standard underwater datasets lack depth, we construct an RGBD representation and extract semantic features:

- **Depth Estimation:** We utilize **Depth Anything v2** to generate high-quality relative depth maps from standard 2D RGB inputs. These maps are normalized and concatenated with the RGB images to form a 4-channel RGBD tensor.
- **Feature Extraction & Reduction:** We use **DINOv2** to extract rich, zero-shot semantic features from the images. Because DINOv2 embeddings are high-dimensional, we use Principal Component Analysis (PCA) via scikit-learn to reduce these embeddings to just 2 principal components. These 2D vectors serve as the latent conditioning variables for our diffusion models.

## 2. Diffusion Models (`models.py`)

We employ two Custom U-Net Denoising Diffusion Probabilistic Models (DDPMs):

- **Conditional DDPM:** Trained to denoise 4-channel RGBD data, conditioned on the 2D PCA-reduced DINOv2 features. The conditioning is integrated via Cross-Attention layers, guiding the model to generate specific semantic structures (e.g., coral reefs vs. sandy bottoms).
- **Unconditional DDPM:** A standard U-Net trained on the same RGBD data, but without any conditioning. This model acts as a powerful prior for inpainting and global structural cohesion.

## 3. Fractal Latent Grid Generation (`fractal_latent.py`)

To generate vast, cohesive terrains, we cannot generate a massive image in one shot due to memory constraints. Instead, we generate the environment piece-by-piece based on a smoothly varying latent landscape.

- **Diamond-Square Algorithm:** We implement the Diamond-Square algorithm to recursively generate a 2D grid of spatial latent embeddings. This ensures that adjacent patches in the final terrain correspond to similar semantic features (e.g., a reef smoothly transitions into sand), mimicking natural landscape distribution.
- **Calibration into PCA space:** The raw fractal field is approximately `N(0, 1)`, whereas the diffusion model was conditioned on the training PCA distribution. `scale_latent_grid` applies a *fixed* affine map `grid * std + mean` (from `latent_stats.json`, or the built-in `DEFAULT_LATENT_STATS`) to land the grid in-distribution. The map is deliberately fixed rather than per-grid min/max normalized, so the roughness/diversity of a given field is preserved (a calm field stays clustered; a varied one spreads out) instead of every grid being stretched to fill the full range.

## 4. Generation & Inpainting (`generation_inpainting.py`)

This module translates the abstract latent grid into a tangible, dense RGBD map.

- **Patch Generation:** We evaluate the Conditional DDPM over every node in the generated fractal latent grid, producing individual 4-channel RGBD patches.
- **Stitching with RePaint:** Simple tiling leads to harsh seams. We stitch these patches together, overlapping them slightly. To seamlessly blend the overlaps, we use the **RePaint** framework combined with our Unconditional DDPM. We mask the overlapping regions (treating them as "unknown") and use RePaint to infer the missing content based on the "known" centers of the patches. This is done using a parallelizable inpainting pattern to maintain efficiency.

## 5. 3D Gaussian Splatting & SDS Optimization (`gs_sds_optimization.py`, `gs_sds_optimization_v2.py`)

The final step converts the dense, global RGBD map into a fully explorable 3D asset.

- **Unprojection:** Using known camera intrinsics, we unproject the global RGBD map into a dense 3D Point Cloud.
- **3DGS Initialization:** We initialize a 3D Gaussian Splatting (3DGS) model based on this point cloud. Crucially, to prevent memory overflow during optimization on massive terrains, the 3D positions of the Gaussians are explicitly **frozen** (paper §3.4). Only the appearance parameters are learned: per-Gaussian **scaling**, **rotation** (quaternion), **opacity**, and **RGB colour** (`features_dc`, a single DC colour per Gaussian — not full spherical harmonics).
- **SDS Optimization:** We refine the Gaussians with **Score Distillation Sampling (SDS) loss** (DreamFusion). A 2D diffusion model acts as a *prior/critic*: a rendered view is noised and the model's predicted score tells us how to nudge the render toward the realistic-seafloor distribution; that gradient flows back through the differentiable renderer into the Gaussian parameters. We use the **unconditional** DDPM as the prior, because a rendered view spans many stitched patches with no single conditioning latent — the unconditional model scores "is this realistic seafloor in general," which is the well-defined question for a multi-patch view.

### Two SDS renderers

| Module | Renderer | Notes |
| --- | --- | --- |
| `gs_sds_optimization_v2.py` (`--rasterizer gsplat`, **default**) | Real differentiable Gaussian rasterizer (gsplat) | Faithful to the paper (Eq. 4–5). Renders **random near-nadir novel viewpoints** each step (a tight cone around top-down, matching the diffusion prior's training views), so scaling/rotation/opacity/colour all receive gradients. Anneals the diffusion timestep coarse→fine. Requires CUDA + `pip install gsplat`. |
| `gs_sds_optimization.py` (`--rasterizer scatter`) | Simplified single-view top-down scatter | A dependency-free fallback. It only uses colour and opacity (no true covariance), so it optimizes just those terms; scaling/rotation keep their depth-based initialization. |

> Both paths feed gradients only through the render (stop-grad on the diffusion
> score), and the printed value is the SDS *surrogate* — it can be negative and
> need not decrease monotonically, so it is not a convergence signal.
