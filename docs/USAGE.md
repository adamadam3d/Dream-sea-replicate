# DreamSea Usage Guide

This guide explains how to use the various components of the DreamSea pipeline, from running the end-to-end integration test to interacting with individual modules.

## Running the End-to-End Pipeline

To verify that the entire pipeline works and all components are correctly integrated, you can run the main execution script. This script performs an abbreviated, "dummy" version of the generation, stitching, and 3D Gaussian Splatting optimization steps on a small grid footprint.

```bash
python -m dreamsea.main
```

This will run through:
1. **Preprocessing Components Initialization**
2. **Fractal Latent Grid Generation**
3. **Generation and Inpainting of RGBD Patches**
4. **3D Gaussian Splatting Initialization**
5. **SDS Optimization (dummy run of 5 iterations)**

## Training the Models

You can train the Conditional and Unconditional DDPMs from scratch on your own dataset.

### 1. Data Preprocessing

First, you need to preprocess your directory of raw RGB images. This script will extract DINOv2 conditions, estimate depth maps, fit a PCA model (saved to `pca_model.pkl`), and save the normalized tensors to disk.

```bash
python -m dreamsea.preprocess_dataset --input_dir /path/to/raw/images --output_dir /path/to/processed/data
```

### 2. Training the DDPMs

Once preprocessed, you can train both models. The training script automatically loads the data, applies `[-1, 1]` diffusion normalization, and monitors for NaN/Inf divergence.

**Train Conditional Model (Patch Generation):**
```bash
python -m dreamsea.train \
  --data_dir /path/to/processed/data \
  --model_type conditional \
  --epochs 2000 \
  --batch_size 12 \
  --save_every 50
```

**Train Unconditional Model (RePaint Stitching):**
```bash
python -m dreamsea.train \
  --data_dir /path/to/processed/data \
  --model_type unconditional \
  --epochs 2000 \
  --batch_size 12 \
  --save_every 50
```

### 3. Generating Samples

You can test your trained conditional model and generate sample RGBD images using the provided generation script:

```bash
python -m dreamsea.generate_sample \
  --cond_model_path checkpoints/conditional_epoch_2000.pt \
  --output_dir samples/ \
  --num_inference_steps 1000
```

## Generating a 3D Scene (`generate_3dgs`)

`generate_3dgs` is the full production pipeline: it builds a fractal latent grid,
generates and stitches RGBD patches, lifts them to a 3D Gaussian Splatting model,
optionally refines it with SDS, and exports a `.ply` for viewing.

```bash
python -m dreamsea.generate_3dgs \
  --cond_ckpt   checkpoints/conditional_epoch_2000.pt \
  --uncond_ckpt checkpoints/unconditional_epoch_2000.pt \
  --grid_size 3 \
  --roughness 0.5 \
  --latent_stats_path /path/to/processed/data/latent_stats.json \
  --sds_iterations 100 \
  --rasterizer gsplat \
  --output_dir outputs/3dgs_gen
```

Outputs (in `--output_dir`): `global_rgbd_map.pt`, `global_rgb_map.png`,
`final_gs_model.pt`, and `final_gs_model.ply`.

### Key arguments

| Flag | Default | Purpose |
| --- | --- | --- |
| `--cond_ckpt` | *(required)* | Conditional DDPM checkpoint — generates the RGBD patches. |
| `--uncond_ckpt` | `None` | Unconditional DDPM checkpoint. Required only when **not** using `--use_conditional_stitching`, or when `--sds_iterations > 0` (SDS uses it as the prior). Do **not** pass the conditional checkpoint here. |
| `--grid_size` | `3` | Latent grid size; must be `2^n + 1` (e.g. `3`, `5`, `9`) or `1` for a single patch. |
| `--roughness` | `0.5` | Diamond-Square roughness — higher = more varied terrain. |
| `--latent_stats_path` | `None` | `latent_stats.json` from preprocessing, used to calibrate the fractal grid into the training PCA range. If omitted, the built-in `DEFAULT_LATENT_STATS` are used. **Pass the file, not the `conditions/` directory.** |
| `--use_conditional_stitching` | off | Inpaint seams with the conditional model instead of the unconditional one. The paper uses the unconditional model (fewer boundary artifacts), which is the default. |
| `--sds_iterations` | `100` | SDS refinement steps. Set `0` to skip SDS entirely (and skip needing `--uncond_ckpt` when also using conditional stitching). |
| `--rasterizer` | `gsplat` | `gsplat` = faithful multi-view differentiable rasterizer (needs `pip install gsplat` + CUDA). `scatter` = simplified single-view fallback, no extra deps. |
| `--device` | auto | `cuda` or `cpu`. `gsplat` SDS requires `cuda`. |

> **Latent stats live at the preprocessing root**, e.g.
> `/path/to/processed/data/latent_stats.json` (saved next to `pca_model.pkl`),
> not inside the `conditions/` subfolder.

## Interacting with Individual Modules

You can also use individual modules of the DreamSea pipeline in your own scripts.

### 1. Data Preprocessing

The `DataPreprocessor` class handles converting RGB images to RGBD tensors using Depth Anything v2, and extracting/reducing DINOv2 features using PCA.

```python
from dreamsea.data_preprocessing import DataPreprocessor

# Initialize preprocessor (loads foundation models)
preprocessor = DataPreprocessor(device='cuda')

# Convert an RGB image to a 4-channel RGBD tensor
rgbd_tensor = preprocessor.process_rgb_to_rgbd("path/to/your/image.jpg")

# Extract DINOv2 features and reduce them to 2D using PCA
features = preprocessor.extract_and_reduce_dino_features(["path/to/image1.jpg", "path/to/image2.jpg"])
```

### 2. Fractal Latent Grid Generation

The Diamond-Square algorithm is used to generate a 2D grid of latent embeddings.

```python
from dreamsea.fractal_latent import diamond_square_2d

# Generate a latent grid of size 5x5 (size must be 2^n + 1)
grid_size = 5
latent_grid = diamond_square_2d(grid_size, roughness=0.5, seed=42)
print(latent_grid.shape) # Output: (5, 5, 2)
```

### 3. Generation and Inpainting

The `GeneratorInpainter` handles generating patches and stitching them together.

```python
from dreamsea.generation_inpainting import GeneratorInpainter

# Initialize generator
generator = GeneratorInpainter(device='cuda')

# Generate a grid of patches from a latent grid
patch_grid = generator.generate_grid(latent_grid)

# Stitch patches into a dense global RGBD map
global_map = generator.stitch_and_inpaint(patch_grid, overlap_size=32)
```

### 4. 3D Gaussian Splatting (3DGS) Optimization

You can initialize a 3DGS model from a global RGBD map and optimize it using SDS.

```python
import dreamsea.gs_sds_optimization as gs_opt
from dreamsea.models import UnconditionalDDPM
from diffusers import DDPMScheduler
import torch

# 1. Unproject RGBD map to 3D point cloud
positions, colors = gs_opt.create_point_cloud_from_rgbd(global_map)

# 2. Initialize 3DGS model
gs_model = gs_opt.GaussianSplattingModel(positions, colors)

# 3. Setup models for SDS loss computation
device = 'cuda' if torch.cuda.is_available() else 'cpu'
uncond_model = UnconditionalDDPM().to(device)  # 4-channel RGBD model (in/out = 4)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 4. Optimize (simplified single-view scatter renderer)
gs_opt.optimize_3dgs_sds(gs_model, uncond_model, scheduler, iterations=100)

# 4b. Or use the faithful multi-view gsplat renderer (CUDA + `pip install gsplat`)
from dreamsea.gs_sds_optimization_v2 import optimize_3dgs_sds_multiview
optimize_3dgs_sds_multiview(gs_model, uncond_model, scheduler, iterations=100)
```

## Troubleshooting

- **`RuntimeError: ... missing/unexpected keys` when loading a checkpoint.**
  You likely passed a conditional checkpoint to `--uncond_ckpt` (or vice versa).
  The two models have different architectures (the conditional one has
  cross-attention `transformer_blocks`); the loader checks this strictly and
  refuses to silently load partial weights — partial loading leaves random
  weights and produces noise. Train and pass a separate unconditional checkpoint,
  or set `--sds_iterations 0 --use_conditional_stitching` to avoid needing one.

- **`IsADirectoryError` on `--latent_stats_path`.** Point it at the
  `latent_stats.json` *file* at the preprocessing root, not the `conditions/`
  directory.

- **`ImportError: gsplat is required` / `gsplat requires a CUDA device`.** The
  default `--rasterizer gsplat` needs `pip install gsplat` and `--device cuda`.
  Use `--rasterizer scatter` for a no-dependency CPU/GPU fallback.

- **Output looks low quality / blurry.** Most often an undertrained checkpoint
  (the paper trains ~2000 epochs), a `--grid_size 1` single patch, or an
  uncalibrated latent grid — pass `--latent_stats_path` (or rely on the built-in
  defaults) so the fractal grid lands in the training PCA range.
