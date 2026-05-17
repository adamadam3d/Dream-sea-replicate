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
uncond_model = UnconditionalDDPM(in_channels=3, out_channels=3).to(device)
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 4. Optimize
gs_opt.optimize_3dgs_sds(gs_model, uncond_model, scheduler, iterations=100)
```
