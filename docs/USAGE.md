# DreamSea Usage Guide

This guide explains how to use the various components of the DreamSea pipeline, from running the end-to-end integration test to interacting with individual modules.

## Running the End-to-End Pipeline

To verify that the entire pipeline works and all components are correctly integrated, you can run the main execution script. This script performs an abbreviated, "dummy" version of the generation, stitching, and 3D Gaussian Splatting optimization steps on a small grid footprint.

```bash
PYTHONPATH=. python dreamsea/main.py
```

This will run through:
1. **Preprocessing Components Initialization**
2. **Fractal Latent Grid Generation**
3. **Generation and Inpainting of RGBD Patches**
4. **3D Gaussian Splatting Initialization**
5. **SDS Optimization (dummy run of 5 iterations)**

## Training the Models

If you wish to train the Conditional and Unconditional DDPMs from scratch or fine-tune them on your own dataset, you can use the `train.py` script as a starting point.

The default script sets up a dummy dataset and runs a mock training loop to verify that the backpropagation and loss functions work correctly without crashing.

```bash
PYTHONPATH=. python dreamsea/train.py
```

### Preparing for Real-World Training

To train the models to a useful point, you must perform several extensive data preparation and tuning steps:

#### 1. Dataset Preparation
- **Collect Data**: Gather thousands of high-quality underwater RGB images.
- **Preprocess**: You must pass every image through the `DataPreprocessor` (see section below).
  - Save the resulting 4-channel RGBD tensors to disk (e.g., as `.pt` or `.npy` files).
  - Save the corresponding 2D PCA-reduced DINOv2 feature vectors.
- **Custom DataLoader**: Replace the `DummyDataset` in `train.py` with a PyTorch `Dataset` that streams your precomputed RGBD tensors and conditions from disk.

#### 2. Hyperparameter Tuning
A dummy run trains for 5 epochs with a learning rate of `1e-4`. To achieve meaningful results:
- **Epochs**: Increase the number of epochs significantly (e.g., 500 - 1000+).
- **Batch Size**: Maximize your batch size based on your VRAM (e.g., 16, 32, or 64). If VRAM is limited, use gradient accumulation.
- **Timesteps**: `num_train_timesteps` is set to 1000. You may experiment with cosine vs linear schedules.

#### 3. Monitoring and Checkpointing
- **Loss Logging**: Integrate a logging framework like Weights & Biases (`wandb`) or TensorBoard to track the MSE loss over time.
- **Checkpoints**: Add code within the training loop to periodically save the `state_dict` of both the `cond_model` and `uncond_model` to disk (e.g., `torch.save(cond_model.state_dict(), f"checkpoints/cond_epoch_{epoch}.pt")`).
- **Validation**: It is highly recommended to add a validation loop that periodically samples an image using the DDPM to visually track generation quality.

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
