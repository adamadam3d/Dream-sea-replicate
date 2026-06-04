# DreamSea

DreamSea is a generative model pipeline designed to create photorealistic 3D underwater terrains from unannotated 2D RGB imagery, based on the concepts from the paper *Infinite Leagues Under the Sea* by Zhang et al.

## Documentation

For comprehensive documentation, please refer to the following guides:

- [**Installation Guide**](docs/INSTALLATION.md): Hardware prerequisites and step-by-step setup instructions.
- [**Usage Guide**](docs/USAGE.md): How to run the pipeline, start training, and interact with individual modules.
- [**Architecture Overview**](docs/ARCHITECTURE.md): Detailed explanation of the pipeline's technical design (Preprocessing, Diffusion, Fractal Latents, RePaint, 3DGS & SDS).

## Architecture Overview

The pipeline consists of several modular steps that convert standard 2D RGB imagery into a rich 3D representation using advanced generative modeling techniques:

1. **Data Preprocessing**:
   - **Depth Estimation**: Uses Depth Anything v2 to generate relative depth maps from RGB images, concatenating them into 4-channel RGBD tensors.
   - **Feature Extraction**: Uses DINOv2 to extract zero-shot semantic features.
   - **Dimensionality Reduction**: Applies PCA via scikit-learn to reduce the DINOv2 embeddings to 2 principal components.

2. **Diffusion Models**:
   - **Conditional DDPM**: A Custom U-Net model trained on 4-channel RGBD data, conditioned on the 2D PCA-reduced DINOv2 features via Cross-Attention layers.
   - **Unconditional DDPM**: A standard U-Net model trained on the same data without any conditioning.

3. **Fractal Latent Field Generation**:
   - Implements the Diamond-Square algorithm to recursively generate a 2D grid of spatial latent embeddings (representing the 2D PCA space).

4. **Generation & Inpainting (RePaint)**:
   - Evaluates the Conditional DDPM over the generated fractal latent grid to create individual RGBD patches.
   - Stitches these adjacent spatial patches into a dense, global RGBD map using the RePaint framework. The Unconditional DDPM is used with a parallelizable inpainting pattern to seamlessly fill the overlaps.

5. **3D Gaussian Splatting (3DGS) Optimization**:
   - Unprojects the final stitched RGBD map into a dense 3D Point Cloud.
   - Initializes a 3D Gaussian Splatting model. The 3D positions of the Gaussians are explicitly frozen to prevent memory overflow.
   - Optimizes the appearance parameters (scaling, rotation, opacity, and per-Gaussian RGB color) using Score Distillation Sampling (SDS) loss from the 2D unconditional diffusion prior. The default renderer is a real multi-view gsplat rasterizer (`--rasterizer gsplat`), with a dependency-free `scatter` fallback.

## Requirements

Due to the memory-intensive nature of both Denoising Diffusion Probabilistic Models (especially during RePaint inpainting on potentially high-resolution canvases) and 3D Gaussian Splatting, this pipeline requires a robust GPU setup.

### Hardware
- **GPU**: At least one NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, RTX 4090, A10G, or A100).
- **System RAM**: 64GB+ recommended to hold the dataset and point cloud operations comfortably.

### Software dependencies
- Python 3.10+
- PyTorch (with CUDA)
- Hugging Face `transformers`
- Hugging Face `diffusers`
- `numpy`, `scikit-learn`, `Pillow`, `torchvision`

## Quick Start

To run a dummy end-to-end integration test of the pipeline (verifying that all the components load, connect, and optimize without crashing):

```bash
PYTHONPATH=. python dreamsea/main.py
```

This will run an abbreviated version of the generation, stitching, and 3DGS optimization steps on a tiny grid footprint.

To generate a real scene from trained checkpoints and export a `.ply`:

```bash
python -m dreamsea.generate_3dgs \
  --cond_ckpt   checkpoints/conditional_epoch_2000.pt \
  --uncond_ckpt checkpoints/unconditional_epoch_2000.pt \
  --grid_size 3 \
  --latent_stats_path /path/to/processed/data/latent_stats.json \
  --sds_iterations 100 \
  --output_dir outputs/3dgs_gen
```

Pass separate conditional and unconditional checkpoints (never the same file for
both). `--sds_iterations 0` skips SDS; `--rasterizer scatter` avoids the gsplat
dependency. For full flag documentation and troubleshooting, see the
[Usage Guide](docs/USAGE.md).
