# DreamSea Installation Guide

This guide will walk you through setting up the DreamSea pipeline. Given the advanced generative models and the 3D generation components, ensuring that your hardware and software meet the requirements is crucial.

## Hardware Requirements

DreamSea leverages complex 2D diffusion models and 3D Gaussian Splatting, both of which are memory-intensive.

- **GPU**: At least one NVIDIA GPU with **24GB+ VRAM**. Recommended cards include:
  - RTX 3090 or RTX 4090
  - A10G or A100
- **System RAM**: **64GB+** is highly recommended to comfortably load datasets, point clouds, and handle RePaint operations.
- **Storage**: SSD storage is recommended for fast model loading and dataset access.

## Software Prerequisites

- **OS**: Linux (Ubuntu 20.04/22.04 recommended) or Windows with WSL2.
- **CUDA Toolkit**: Compatible with your NVIDIA driver and the PyTorch version you intend to install (CUDA 11.8 or 12.1 is typical).

## Step-by-Step Installation

### 1. Set Up the Python Environment

We strongly recommend using [Conda](https://docs.conda.io/en/latest/) to manage your environment, although `venv` can also be used. The project requires Python 3.10 or higher.

**Using Conda:**

```bash
# Create a new conda environment named 'dreamsea'
conda create -n dreamsea python=3.10 -y

# Activate the environment
conda activate dreamsea
```

**Using venv:**

```bash
# Create a virtual environment named 'dreamsea-env'
python3 -m venv dreamsea-env

# Activate the environment
source dreamsea-env/bin/activate
```

### 2. Install PyTorch

Install PyTorch along with `torchvision` and `torchaudio`. Make sure to select the build that matches your installed CUDA version. For example, for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*Note: Please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the most up-to-date commands corresponding to your specific OS and CUDA version.*

### 3. Install Required Libraries

DreamSea relies on several core libraries including `diffusers`, `transformers`, and basic data science packages.

Install the required packages using `pip`:

```bash
pip install diffusers transformers numpy scikit-learn Pillow
```

### 4. Clone the Repository

Clone the DreamSea repository to your local machine:

```bash
git clone https://github.com/your-username/dreamsea.git
cd dreamsea
```

*(Replace the URL with the actual repository URL if applicable).*

### 5. Verify the Installation

To verify that the installation was successful and all components load correctly, you can run the dummy end-to-end integration test:

```bash
PYTHONPATH=. python dreamsea/main.py
```

If the installation is correct, this script will run an abbreviated version of the generation, stitching, and 3DGS optimization steps without crashing.
