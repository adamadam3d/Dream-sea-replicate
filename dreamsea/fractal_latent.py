import numpy as np


def scale_latent_grid(grid, latent_min, latent_max):
    """
    Rescales a fractal grid (sampled from ~N(0,1)) into the observed PCA
    coordinate range of the training data.  Pass latent_min / latent_max from
    the latent_stats.json saved by preprocess_dataset.py.
    """
    latent_min = np.array(latent_min, dtype=np.float32)
    latent_max = np.array(latent_max, dtype=np.float32)
    grid_min = grid.min(axis=(0, 1), keepdims=True)
    grid_max = grid.max(axis=(0, 1), keepdims=True)
    normalized = (grid - grid_min) / np.maximum(grid_max - grid_min, 1e-8)
    return normalized * (latent_max - latent_min) + latent_min


def diamond_square_2d(size, roughness=0.5, seed=None):
    """
    Generates a 2D grid of latent embeddings using the Diamond-Square algorithm.
    Size must be 2^n + 1. Returns a grid of shape (size, size, 2) since we want
    to generate 2D PCA latent vectors.
    """
    if seed is not None:
        np.random.seed(seed)

    if size < 3:
        raise ValueError("Size must be at least 3 (2^1 + 1)")
    if (size - 1) & (size - 2) != 0:
        raise ValueError("Size must be 2^n + 1")

    grid = np.zeros((size, size, 2), dtype=np.float32)

    # Initialize the four corners with random values from N(0, 1)
    grid[0, 0] = np.random.randn(2)
    grid[0, size - 1] = np.random.randn(2)
    grid[size - 1, 0] = np.random.randn(2)
    grid[size - 1, size - 1] = np.random.randn(2)

    step = size - 1
    scale = 1.0

    while step > 1:
        half_step = step // 2

        # Diamond step: calculate center of each square
        for y in range(0, size - 1, step):
            for x in range(0, size - 1, step):
                # Calculate mean of corners
                avg = (grid[y, x] +
                       grid[y, x + step] +
                       grid[y + step, x] +
                       grid[y + step, x + step]) / 4.0

                # Add random noise s*sigma where sigma ~ N(0, 1)
                grid[y + half_step, x + half_step] = avg + np.random.randn(2) * scale * roughness

        # Square step: calculate centers of each diamond
        for y in range(0, size, half_step):
            # Offset x by half_step every other row
            x_start = half_step if (y // half_step) % 2 == 0 else 0
            for x in range(x_start, size, step):
                sum_vals = np.zeros(2, dtype=np.float32)
                count = 0

                if x >= half_step:
                    sum_vals += grid[y, x - half_step]
                    count += 1
                if x + half_step < size:
                    sum_vals += grid[y, x + half_step]
                    count += 1
                if y >= half_step:
                    sum_vals += grid[y - half_step, x]
                    count += 1
                if y + half_step < size:
                    sum_vals += grid[y + half_step, x]
                    count += 1

                avg = sum_vals / count
                grid[y, x] = avg + np.random.randn(2) * scale * roughness

        # Reduce the random scale factor
        scale *= (2 ** (-roughness))
        step = half_step

    return grid
