import numpy as np

def generate_fractal_latents(size: int, base_scale: float = 0.6) -> np.ndarray:
    """
    Generates a 2D grid of 2D latent vectors using the Diamond-Square Algorithm.
    The resulting grid has dimensions (size, size, 2).
    Note: `size` must be of the form 2^n + 1 (e.g., 5, 9, 17, 33, ...).

    Args:
        size (int): The dimension of the grid (must be 2^n + 1).
        base_scale (float): The initial scale for the random noise added at each step.

    Returns:
        np.ndarray: A grid of shape (size, size, 2) containing 2D latent vectors.
    """
    if (size - 1) & (size - 2) != 0:
        raise ValueError(f"Size must be of the form 2^n + 1, but got {size}.")

    # Initialize the grid with zeros. Shape: (H, W, 2)
    grid = np.zeros((size, size, 2), dtype=np.float32)

    # 1. Initialize the 4 corners with random 2D latent vectors
    grid[0, 0] = np.random.randn(2)
    grid[0, size - 1] = np.random.randn(2)
    grid[size - 1, 0] = np.random.randn(2)
    grid[size - 1, size - 1] = np.random.randn(2)

    step_size = size - 1
    scale = base_scale

    # Recursively perform Diamond and Square steps
    while step_size > 1:
        half_step = step_size // 2

        # --- Diamond Step ---
        # For each square in the grid, calculate the midpoint
        for y in range(0, size - 1, step_size):
            for x in range(0, size - 1, step_size):
                # Coordinates of the corners
                top_left = grid[y, x]
                top_right = grid[y, x + step_size]
                bottom_left = grid[y + step_size, x]
                bottom_right = grid[y + step_size, x + step_size]

                # Average the 4 corners and add random noise
                avg = (top_left + top_right + bottom_left + bottom_right) / 4.0
                noise = np.random.randn(2) * scale

                # Assign to the midpoint (center of the diamond)
                grid[y + half_step, x + half_step] = avg + noise

        # --- Square Step ---
        # For each diamond in the grid, calculate the midpoint
        for y in range(0, size, half_step):
            # Offset x start position based on row
            offset = half_step if (y // half_step) % 2 == 0 else 0
            for x in range(offset, size, step_size):
                count = 0
                total = np.zeros(2, dtype=np.float32)

                # Check top neighbor
                if y >= half_step:
                    total += grid[y - half_step, x]
                    count += 1
                # Check bottom neighbor
                if y + half_step < size:
                    total += grid[y + half_step, x]
                    count += 1
                # Check left neighbor
                if x >= half_step:
                    total += grid[y, x - half_step]
                    count += 1
                # Check right neighbor
                if x + half_step < size:
                    total += grid[y, x + half_step]
                    count += 1

                # Average valid neighbors and add random noise
                avg = total / count
                noise = np.random.randn(2) * scale

                grid[y, x] = avg + noise

        # Reduce step size and scale for the next iteration
        step_size = half_step
        scale /= 2.0  # The scale decays over iterations to create fractal detail

    return grid

if __name__ == "__main__":
    # Example: Generate a 9x9 grid of latents
    grid_size = 9
    latent_grid = generate_fractal_latents(size=grid_size, base_scale=0.6)

    print(f"Generated fractal latent grid of shape: {latent_grid.shape}")
    print(f"Latent at (0, 0): {latent_grid[0, 0]}")
    print(f"Latent at center: {latent_grid[grid_size//2, grid_size//2]}")
