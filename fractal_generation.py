import numpy as np
import matplotlib.pyplot as plt

def diamond_square_2d(size, base_roughness=0.6, dim=2):
    """
    Generates a 2D grid of latent vectors using the Diamond-Square algorithm.
    This creates spatially coherent yet diverse "fractal" latents.

    Args:
        size (int): Size of the grid (must be 2^n + 1).
        base_roughness (float): Initial scaling factor for the random noise.
        dim (int): Dimensionality of the latent vector at each point (default 2 for DINOv2 PCA).

    Returns:
        np.ndarray: Grid of shape (size, size, dim) containing the latents.
    """
    # Check if size is 2^n + 1
    if (size - 1) & (size - 2) != 0:
        raise ValueError("Size must be of the form 2^n + 1 (e.g., 5, 9, 17, 33, 65)")

    grid = np.zeros((size, size, dim))

    # Initialize corners with random values from normal distribution
    grid[0, 0] = np.random.randn(dim)
    grid[0, size-1] = np.random.randn(dim)
    grid[size-1, 0] = np.random.randn(dim)
    grid[size-1, size-1] = np.random.randn(dim)

    step = size - 1
    roughness = base_roughness

    while step > 1:
        half_step = step // 2

        # Diamond step
        # For each square in the array, find the midpoint and set it to the average
        # of the four corners plus a random value.
        for y in range(0, size - 1, step):
            for x in range(0, size - 1, step):
                # Corners
                top_left = grid[y, x]
                top_right = grid[y, x + step]
                bottom_left = grid[y + step, x]
                bottom_right = grid[y + step, x + step]

                # Average + noise
                avg = (top_left + top_right + bottom_left + bottom_right) / 4.0
                noise = np.random.randn(dim) * roughness

                grid[y + half_step, x + half_step] = avg + noise

        # Square step
        # For each diamond in the array, find the midpoint and set it to the average
        # of the four corners plus a random value.
        for y in range(0, size, half_step):
            # offset x to align with diamond centers
            offset = 0 if (y // half_step) % 2 == 0 else half_step
            for x in range(offset, size, step):
                count = 0
                avg = np.zeros(dim)

                # Top
                if y - half_step >= 0:
                    avg += grid[y - half_step, x]
                    count += 1
                # Bottom
                if y + half_step < size:
                    avg += grid[y + half_step, x]
                    count += 1
                # Left
                if x - half_step >= 0:
                    avg += grid[y, x - half_step]
                    count += 1
                # Right
                if x + half_step < size:
                    avg += grid[y, x + half_step]
                    count += 1

                avg = avg / count
                noise = np.random.randn(dim) * roughness

                grid[y, x] = avg + noise

        # Reduce roughness for next iteration (decay)
        step //= 2
        roughness *= 0.5  # Standard decay, can be adjusted

    return grid

if __name__ == "__main__":
    print("Fractal Latent Generation initialized.")
    # Example usage:
    # grid_size = 17 # 2^4 + 1
    # latents_grid = diamond_square_2d(size=grid_size, base_roughness=1.0, dim=2)
    # print(f"Generated latent grid of shape: {latents_grid.shape}")

    # Optional: Visualize one dimension of the generated latents
    # plt.imshow(latents_grid[:,:,0], cmap='viridis')
    # plt.colorbar()
    # plt.title("Diamond-Square Fractal Latent (Dim 0)")
    # plt.show()
