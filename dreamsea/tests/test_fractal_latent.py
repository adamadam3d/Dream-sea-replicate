import pytest
import numpy as np
from dreamsea.fractal_latent import diamond_square_2d

def test_diamond_square_2d_shape():
    """Test that valid sizes produce the correct output shape."""
    valid_sizes = [5, 9, 17, 33]  # 2^n + 1 sizes
    for size in valid_sizes:
        grid = diamond_square_2d(size)
        assert grid.shape == (size, size, 2), f"Failed for size {size}"
        assert grid.dtype == np.float32

def test_diamond_square_2d_invalid_size():
    """Test that invalid sizes raise a ValueError."""
    invalid_sizes = [4, 10, 16, 20]
    for size in invalid_sizes:
        with pytest.raises(ValueError, match="Size must be 2\\^n \\+ 1"):
            diamond_square_2d(size)

def test_diamond_square_2d_seed():
    """Test that the same seed produces identical grids."""
    size = 17
    seed = 42

    grid1 = diamond_square_2d(size, seed=seed)
    grid2 = diamond_square_2d(size, seed=seed)

    np.testing.assert_array_equal(grid1, grid2)

    # Different seed should produce different grids
    grid3 = diamond_square_2d(size, seed=43)
    assert not np.array_equal(grid1, grid3)

def test_diamond_square_2d_roughness():
    """Test that changing roughness alters the output grid."""
    size = 33
    seed = 123

    grid_low = diamond_square_2d(size, roughness=0.1, seed=seed)
    grid_high = diamond_square_2d(size, roughness=0.9, seed=seed)

    # Corners are initialized randomly from N(0, 1) independently of roughness.
    # Since seed is the same, corners should be exactly the same.
    np.testing.assert_array_equal(grid_low[0, 0], grid_high[0, 0])
    np.testing.assert_array_equal(grid_low[0, -1], grid_high[0, -1])
    np.testing.assert_array_equal(grid_low[-1, 0], grid_high[-1, 0])
    np.testing.assert_array_equal(grid_low[-1, -1], grid_high[-1, -1])

    # The grids as a whole should be different due to the roughness factor
    assert not np.array_equal(grid_low, grid_high)

def test_diamond_square_2d_deterministic_without_seed():
    """
    If no seed is provided, but global seed is set,
    diamond_square_2d should still be reproducible if called consecutively
    with reset global seed, but let's test just ensuring the function runs.
    """
    size = 5
    grid = diamond_square_2d(size, seed=None)
    assert grid.shape == (size, size, 2)
