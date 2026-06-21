import pytest
import numpy as np
from dreamsea.fractal_latent import diamond_square_2d

def test_diamond_square_2d_shape():
    """Test that the output shape is correctly sized for size 2^n + 1."""
    size = 5  # 2^2 + 1
    grid = diamond_square_2d(size)
    assert grid.shape == (5, 5, 2)

def test_diamond_square_2d_type():
    """Test that the output type is float32."""
    size = 5
    grid = diamond_square_2d(size)
    assert grid.dtype == np.float32

def test_diamond_square_2d_determinism():
    """Test that setting a seed makes the output deterministic."""
    size = 5
    seed = 42
    grid1 = diamond_square_2d(size, seed=seed)
    grid2 = diamond_square_2d(size, seed=seed)
    np.testing.assert_allclose(grid1, grid2)

def test_diamond_square_2d_value_error():
    """Test that invalid sizes raise a ValueError."""
    # 6 is not 2^n + 1
    with pytest.raises(ValueError, match=r"Size must be 2\^n \+ 1"):
        diamond_square_2d(6)
