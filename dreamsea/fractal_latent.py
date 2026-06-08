import numpy as np
from functools import lru_cache


def scale_latent_grid(grid, latent_mean, latent_std):
    """
    Maps a fractal latent grid (unit population variance, mean ~0) into the
    training PCA distribution with a FIXED affine transform: out = grid*std + mean.

    diamond_square_2d now divides its output by the field's analytic std, so the
    "values ~N(0,1)" assumption this map relies on holds exactly: the generated
    conditions reproduce the training per-axis std instead of being slightly
    under-spread (interior cells are averages of corners, so their raw variance
    is < 1).

    This is deliberately independent of the particular grid's own spread. The
    previous implementation normalized by the grid's own min/max, which stretched
    EVERY grid to fill the full training range and thereby erased the fractal
    scale factor s that controls terrain diversity (s=0 -> calm/flat, s>0 ->
    varied; see paper Figs. 11-12). A fixed mean/std map preserves it: a
    low-variance (calm) field stays tightly clustered in PCA space, a
    high-variance field spreads out. It is also well-defined for a 1x1 grid
    (which min/max normalization collapsed to a single corner value).

    Pass latent_mean / latent_std from the latent_stats.json saved by
    preprocess_dataset.py.
    """
    latent_mean = np.array(latent_mean, dtype=np.float32)
    latent_std = np.array(latent_std, dtype=np.float32)
    return (grid * latent_std + latent_mean).astype(np.float32)


@lru_cache(maxsize=None)
def diamond_square_analytic_std(size, roughness):
    """
    Exact population std of the (single-channel) diamond_square_2d field for a
    given (size, roughness) — computed analytically, with no RNG.

    Every cell is a fixed linear combination of independent unit-variance draws:
    one "own" draw per cell, plus the averaging of already-assigned neighbours.
    We replay the EXACT traversal of diamond_square_2d on coefficient vectors
    instead of values — row c of C holds cell c's coefficients over the size*size
    independent draws. Per-cell variance is then that row's sum of squares (draws
    are independent, unit-variance), and the field's population variance (what
    scale_latent_grid needs to equal 1) is the mean of the per-cell variances.

    Because the result depends only on (size, roughness), dividing every
    generated field by it removes the systematic under-spread WITHOUT touching
    the draw-to-draw diversity the fixed affine map in scale_latent_grid relies
    on (a calm draw stays below this std, a varied draw above it).

    NB: the loop structure here must stay in lock-step with diamond_square_2d.
    """
    if size == 1:
        return 1.0

    n = size * size

    def idx(y, x):
        return y * size + x

    # C[cell, source]; each cell contributes exactly one independent draw, and no
    # cell is ever assigned twice, so size*size sources cover the whole field.
    C = np.zeros((n, n), dtype=np.float64)

    # Corners: own draw at amplitude 1.0 (matches grid[corner] = randn()).
    for (y, x) in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]:
        C[idx(y, x), idx(y, x)] = 1.0

    step = size - 1
    scale = 1.0
    while step > 1:
        half_step = step // 2

        # Diamond step
        for y in range(0, size - 1, step):
            for x in range(0, size - 1, step):
                avg = (C[idx(y, x)] +
                       C[idx(y, x + step)] +
                       C[idx(y + step, x)] +
                       C[idx(y + step, x + step)]) / 4.0
                avg[idx(y + half_step, x + half_step)] += scale * roughness
                C[idx(y + half_step, x + half_step)] = avg

        # Square step
        for y in range(0, size, half_step):
            x_start = half_step if (y // half_step) % 2 == 0 else 0
            for x in range(x_start, size, step):
                sum_rows = np.zeros(n, dtype=np.float64)
                count = 0
                if x >= half_step:
                    sum_rows += C[idx(y, x - half_step)]; count += 1
                if x + half_step < size:
                    sum_rows += C[idx(y, x + half_step)]; count += 1
                if y >= half_step:
                    sum_rows += C[idx(y - half_step, x)]; count += 1
                if y + half_step < size:
                    sum_rows += C[idx(y + half_step, x)]; count += 1
                row = sum_rows / count
                row[idx(y, x)] += scale * roughness
                C[idx(y, x)] = row

        scale *= (2 ** (-roughness))
        step = half_step

    per_cell_var = (C ** 2).sum(axis=1)
    return float(np.sqrt(per_cell_var.mean()))


def diamond_square_2d(size, roughness=0.5, seed=None):
    """
    Generates a 2D grid of latent embeddings using the Diamond-Square algorithm.
    Size must be 2^n + 1 (or 1 for a single patch). Returns a grid of shape (size, size, 2) since we want
    to generate 2D PCA latent vectors.

    The field is divided by its analytic std (see diamond_square_analytic_std) so
    it has unit population variance — only the corners are exactly N(0,1) before
    this, while interior cells (averages of corners) are under-spread. Normalizing
    by a constant that depends only on (size, roughness) keeps draw-to-draw
    diversity intact and makes the downstream scale_latent_grid map land
    generated conditions on the true training per-axis std.
    """
    if seed is not None:
        np.random.seed(seed)

    if size == 1:
        # Handle the base case of a 1x1 grid (single patch)
        grid = np.zeros((1, 1, 2), dtype=np.float32)
        grid[0, 0] = np.random.randn(2)
        return grid

    if size < 3:
        raise ValueError("Size must be at least 3 (2^1 + 1), or 1")
    if (size - 1) & (size - 2) != 0:
        raise ValueError("Size must be 2^n + 1 (e.g., 3, 5, 9, 17...) or 1")

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

    # Normalize to unit population variance so scale_latent_grid maps the field
    # onto the true training per-axis std (see diamond_square_analytic_std).
    grid /= diamond_square_analytic_std(size, roughness)

    return grid
