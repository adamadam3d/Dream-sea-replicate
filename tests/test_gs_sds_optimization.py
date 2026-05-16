import numpy as np
import pytest
from dreamsea.gs_sds_optimization import create_point_cloud_from_rgbd

def test_create_point_cloud_from_rgbd_basic():
    """Verify output shapes and types with valid input data."""
    # Create dummy 4-channel RGBD map (RGB + Depth)
    # Shape: (4, H, W)
    H, W = 100, 100
    # RGB values between 0 and 1
    rgb = np.random.rand(3, H, W).astype(np.float32)
    # Depth values all > 0 so that no filtering occurs
    depth = np.ones((1, H, W), dtype=np.float32) * 0.5

    rgbd_map = np.concatenate([rgb, depth], axis=0)

    positions, colors = create_point_cloud_from_rgbd(rgbd_map)

    # Expected number of points is H * W since all depth > 0 (after scaling, z > 0.1)
    # Depth scaling is `z = depth * 10.0 + 1.0` -> `0.5 * 10.0 + 1.0 = 6.0` > 0.1
    expected_points = H * W

    assert positions.shape == (expected_points, 3)
    assert colors.shape == (expected_points, 3)
    assert positions.dtype == np.float32 or positions.dtype == np.float64
    assert colors.dtype == np.float32

def test_create_point_cloud_from_rgbd_filtering():
    """Verify that invalid depth values are correctly filtered out."""
    H, W = 10, 10
    rgb = np.random.rand(3, H, W).astype(np.float32)

    # Set depth so that resulting `z` will be <= 0.1
    # z = depth * 10.0 + 1.0
    # For z <= 0.1, depth * 10.0 <= -0.9, depth <= -0.09
    depth = np.ones((1, H, W), dtype=np.float32)

    # Half the depth values are valid, half are invalid
    depth[0, :, :5] = 0.5  # valid: z = 6.0 > 0.1
    depth[0, :, 5:] = -0.5 # invalid: z = -4.0 <= 0.1

    rgbd_map = np.concatenate([rgb, depth], axis=0)

    positions, colors = create_point_cloud_from_rgbd(rgbd_map)

    # Expected number of points is half of H * W
    expected_points = (H * W) // 2

    assert positions.shape[0] == expected_points
    assert colors.shape[0] == expected_points

def test_create_point_cloud_from_rgbd_math():
    """Verify that 2D pixels are correctly unprojected into 3D space."""
    H, W = 100, 100
    rgb = np.zeros((3, H, W), dtype=np.float32)
    depth = np.ones((1, H, W), dtype=np.float32) * 0.5

    rgbd_map = np.concatenate([rgb, depth], axis=0)

    # Default FOV is 60.0
    fov = 60.0
    focal = 0.5 * W / np.tan(0.5 * np.radians(fov))
    cx, cy = W / 2.0, H / 2.0

    positions, _ = create_point_cloud_from_rgbd(rgbd_map, fov=fov)

    # Find the central pixel (or near central if even)
    # The meshgrid is 0-indexed, so we can pick pixel at x=50, y=50
    # In flatten order, pixel (y=50, x=50) is index 50*100 + 50 = 5050
    # Wait, the unprojection code flattens with reshape(-1, 3).
    # Meshgrid creates x of shape (H,W) and y of shape (H,W)
    # Flattening by reshape(-1) goes row by row (C-order).
    idx = 50 * W + 50

    # Expected calculation for pixel (50, 50)
    x, y = 50.0, 50.0
    z = 0.5 * 10.0 + 1.0 # 6.0

    expected_x = (x - cx) * z / focal
    expected_y = (y - cy) * z / focal
    expected_z = z

    np.testing.assert_allclose(positions[idx], [expected_x, expected_y, expected_z], rtol=1e-5)
