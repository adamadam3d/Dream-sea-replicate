import unittest
import torch
import numpy as np
from diffusers import DDPMScheduler

from dreamsea.gs_sds_optimization import (
    GaussianSplattingModel,
    create_point_cloud_from_rgbd,
    compute_sds_loss as compute_sds_loss_v1
)
from dreamsea.gs_sds_optimization_v2 import (
    compute_sds_loss as compute_sds_loss_v2,
    intrinsics,
    look_at,
    sample_topdown_camera,
    render_rgbd
)
from dreamsea.models import ConditionalDDPM, UnconditionalDDPM

class TestSDSComponents(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Create dummy small diffusion models to run tests quickly without checkpoint files
        # We override standard settings with small channels/sizes for speed
        self.uncond_model = UnconditionalDDPM(in_channels=4, out_channels=4, sample_size=64).to(self.device)
        self.cond_model = ConditionalDDPM(in_channels=4, out_channels=4, sample_size=64).to(self.device)

    def test_point_cloud_creation_and_filtering(self):
        """Test unprojecting a dummy RGBD map into a 3D point cloud."""
        H, W = 64, 64
        # Case 1: [0, 1] range input
        dummy_rgb = np.random.rand(3, H, W).astype(np.float32)
        dummy_depth = np.random.uniform(0.2, 0.8, (1, H, W)).astype(np.float32)
        dummy_rgbd = np.concatenate([dummy_rgb, dummy_depth], axis=0)

        positions, colors, conds = create_point_cloud_from_rgbd(dummy_rgbd)
        self.assertEqual(positions.shape[1], 3)
        self.assertEqual(colors.shape[1], 3)
        self.assertIsNone(conds)

        # Case 2: [-1, 1] range input (auto-detected and normalized)
        dummy_rgbd_neg = dummy_rgbd * 2.0 - 1.0
        positions_neg, colors_neg, _ = create_point_cloud_from_rgbd(dummy_rgbd_neg)
        self.assertEqual(positions_neg.shape, positions.shape)
        self.assertEqual(colors_neg.shape, colors.shape)

    def test_gaussian_splatting_model_init(self):
        """Test GaussianSplattingModel initialization and its shapes."""
        positions = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        conds = np.random.rand(100, 2).astype(np.float32)

        model = GaussianSplattingModel(positions, colors, point_cloud_conds=conds, device=self.device)
        self.assertEqual(model.positions.shape, (100, 3))
        self.assertEqual(model.scaling.shape, (100, 3))
        self.assertEqual(model.rotation.shape, (100, 4))
        self.assertEqual(model.opacity.shape, (100, 1))
        self.assertEqual(model.features_dc.shape, (100, 3))
        self.assertEqual(model.point_conds.shape, (100, 2))

    def test_scatter_renderer_forward_and_gradients(self):
        """Test differentiable forward pass in gs_sds_optimization.py."""
        positions = np.random.rand(100, 3).astype(np.float32)
        colors = np.random.rand(100, 3).astype(np.float32)
        model = GaussianSplattingModel(positions, colors, device=self.device)

        # Enable gradients for optimized parameters
        model.opacity.requires_grad = True
        model.features_dc.requires_grad = True

        rendered = model(canvas_size=32)
        self.assertEqual(rendered.shape, (1, 4, 32, 32))

        # Check gradient flow
        loss = rendered.sum()
        loss.backward()

        self.assertIsNotNone(model.opacity.grad)
        self.assertIsNotNone(model.features_dc.grad)
        self.assertFalse(torch.isnan(model.opacity.grad).any())

    def test_sds_loss_v1_computation(self):
        """Test compute_sds_loss from the scatter path."""
        rendered = torch.randn((1, 4, 64, 64), device=self.device, requires_grad=True)
        loss = compute_sds_loss_v1(self.uncond_model, self.scheduler, rendered)

        self.assertEqual(loss.shape, ())
        loss.backward()
        self.assertIsNotNone(rendered.grad)
        self.assertFalse(torch.isnan(rendered.grad).any())

    def test_camera_helpers(self):
        """Test matrix construction and nadir camera sampling."""
        K = intrinsics(60.0, 224, 224, self.device)
        self.assertEqual(K.shape, (3, 3))

        cam_pos = [0, 0, 5]
        target = [0, 0, 0]
        vm = look_at(cam_pos, target, device=self.device)
        self.assertEqual(vm.shape, (4, 4))

        positions = torch.randn(10, 3, device=self.device)
        vm_sampled = sample_topdown_camera(positions, target=None, fov_deg=60.0, device=self.device)
        self.assertEqual(vm_sampled.shape, (4, 4))

    def test_sds_loss_v2_unconditional(self):
        """Test compute_sds_loss_v2 in unconditional mode."""
        rendered = torch.randn((1, 4, 64, 64), device=self.device, requires_grad=True)
        loss = compute_sds_loss_v2(
            self.uncond_model,
            self.scheduler,
            rendered,
            cond=None,
            guidance=0.0
        )
        self.assertEqual(loss.shape, ())
        loss.backward()
        self.assertIsNotNone(rendered.grad)

    def test_sds_loss_v2_conditional_cfg(self):
        """Test compute_sds_loss_v2 in conditional mode with CFG."""
        rendered = torch.randn((1, 4, 64, 64), device=self.device, requires_grad=True)
        cond = torch.randn((1, 1, 2), device=self.device)
        
        # Test CFG path (guidance > 0.0)
        loss = compute_sds_loss_v2(
            self.cond_model,
            self.scheduler,
            rendered,
            cond=cond,
            guidance=2.0
        )
        self.assertEqual(loss.shape, ())
        loss.backward()
        self.assertIsNotNone(rendered.grad)

    def test_conditional_cfg_fallback_bug(self):
        """Verifies that calling conditional model without cond fails when CFG is disabled.

        This test serves to document and guard against the crash described in Consideration Loop 5.
        """
        rendered = torch.randn((1, 4, 64, 64), device=self.device, requires_grad=True)
        
        # With cond=None or guidance=0.0, it should fallback to calling the model directly.
        # This will fail with a TypeError if the model is ConditionalDDPM (missing encoder_hidden_states).
        with self.assertRaises(TypeError):
            compute_sds_loss_v2(
                self.cond_model,
                self.scheduler,
                rendered,
                cond=None,
                guidance=0.0
            )

if __name__ == '__main__':
    unittest.main()
