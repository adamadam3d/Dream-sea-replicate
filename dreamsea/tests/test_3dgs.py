import unittest
import torch
import numpy as np
from diffusers import DDPMScheduler
import dreamsea.gs_sds_optimization as gs_opt
from dreamsea.models import UnconditionalDDPM

class Test3DGS(unittest.TestCase):
    def test_3dgs_pipeline(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Create a dummy RGBD map
        H, W = 224, 224
        dummy_rgb = np.random.rand(3, H, W).astype(np.float32)
        dummy_depth = np.random.uniform(0.2, 1.0, (1, H, W)).astype(np.float32)
        dummy_rgbd = np.concatenate([dummy_rgb, dummy_depth], axis=0)

        # 2. Extract Point Cloud
        positions, colors = gs_opt.create_point_cloud_from_rgbd(dummy_rgbd)
        self.assertGreater(positions.shape[0], 0, "Failed to extract any points!")

        # 3. Initialize Gaussian Splatting Model
        gs_model = gs_opt.GaussianSplattingModel(positions, colors, device=device)
        self.assertEqual(gs_model.positions.shape[0], positions.shape[0])

        # 4. Run dummy forward pass (Rasterization)
        dummy_camera_pose = None
        rendered_image = gs_model(dummy_camera_pose)
        self.assertEqual(rendered_image.shape, (1, 3, 224, 224))

        # 5. Test SDS Optimization Loop
        uncond_model = UnconditionalDDPM(in_channels=3, out_channels=3, sample_size=224).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)

        # Should run without raising exceptions
        try:
            gs_opt.optimize_3dgs_sds(gs_model, uncond_model, scheduler, iterations=1)
        except Exception as e:
            self.fail(f"optimize_3dgs_sds raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
