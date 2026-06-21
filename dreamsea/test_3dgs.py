import argparse
import torch
import numpy as np
from diffusers import DDPMScheduler
import dreamsea.gs_sds_optimization as gs_opt
from dreamsea.models import UnconditionalDDPM

def test_3dgs(device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Testing 3DGS and SDS optimization on {device}...")

    # 1. Create a dummy RGBD map
    # Shape: (4, H, W). Using 224x224 for speed.
    print("\n--- 1. Generating Dummy RGBD Map ---")
    H, W = 224, 224
    dummy_rgb = np.random.rand(3, H, W).astype(np.float32)
    # Ensure depth > 0.1 so the filter extracts valid points
    dummy_depth = np.random.uniform(0.2, 1.0, (1, H, W)).astype(np.float32) 
    dummy_rgbd = np.concatenate([dummy_rgb, dummy_depth], axis=0)
    print(f"Dummy RGBD map shape: {dummy_rgbd.shape}")

    # 2. Extract Point Cloud
    print("\n--- 2. Extracting Point Cloud from RGBD ---")
    positions, colors, _ = gs_opt.create_point_cloud_from_rgbd(dummy_rgbd)
    print(f"Extracted {positions.shape[0]} valid points.")
    
    if positions.shape[0] == 0:
        print("Failed to extract any points! Check the depth values.")
        return

    # 3. Initialize Gaussian Splatting Model
    print("\n--- 3. Initializing GaussianSplattingModel ---")
    try:
        gs_model = gs_opt.GaussianSplattingModel(positions, colors, device=device)
        print("Successfully initialized GaussianSplattingModel.")
        print(f"Number of Gaussians (N): {gs_model.positions.shape[0]}")
    except Exception as e:
        print(f"Error initializing 3DGS model: {e}")
        return

    # 4. Run dummy forward pass (Rasterization)
    print("\n--- 4. Testing 3DGS Forward Pass (Dummy Rasterization) ---")
    try:
        dummy_camera_pose = None
        rendered_image = gs_model(dummy_camera_pose)
        print(f"Rendered image shape: {rendered_image.shape} (Expected: 1, 3, 224, 224)")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return

    # 5. Test SDS Optimization Loop
    print("\n--- 5. Testing SDS Optimization Loop ---")
    try:
        # The optimization needs a diffusion model to act as the "critic" (SDS guidance)
        print("Initializing dummy Unconditional DDPM for SDS guidance (3 channels)...")
        # In main.py, it uses in_channels=3, out_channels=3 to evaluate rendered RGB images
        uncond_model = UnconditionalDDPM(in_channels=3, out_channels=3, sample_size=224).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        print("Running 5 iterations of optimize_3dgs_sds...")
        gs_opt.optimize_3dgs_sds(gs_model, uncond_model, scheduler, iterations=5)
        print("Successfully completed SDS optimization loop!")
    except Exception as e:
        print(f"Error during SDS optimization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 3D Gaussian Splatting and SDS.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    args = parser.parse_args()
    
    test_3dgs(args.device)
