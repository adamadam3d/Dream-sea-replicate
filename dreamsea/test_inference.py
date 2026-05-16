import argparse
import torch
import numpy as np
from pathlib import Path
from dreamsea.generation_inpainting import GeneratorInpainter

def main():
    parser = argparse.ArgumentParser(description="Test DDPM inference using trained checkpoints.")
    parser.add_argument("--cond_model_path", type=str, default=None, help="Path to conditional DDPM checkpoint (.pt file).")
    parser.add_argument("--uncond_model_path", type=str, default=None, help="Path to unconditional DDPM checkpoint (.pt file).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    args = parser.parse_args()

    print(f"Initializing GeneratorInpainter on {args.device}...")
    try:
        inpainter = GeneratorInpainter(
            cond_model_path=args.cond_model_path,
            uncond_model_path=args.uncond_model_path,
            device=args.device
        )
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        return

    # Test Conditional Generation
    if args.cond_model_path:
        print(f"\n--- Testing Conditional Generation ---")
        print(f"Using checkpoint: {args.cond_model_path}")
        
        # 2D dummy latent condition (simulating reduced DINOv2 features)
        latent_condition = np.array([0.5, -0.5], dtype=np.float32)
        print(f"Generating patch with latent condition: {latent_condition}")
        
        try:
            patch = inpainter.generate_patch(latent_condition)
            print(f"Success! Generated patch shape: {patch.shape} (Expected: 1, 4, 224, 224)")
            print(f"Output range: min={patch.min():.4f}, max={patch.max():.4f}")
        except Exception as e:
            print(f"Error during conditional generation: {e}")

    # Test Unconditional Inpainting
    if args.uncond_model_path:
        print(f"\n--- Testing Unconditional Inpainting ---")
        print(f"Using checkpoint: {args.uncond_model_path}")
        
        # Dummy image and mask (1, 4, 224, 224)
        dummy_image = np.random.randn(1, 4, 224, 224).astype(np.float32)
        dummy_mask = np.ones((1, 1, 224, 224), dtype=np.float32)
        
        # Create a "hole" to inpaint
        dummy_mask[:, :, 100:150, 100:150] = 0.0 
        
        print("Running RePaint inpainting (this may take a moment)...")
        try:
            # Using 20 inference steps to speed up the test
            inpainted = inpainter.repaint_inpaint(dummy_image, dummy_mask, num_inference_steps=20)
            print(f"Success! Inpainted image shape: {inpainted.shape} (Expected: 1, 4, 224, 224)")
            print(f"Output range: min={inpainted.min():.4f}, max={inpainted.max():.4f}")
        except Exception as e:
            print(f"Error during unconditional inpainting: {e}")

    if not args.cond_model_path and not args.uncond_model_path:
        print("\nNo checkpoints provided! Testing architecture with random weights...")
        latent_condition = np.array([0.0, 0.0], dtype=np.float32)
        try:
            patch = inpainter.generate_patch(latent_condition)
            print(f"Success! Uninitialized conditional model generated patch shape: {patch.shape}")
        except Exception as e:
             print(f"Error during testing: {e}")
             
        print("\nTo test with your trained weights, run:")
        print("python dreamsea/test_inference.py --cond_model_path checkpoints/your_checkpoint.pt")

if __name__ == "__main__":
    main()