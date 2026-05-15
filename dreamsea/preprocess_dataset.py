import os
import glob
import torch
from pathlib import Path
from dreamsea.data_preprocessing import DataPreprocessor

def preprocess_dataset(input_dir: str, output_dir: str, device: str = 'cuda'):
    """
    Preprocesses a directory of RGB images, saving RGBD tensors and DINOv2 condition vectors.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    rgbd_out_dir = output_path / "rgbd"
    cond_out_dir = output_path / "conditions"
    rgbd_out_dir.mkdir(parents=True, exist_ok=True)
    cond_out_dir.mkdir(parents=True, exist_ok=True)

    # Find all images (jpg, png)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(str(input_path / ext)))
        image_paths.extend(glob.glob(str(input_path / ext.upper())))

    # Deduplicate in case of case-insensitive file systems
    image_paths = list(set(image_paths))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Initializing preprocessor...")
    preprocessor = DataPreprocessor(device=device)

    # 1. Extract and reduce DINOv2 features for all images
    # We do this together so PCA is fit on the entire dataset distribution
    print("Extracting and reducing DINOv2 features (this may take a while depending on dataset size)...")
    features_dict = preprocessor.extract_and_reduce_dino_features(image_paths)

    # 2. Process RGB to RGBD and save everything
    print("Processing RGBD and saving tensors...")
    for idx, path_str in enumerate(image_paths):
        base_name = Path(path_str).stem

        # Save condition vector
        cond_tensor = features_dict[path_str]
        cond_save_path = cond_out_dir / f"{base_name}_cond.pt"
        torch.save(cond_tensor.cpu(), cond_save_path)

        # Process and save RGBD tensor
        try:
            rgbd_tensor = preprocessor.process_rgb_to_rgbd(path_str)
            rgbd_save_path = rgbd_out_dir / f"{base_name}_rgbd.pt"
            # Ensure tensor is saved to CPU to save GPU memory during dataloading later
            torch.save(rgbd_tensor.cpu(), rgbd_save_path)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images.")
        except Exception as e:
            print(f"Failed to process {path_str}: {e}")

    print("\nPreprocessing complete!")
    print(f"RGBD tensors saved to: {rgbd_out_dir}")
    print(f"Condition vectors saved to: {cond_out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess dataset for DreamSea training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to raw RGB images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed .pt files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    
    args = parser.parse_args()
    preprocess_dataset(args.input_dir, args.output_dir, args.device)
