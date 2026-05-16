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

    from PIL import Image

    # 1. Process RGB to RGBD and extract raw DINOv2 features simultaneously
    print("Processing RGBD and extracting DINOv2 features...")
    raw_features = []

    for idx, path_str in enumerate(image_paths):
        base_name = Path(path_str).stem

        try:
            # Load the image once
            image = Image.open(path_str).convert("RGB")

            # Extract raw DINOv2 feature
            feature = preprocessor.extract_dino_feature(image)

            # Process and save RGBD tensor
            rgbd_tensor = preprocessor.process_rgb_to_rgbd(image)
            rgbd_save_path = rgbd_out_dir / f"{base_name}_rgbd.pt"

            # Ensure tensor is saved to CPU to save GPU memory during dataloading later
            torch.save(rgbd_tensor.cpu(), rgbd_save_path)
            
            # Append only if both successful
            raw_features.append(feature)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(image_paths)} images.")
        except Exception as e:
            print(f"Failed to process {path_str}: {e}")
            # Do not append anything if failure, to keep aligned we will just append zero later
            # Wait, no, we need to keep indexing aligned. But a better way is to use a dictionary or handle correctly.
            # However, the original code collected DINO features for ALL paths, then iterated ALL paths again.
            # If a path fails in original code, DINO feature is extracted but RGBD fails, and DINO feature is just skipped / unused for training data since RGBD is missing, BUT the condition tensor is still saved.
            # We can mimic the original code's fail handling by safely saving the feature if extracted.
            # In the original code, `features_dict = preprocessor.extract_and_reduce_dino_features(image_paths)` was called on ALL image_paths FIRST, which would crash if DINO extraction failed.
            # We will use a dummy feature matching the extractor dimension dynamically, if needed.
            # Actually, instead of appending dummy features, let's determine the hidden size dynamically or just use zeros of the correct size.

            # Since dinov2-small cls token dim is 384, we could hardcode it, but better is to get it from preprocessor
            dim = preprocessor.dino_model.config.hidden_size if hasattr(preprocessor, "dino_model") else 384
            raw_features.append(torch.zeros(dim).numpy())

    # 2. Reduce DINOv2 features and save condition vectors
    # We do this together so PCA is fit on the entire dataset distribution
    print("Reducing DINOv2 features and saving condition vectors...")
    if raw_features:
        reduced_features_tensor = preprocessor.reduce_dino_features(raw_features)

        for idx, path_str in enumerate(image_paths):
            base_name = Path(path_str).stem
            cond_tensor = reduced_features_tensor[idx]
            cond_save_path = cond_out_dir / f"{base_name}_cond.pt"
            torch.save(cond_tensor.cpu(), cond_save_path)

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
