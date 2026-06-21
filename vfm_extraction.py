import torch
import torch.nn.functional as F
from transformers import pipeline, AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

class VFMExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        print("Initializing Depth Anything v2...")
        # For demonstration purposes, we are using the v1 pipeline here or a mock
        # since Depth Anything v2 might not be directly available in diffusers out of the box.
        # "depth-anything/Depth-Anything-V2-Small-hf" is standard for v2.
        self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=self.device)

        print("Initializing DINOv2...")
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.dino_model.eval()

    def extract_depth(self, pil_image):
        """
        Extracts depth map from an RGB image and returns an RGBD tensor.

        Args:
            pil_image (PIL.Image): Input RGB image.

        Returns:
            torch.Tensor: RGBD tensor of shape (4, H, W) normalized to [0, 1].
        """
        # Depth prediction
        prediction = self.depth_estimator(pil_image)
        depth_map = prediction["predicted_depth"] # tensor usually

        # In case the pipeline returns a PIL image for depth
        if isinstance(depth_map, Image.Image):
            depth_map = torch.from_numpy(np.array(depth_map)).float().unsqueeze(0)

        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)

        # Normalize depth map to [0, 1]
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

        # Convert PIL image to tensor
        rgb_tensor = torch.from_numpy(np.array(pil_image)).float().permute(2, 0, 1) / 255.0

        # Resize depth map to match RGB dimensions if needed
        if depth_map.shape[1:] != rgb_tensor.shape[1:]:
            depth_map = F.interpolate(depth_map.unsqueeze(0), size=rgb_tensor.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

        # Concatenate RGB and Depth
        rgbd_tensor = torch.cat([rgb_tensor, depth_map], dim=0)
        return rgbd_tensor

    def extract_semantics(self, pil_image, n_components=2):
        """
        Extracts DINOv2 semantics and reduces dimensions using PCA.

        Args:
            pil_image (PIL.Image): Input RGB image.
            n_components (int): Target dimensions for PCA.

        Returns:
            np.ndarray: PCA-reduced semantic embeddings (H', W', 2).
        """
        inputs = self.dino_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # Extract patch tokens
        # outputs.last_hidden_state shape: (batch_size, num_patches, hidden_size)
        # For dinov2-base, hidden_size is 768
        patch_tokens = outputs.last_hidden_state[0, 1:, :] # Skip the CLS token

        # Determine spatial dimensions based on number of patches
        # DINOv2 uses patch size 14
        num_patches = patch_tokens.shape[0]
        h_patches = inputs['pixel_values'].shape[2] // 14
        w_patches = inputs['pixel_values'].shape[3] // 14

        patch_tokens_np = patch_tokens.cpu().numpy()

        # Apply PCA to reduce down to 2 dimensions
        pca = PCA(n_components=n_components)
        reduced_tokens = pca.fit_transform(patch_tokens_np)

        # Reshape to spatial grid
        semantic_map = reduced_tokens.reshape(h_patches, w_patches, n_components)
        return semantic_map

def process_dataset(image_paths):
    extractor = VFMExtractor()
    results = []

    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            print(f"Processing {path}...")

            # 1. Depth Extraction (RGBD)
            rgbd = extractor.extract_depth(img)

            # 2. Semantic Extraction (DINOv2 + PCA)
            semantics = extractor.extract_semantics(img, n_components=2)

            results.append({
                "path": path,
                "rgbd": rgbd,
                "semantics": semantics
            })
            print(f"Successfully processed {path}. RGBD shape: {rgbd.shape}, Semantics shape: {semantics.shape}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    return results

if __name__ == "__main__":
    print("VFM Extraction script ready.")
    # Example usage:
    # process_dataset(["sample1.jpg", "sample2.jpg"])
