import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np

from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from transformers import AutoModel

class VFMExtractor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Initializing VFM Extractor on {self.device}")

        self.depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(self.device)

        self.dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)

    def extract_depth_and_concat(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """
        Uses Depth Anything v2 to generate a depth map, normalizes it to [0, 1],
        and concatenates it as a 4th channel to the RGB image.

        Args:
            rgb_image (torch.Tensor): RGB image of shape (3, H, W).

        Returns:
            torch.Tensor: RGBD image of shape (4, H, W).
        """
        _, H, W = rgb_image.shape

        # Preprocess rgb_image
        # Note: transformers ImageProcessor expects numpy arrays or PIL Images usually,
        # but can accept tensors.
        # Assuming rgb_image is in [0, 1] range. Convert to [0, 255] for processor if needed.
        img_input = rgb_image.cpu().numpy()
        inputs = self.depth_processor(images=img_input, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original resolution
        predicted_depth = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )

        # Normalize depth to [0, 1]
        depth_min = predicted_depth.min()
        depth_max = predicted_depth.max()
        depth_normalized = (predicted_depth - depth_min) / (depth_max - depth_min + 1e-8)

        depth_normalized = depth_normalized.squeeze(0) # Shape: (1, H, W)

        # Concatenate RGB and Depth to create RGBD (4 channels)
        rgbd_image = torch.cat([rgb_image.to(self.device), depth_normalized], dim=0)

        return rgbd_image

    def extract_semantic_pca(self, rgb_images: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Uses DINOv2 to extract semantic embeddings for a list of images,
        then applies PCA to reduce the high-dimensional embeddings to 2 dimensions.

        Args:
            rgb_images (list[torch.Tensor]): List of RGB images (3, H, W).

        Returns:
            list[torch.Tensor]: List of 2D embeddings, each of shape (2,).
        """
        embeddings = []

        for img in rgb_images:
            img_input = img.cpu().numpy()
            inputs = self.dino_processor(images=img_input, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                # Use pooler output for image-level representation
                emb = outputs.pooler_output.squeeze().cpu().numpy()
                embeddings.append(emb)

        embeddings_np = np.stack(embeddings) # Shape: (N, 768)

        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_np) # Shape: (N, 2)

        # Convert back to torch tensors
        reduced_tensors = [torch.tensor(emb, dtype=torch.float32) for emb in reduced_embeddings]

        return reduced_tensors

if __name__ == "__main__":
    extractor = VFMExtractor()
    dummy_img1 = torch.rand(3, 256, 256)
    dummy_img2 = torch.rand(3, 256, 256)

    rgbd = extractor.extract_depth_and_concat(dummy_img1)
    print("RGBD shape:", rgbd.shape) # Expected: (4, 256, 256)

    embeddings = extractor.extract_semantic_pca([dummy_img1, dummy_img2])
    print("Reduced embeddings:", [e.shape for e in embeddings]) # Expected: [(2,), (2,)]
