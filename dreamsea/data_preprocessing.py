import torch
from transformers import pipeline, AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load Depth Anything v2 (we use the small model for this implementation)
        self.depth_estimator = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

        # Load DINOv2
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-small').to(self.device)
        self.pca = PCA(n_components=2)

    def process_rgb_to_rgbd(self, image_input):
        """
        Loads an RGB image (or takes a PIL Image), computes relative depth [0, 1],
        and returns a 4-channel RGBD tensor.
        """
        # Load image
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        image_np = np.array(image).astype(np.float32) / 255.0

        # Estimate depth
        depth_output = self.depth_estimator(image)
        depth_map = depth_output['depth']
        depth_np = np.array(depth_map).astype(np.float32)

        # Normalize depth map to [0, 1]
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max > depth_min:
            depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_np)

        # Resize depth map to match image shape if necessary
        # pipeline output should already match input PIL image, but just in case
        if depth_normalized.shape != image_np.shape[:2]:
            depth_pil = Image.fromarray(depth_normalized)
            depth_pil = depth_pil.resize((image_np.shape[1], image_np.shape[0]), Image.BILINEAR)
            depth_normalized = np.array(depth_pil)

        # Expand depth dimension to concatenate
        depth_normalized = np.expand_dims(depth_normalized, axis=-1)

        # Concatenate to form 4-channel RGBD
        rgbd_np = np.concatenate([image_np, depth_normalized], axis=-1)

        # Convert to tensor (C, H, W)
        rgbd_tensor = torch.from_numpy(rgbd_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return rgbd_tensor

    def extract_dino_feature(self, image_input):
        """
        Extracts DINOv2 features for a single image (path or PIL Image).
        Returns the feature vector as a numpy array.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        # Use the CLS token representation as the semantic feature vector
        cls_token = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_token

    def reduce_dino_features(self, features_list):
        """
        Applies PCA to reduce a list of feature vectors to 2 dimensions.
        Returns the reduced features as a tensor.
        """
        features_np = np.array(features_list)

        # Fit PCA and transform
        # If there's only 1 sample, PCA with n_components=2 might fail or give zeros.
        # We handle this by setting to zero or random if n_samples < 2
        if features_np.shape[0] < 2:
             reduced_features = np.zeros((features_np.shape[0], 2))
        else:
            reduced_features = self.pca.fit_transform(features_np)

        # Convert back to tensor
        reduced_features_tensor = torch.from_numpy(reduced_features).float().to(self.device)
        return reduced_features_tensor

    def extract_and_reduce_dino_features(self, image_paths):
        """
        Extracts DINOv2 features for a list of images and applies PCA to reduce them to 2 dimensions.
        Returns a dictionary mapping image_path to its 2D feature vector.
        """
        features = [self.extract_dino_feature(path) for path in image_paths]
        reduced_features_tensor = self.reduce_dino_features(features)
        return {path: reduced_features_tensor[i] for i, path in enumerate(image_paths)}
