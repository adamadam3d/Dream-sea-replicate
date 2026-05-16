import unittest
import torch
import numpy as np
from dreamsea.generation_inpainting import GeneratorInpainter

class TestGeneratorInpainter(unittest.TestCase):
    def setUp(self):
        # Initialize without pre-trained weights for testing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpainter = GeneratorInpainter(device=self.device)

    def test_generate_patch_with_numpy_array(self):
        latent_condition = np.array([0.5, -0.5], dtype=np.float32)
        # Using a small number of inference steps to make the test faster
        patch = self.inpainter.generate_patch(latent_condition, num_inference_steps=2)

        self.assertEqual(patch.shape, (1, 4, 224, 224))
        self.assertTrue(isinstance(patch, np.ndarray))

    def test_generate_patch_with_torch_tensor(self):
        latent_condition = torch.tensor([0.5, -0.5], dtype=torch.float32)
        # Using a small number of inference steps to make the test faster
        patch = self.inpainter.generate_patch(latent_condition, num_inference_steps=2)

        self.assertEqual(patch.shape, (1, 4, 224, 224))
        self.assertTrue(isinstance(patch, np.ndarray))

if __name__ == '__main__':
    unittest.main()
