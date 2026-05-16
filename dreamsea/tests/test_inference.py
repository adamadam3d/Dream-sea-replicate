import unittest
import torch
import numpy as np
from dreamsea.generation_inpainting import GeneratorInpainter

class TestInference(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.inpainter = GeneratorInpainter(device=self.device)

    def test_random_weights_inference(self):
        latent_condition = np.array([0.0, 0.0], dtype=np.float32)
        try:
            patch = self.inpainter.generate_patch(latent_condition, num_inference_steps=2)
            self.assertEqual(patch.shape, (1, 4, 224, 224))
        except Exception as e:
            self.fail(f"generate_patch raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
