import os
import torch
import unittest
from tempfile import TemporaryDirectory
from dreamsea.train import train_ddpm

class TestTrain(unittest.TestCase):
    def test_train_dataloader_empty(self):
        with TemporaryDirectory() as temp_dir:
            os.makedirs(os.path.join(temp_dir, "rgbd"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "conditions"), exist_ok=True)

            # create 5 mock files
            for i in range(5):
                torch.save(torch.randn(1, 4, 224, 224), os.path.join(temp_dir, f"rgbd/img{i}_rgbd.pt"))
                torch.save(torch.randn(1, 2), os.path.join(temp_dir, f"conditions/img{i}_cond.pt"))

            # using batch_size 16 > 5 images -> should trigger ValueError
            with self.assertRaises(ValueError) as context:
                train_ddpm(temp_dir, batch_size=16, epochs=1)

            self.assertTrue("smaller than the batch size" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
