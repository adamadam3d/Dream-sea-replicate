import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

class UnderwaterDataset(Dataset):
    def __init__(self, rgbd_images, semantic_embeddings):
        """
        Args:
            rgbd_images (list of torch.Tensor): List of RGBD tensors (4, H, W).
            semantic_embeddings (list of np.ndarray): List of 2D semantics (H', W', 2).
        """
        self.rgbd_images = rgbd_images
        self.semantic_embeddings = semantic_embeddings

    def __len__(self):
        return len(self.rgbd_images)

    def __getitem__(self, idx):
        rgbd = self.rgbd_images[idx]

        # Flatten and pad semantic embeddings to serve as cross-attention context
        # Convert (H', W', 2) -> (H'*W', 2) -> Pad to match cross_attention_dim
        sem_map = torch.tensor(self.semantic_embeddings[idx]).float()
        H_prime, W_prime, _ = sem_map.shape
        seq_len = H_prime * W_prime

        # Cross attention context expected shape: (sequence_length, cross_attention_dim)
        # We'll map the 2D semantics to a higher dimension for the UNet using a linear layer later,
        # but for now we just return the raw flattened semantics.
        sem_flat = sem_map.view(seq_len, 2)

        return {"images": rgbd, "semantics": sem_flat}

def train_conditional_ddpm(dataset, epochs=100, batch_size=4, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Cross attention dim (we'll project our 2D semantics to this dim)
    cross_attention_dim = 256

    # 1. Initialize UNet with 4 input channels (RGBD)
    model = UNet2DConditionModel(
        sample_size=64, # Adjust based on image resolution
        in_channels=4,  # RGBD
        out_channels=4, # Predicting noise for RGBD
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=cross_attention_dim,
    ).to(device)

    # 2. Linear projection for 2D semantics to cross_attention_dim
    # We add this small network to project DINOv2 2D PCA embeddings to UNet's context dimension
    semantic_projector = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, cross_attention_dim)
    ).to(device)

    # 3. Setup Noise Scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 4. Optimizer and Learning Rate Scheduler
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(semantic_projector.parameters()), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataloader) * epochs),
    )

    model.train()
    semantic_projector.train()

    print("Starting Training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            clean_images = batch["images"].to(device)
            semantics = batch["semantics"].to(device) # (B, SeqLen, 2)

            # Project semantics to target cross-attention dim
            encoder_hidden_states = semantic_projector(semantics) # (B, SeqLen, cross_attention_dim)

            # Sample noise to add to the images
            noise = torch.randn_like(clean_images)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            # Calculate loss (MSE)
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training Complete.")
    return model, semantic_projector

if __name__ == "__main__":
    print("Conditional DDPM Training script initialized.")
    # Example usage:
    # dummy_rgbd = [torch.rand(4, 64, 64) for _ in range(10)]
    # dummy_semantics = [np.random.rand(16, 16, 2) for _ in range(10)]
    # dataset = UnderwaterDataset(dummy_rgbd, dummy_semantics)
    # trained_model, trained_proj = train_conditional_ddpm(dataset, epochs=2)
