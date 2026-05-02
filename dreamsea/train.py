import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from .models import ConditionalDDPM, UnconditionalDDPM

class DummyDataset(Dataset):
    """A dummy dataset to allow testing of the training loop."""
    def __init__(self, num_samples=100, conditional=True):
        self.num_samples = num_samples
        self.conditional = conditional

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 4-channel RGBD images of size 224x224
        image = torch.randn(4, 224, 224)
        if self.conditional:
            # 2D PCA reduced feature
            condition = torch.randn(1, 2)
            return image, condition
        return image


def train_ddpm(model_type='conditional', epochs=2000, batch_size=12, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop for DDPM models.
    """
    if model_type == 'conditional':
        model = ConditionalDDPM().to(device)
        dataset = DummyDataset(conditional=True)
    elif model_type == 'unconditional':
        model = UnconditionalDDPM().to(device)
        dataset = DummyDataset(conditional=False)
    else:
        raise ValueError("model_type must be 'conditional' or 'unconditional'")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model.train()
    print(f"Starting training for {model_type} DDPM for {epochs} epochs...")

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            if model_type == 'conditional':
                clean_images, conditions = batch
                clean_images = clean_images.to(device)
                conditions = conditions.to(device)
            else:
                clean_images = batch.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residual
            if model_type == 'conditional':
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=conditions)
            else:
                noise_pred = model(noisy_images, timesteps)

            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    print("Training complete.")
    return model
