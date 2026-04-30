import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalUNet(nn.Module):
    """
    A simplified UNet architecture for a Conditional DDPM.
    It accepts a 4-channel input (RGBD) and is conditioned on a 2D embedding.
    """
    def __init__(self, in_channels=4, cond_dim=2, base_channels=64):
        super().__init__()

        # We project the 2D condition (PCA-reduced DINOv2) to match base channels
        self.cond_proj = nn.Linear(cond_dim, base_channels)

        # Time step embedding projection
        self.time_proj = nn.Linear(1, base_channels)

        # Simplified Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        # Simplified Decoder
        self.dec1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv2d(base_channels * 2, in_channels, kernel_size=3, padding=1) # *2 because of skip connection

    def forward(self, x, t, cond):
        """
        Args:
            x (torch.Tensor): Noisy RGBD image of shape (B, 4, H, W)
            t (torch.Tensor): Time step of shape (B, 1)
            cond (torch.Tensor): 2D semantic condition of shape (B, 2)

        Returns:
            torch.Tensor: Predicted noise of shape (B, 4, H, W)
        """
        # Embed time and condition
        t_emb = self.time_proj(t).unsqueeze(-1).unsqueeze(-1) # (B, base_channels, 1, 1)
        c_emb = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1) # (B, base_channels, 1, 1)

        # Combine embeddings
        emb = t_emb + c_emb

        # Encoder
        e1 = F.relu(self.enc1(x) + emb) # Add embeddings
        e2 = F.relu(self.enc2(e1))

        # Decoder
        d1 = F.relu(self.dec1(e2))

        # Skip connection: concatenate d1 and e1 along channel dimension
        out = self.dec2(torch.cat([d1, e1], dim=1))

        return out

class CustomDDPMTrainer:
    def __init__(self, num_timesteps=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        # Define a linear variance schedule
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.model = ConditionalUNet(in_channels=4, cond_dim=2).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: adds noise to the data.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)

        # \sqrt{\bar{\alpha}_t} * x_0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise
        return x_t, noise

    def train_step(self, rgbd_batch, cond_batch):
        """
        Executes one training step.

        Args:
            rgbd_batch (torch.Tensor): Batch of RGBD images (B, 4, H, W).
            cond_batch (torch.Tensor): Batch of 2D embeddings (B, 2).
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch_size = rgbd_batch.shape[0]

        # Randomly sample time steps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        # Add noise to the batch
        noise = torch.randn_like(rgbd_batch)
        x_t, target_noise = self.q_sample(rgbd_batch, t, noise)

        # Prepare time tensor for model (B, 1)
        t_float = t.float().unsqueeze(-1) / self.num_timesteps

        # Predict noise
        predicted_noise = self.model(x_t, t_float, cond_batch)

        # Calculate MSE Loss
        loss = F.mse_loss(predicted_noise, target_noise)

        # Backprop
        loss.backward()
        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":
    trainer = CustomDDPMTrainer(device="cpu")

    # Dummy batch
    batch_size = 4
    dummy_rgbd = torch.randn(batch_size, 4, 64, 64)
    dummy_cond = torch.randn(batch_size, 2)

    loss = trainer.train_step(dummy_rgbd, dummy_cond)
    print(f"Training step completed with Loss: {loss:.4f}")
