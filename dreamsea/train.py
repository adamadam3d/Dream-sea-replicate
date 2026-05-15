import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from dreamsea.models import ConditionalDDPM, UnconditionalDDPM

class PreprocessedDataset(Dataset):
    """Loads preprocessed RGBD tensors and condition vectors from disk."""
    def __init__(self, data_dir, conditional=True):
        self.data_dir = Path(data_dir)
        self.conditional = conditional
        
        # Get list of all rgbd files
        rgbd_dir = self.data_dir / "rgbd"
        if not rgbd_dir.exists():
            raise FileNotFoundError(f"RGBD directory not found at {rgbd_dir}")
            
        self.rgbd_files = list(rgbd_dir.glob("*.pt"))
        if len(self.rgbd_files) == 0:
            raise ValueError(f"No .pt files found in {rgbd_dir}")

    def __len__(self):
        return len(self.rgbd_files)

    def __getitem__(self, idx):
        rgbd_path = self.rgbd_files[idx]
        image = torch.load(rgbd_path)
        
        # The preprocessor saved tensors as [1, 4, H, W]
        # We need to squeeze out all leading 1s to get [4, H, W]
        while image.dim() > 3 and image.shape[0] == 1:
            image = image.squeeze(0)
            
        # The UNet is initialized with sample_size=224, so we need to resize
        # the inputs to 224x224, otherwise they will be too large and cause dimension errors or OOM.
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        if self.conditional:
            # Load corresponding condition vector
            base_name = rgbd_path.name.replace("_rgbd.pt", "")
            cond_path = self.data_dir / "conditions" / f"{base_name}_cond.pt"
            condition = torch.load(cond_path)
            # Squeeze to ensure it's a 1D tensor of size 2 instead of (1, 2)
            condition = condition.squeeze()
            return image, condition
            
        return image

def train_ddpm(data_dir, model_type='conditional', epochs=500, batch_size=16, 
               checkpoint_dir='checkpoints', save_every=50, resume_from=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop for DDPM models using preprocessed data.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    if model_type == 'conditional':
        model = ConditionalDDPM().to(device)
        dataset = PreprocessedDataset(data_dir, conditional=True)
    elif model_type == 'unconditional':
        model = UnconditionalDDPM().to(device)
        dataset = PreprocessedDataset(data_dir, conditional=False)
    else:
        raise ValueError("model_type must be 'conditional' or 'unconditional'")

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from: {resume_from}")
        state_dict = torch.load(resume_from, map_location=device)
        model.load_state_dict(state_dict)
    elif resume_from:
        print(f"Warning: Checkpoint not found at {resume_from}. Starting from scratch.")

    # Multi-GPU support
    if torch.cuda.device_count() > 1 and "cuda" in str(device):
        print(f"--- Using {torch.cuda.device_count()} GPUs for training! ---")
        model = torch.nn.DataParallel(model)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    model.train()
    print(f"Starting training for {model_type} DDPM on {device}...")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batch size: {batch_size} (Total across all GPUs), Epochs: {epochs}")
    print(f"Checkpoints will be saved to '{checkpoint_dir}' every {save_every} epochs.\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            if model_type == 'conditional':
                clean_images, conditions = batch
                clean_images = clean_images.to(device)
                
                # The model expects conditions of shape (batch, seq_len, embed_dim)
                # Ensure conditions are (batch, 1, 2)
                conditions = conditions.view(clean_images.shape[0], 1, 2).to(device)
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
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # Checkpointing
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_epoch_{epoch+1}.pt")
            
            # Handle DataParallel state_dict saving
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

    print("\nTraining complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DreamSea DDPM models.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed 'rgbd' and 'conditions' folders.")
    parser.add_argument("--model_type", type=str, choices=['conditional', 'unconditional'], default='conditional', help="Which model to train.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--save_every", type=int, default=50, help="Save a checkpoint every N epochs.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint .pt file to resume training from.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")

    args = parser.parse_args()

    train_ddpm(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        device=args.device
    )
