import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
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
               checkpoint_dir='checkpoints', save_every=50, resume_from=None, 
               device='cuda' if torch.cuda.is_available() else 'cpu', multi_gpu=False,
               learning_rate=1e-4, gradient_accumulation_steps=1, mixed_precision='fp16'):
    """
    Training loop for DDPM models using preprocessed data.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Accelerator
    # If multi_gpu is True, we let accelerate handle the distributed backend.
    # Otherwise, we restrict it to a single process.
    accelerator = Accelerator(
        mixed_precision=mixed_precision if mixed_precision != 'no' else 'no',
        gradient_accumulation_steps=gradient_accumulation_steps,
        cpu=device == 'cpu'
    )
    
    if multi_gpu:
        print(f"DEBUG: Using Accelerate. Multi-GPU environment depends on accelerate launch config.")
    else:
        print("DEBUG: Using Accelerate for single GPU/CPU.")

    if model_type == 'conditional':
        model = ConditionalDDPM()
        dataset = PreprocessedDataset(data_dir, conditional=True)
    elif model_type == 'unconditional':
        model = UnconditionalDDPM()
        dataset = PreprocessedDataset(data_dir, conditional=False)
    else:
        raise ValueError("model_type must be 'conditional' or 'unconditional'")

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        accelerator.print(f"Loading checkpoint from: {resume_from}")
        # When using accelerate, we load weights onto CPU first, then let accelerate place them
        state_dict = torch.load(resume_from, map_location='cpu', weights_only=True)
        
        # Clean up any 'module.' prefixes from old DataParallel saves
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)
    elif resume_from:
        accelerator.print(f"Warning: Checkpoint not found at {resume_from}. Starting from scratch.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    accelerator.print(f"Starting training for {model_type} DDPM...")
    accelerator.print(f"Dataset size: {len(dataset)} images")
    accelerator.print(f"Batch size per device: {batch_size}, Epochs: {epochs}")
    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Checkpoints will be saved to '{checkpoint_dir}' every {save_every} epochs.\n")

    for epoch in range(epochs):
        # Clear cache to prevent fragmentation on small VRAM GPUs
        if torch.cuda.is_available() and accelerator.is_main_process:
            torch.cuda.empty_cache()
            
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                if model_type == 'conditional':
                    clean_images, conditions = batch
                    # Conditions need reshaping
                    conditions = conditions.view(clean_images.shape[0], 1, 2)
                else:
                    clean_images = batch

                # Sample noise
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # Add noise
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # Predict the noise residual
                if model_type == 'conditional':
                    noise_pred = model(noisy_images, timesteps, encoder_hidden_states=conditions)
                else:
                    noise_pred = model(noisy_images, timesteps)
                
                # Compute loss
                loss = F.mse_loss(noise_pred, noise)
                
                # Backward pass handled by accelerate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                # Accumulate average loss (only logging on main process later)
                epoch_loss += loss.item()

        # Log average loss (only on main process to prevent duplicate prints)
        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

            # Checkpointing
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_epoch_{epoch+1}.pt")
                
                # Unwrap model before saving to ensure clean state dict
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), checkpoint_path)
                print(f"--> Saved checkpoint: {checkpoint_path}")

    accelerator.print("\nTraining complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DreamSea DDPM models.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing preprocessed 'rgbd' and 'conditions' folders.")
    parser.add_argument("--model_type", type=str, choices=['conditional', 'unconditional'], default='conditional', help="Which model to train.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--mixed_precision", type=str, choices=['no', 'fp16'], default='fp16', help="Whether to use mixed precision (fp16).")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--save_every", type=int, default=50, help="Save a checkpoint every N epochs.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint .pt file to resume training from.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU training if multiple GPUs are available.")

    args = parser.parse_args()

    train_ddpm(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        device=args.device,
        multi_gpu=args.multi_gpu
    )