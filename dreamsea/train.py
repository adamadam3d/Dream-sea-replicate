import os
import argparse
import time
import traceback
import urllib.request
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from dreamsea.models import ConditionalDDPM, UnconditionalDDPM
from dreamsea.generation_inpainting import GeneratorInpainter

# --- Push notification helper (ntfy.sh) ---
def send_ntfy(topic, title, message, priority="default", tags="", image_path=None):
    """Send a push notification via ntfy.sh. Fails silently if unavailable."""
    if not topic:
        return
    try:
        # Convert priority string to ntfy integer levels
        prio_map = {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}
        prio_int = prio_map.get(priority, 3)
        
        if image_path and os.path.exists(image_path):
            import urllib.parse
            # ntfy allows sending a file by PUTing the raw bytes
            with open(image_path, 'rb') as f:
                data = f.read()
            
            # Use query parameters to avoid urllib header encoding crashes with emojis
            params = {
                "title": title,
                "message": message,
                "priority": str(prio_int),
                "filename": os.path.basename(image_path)
            }
            if tags:
                params["tags"] = tags
                
            query_string = urllib.parse.urlencode(params)
            url = f"https://ntfy.sh/{topic}?{query_string}"
            
            req = urllib.request.Request(
                url,
                data=data,
                method='PUT'
            )
        else:
            # Standard JSON payload for text-only
            payload = {
                "topic": topic,
                "message": message,
                "title": title,
                "priority": prio_int
            }
            if tags:
                payload["tags"] = [tags]

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                "https://ntfy.sh/",
                data=data,
                headers={"Content-Type": "application/json"}
            )
        
        urllib.request.urlopen(req, timeout=30)
    except Exception as e:
        print(f"Failed to send notification: {e}")
        pass  # Never let a notification failure crash training

def generate_samples_during_training(model, epoch, output_dir, device, model_type='conditional', num_samples=10):
    """Generates a collage of samples using current weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize a temporary inpainter with current model weights
    # We don't want to load from disk, we want to use the live 'model'
    inpainter = GeneratorInpainter(device=device)
    
    # Overwrite the inpainter's model with the live training model
    if model_type == 'conditional':
        inpainter.cond_model = model
        inpainter.cond_model.eval()
    else:
        inpainter.uncond_model = model
        inpainter.uncond_model.eval()

    all_rgb = []
    all_depth = []

    print(f"  [INFO] Generating {num_samples} samples for visual check...")

    # Helper to normalize tensor for PIL
    def to_pil(tensor):
        t = (tensor.detach().cpu() + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        img_np = (t.numpy() * 255).astype(np.uint8)
        if img_np.ndim == 3: # RGB
            img_np = np.transpose(img_np, (1, 2, 0))
            return Image.fromarray(img_np, mode='RGB')
        else: # Depth
            return Image.fromarray(img_np, mode='L')

    with torch.no_grad():
        for i in range(num_samples):
            if model_type == 'conditional':
                # Constant [0.0, 0.0] latent to probe the model at the PCA-space origin
                latent = np.array([0.0, 0.0], dtype=np.float32)
                patch = inpainter.generate_patch(latent, num_inference_steps=250)
                patch_tensor = torch.from_numpy(patch[0])
            else:
                # Run denoising directly with the trained unconditional model
                image = torch.randn(1, 4, 224, 224, device=device)
                inpainter.scheduler.set_timesteps(num_inference_steps=250)
                for t in inpainter.scheduler.timesteps:
                    noise_pred = inpainter.uncond_model(image, t)
                    image = inpainter.scheduler.step(noise_pred, t, image).prev_sample
                patch_tensor = image[0].cpu()
            
            all_rgb.append(to_pil(patch_tensor[:3]))
            all_depth.append(to_pil(patch_tensor[3]))

    # Create Collage
    cols = 5
    rows = (num_samples + cols - 1) // cols
    w, h = all_rgb[0].size
    
    collage_rgb = Image.new('RGB', (cols * w, rows * h))
    collage_depth = Image.new('L', (cols * w, rows * h))
    
    for idx, (rgb, depth) in enumerate(zip(all_rgb, all_depth)):
        r, c = idx // cols, idx % cols
        collage_rgb.paste(rgb, (c * w, r * h))
        collage_depth.paste(depth, (c * w, r * h))
        
    rgb_path = output_dir / f"collage_epoch_{epoch}_rgb.png"
    depth_path = output_dir / f"collage_epoch_{epoch}_depth.png"
    
    collage_rgb.save(rgb_path)
    collage_depth.save(depth_path)
    
    return str(rgb_path), str(depth_path)

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
        image = torch.load(rgbd_path, weights_only=True)
        
        # Fallback for old preprocessed data that wasn't squeezed
        while image.dim() > 3 and image.shape[0] == 1:
            image = image.squeeze(0)
            
        # Fallback resize for old data
        if image.shape[-2:] != (224, 224):
            image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            
        # Scale from [0, 1] to [-1, 1] (Crucial for Diffusion stability)
        image = image * 2.0 - 1.0
        
        if self.conditional:
            # Load corresponding condition vector
            base_name = rgbd_path.name.replace("_rgbd.pt", "")
            cond_path = self.data_dir / "conditions" / f"{base_name}_cond.pt"
            condition = torch.load(cond_path, weights_only=True)
            # Squeeze to ensure it's a 1D tensor of size 2 instead of (1, 2)
            condition = condition.squeeze()
            return image, condition
            
        return image

def train_ddpm(data_dir, model_type='conditional', epochs=500, batch_size=16, 
               checkpoint_dir='checkpoints', save_every=50, resume_from=None, 
               device='cuda' if torch.cuda.is_available() else 'cpu', multi_gpu=False,
               learning_rate=1e-4, gradient_accumulation_steps=1, mixed_precision='fp16',
               gradient_checkpointing=False, ntfy_topic=None):
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

    if gradient_checkpointing:
        accelerator.print("Enabling Gradient Checkpointing to save VRAM...")
        model.unet.enable_gradient_checkpointing()

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        accelerator.print(f"Loading checkpoint from: {resume_from}")
        # When using accelerate, we load weights onto CPU first, then let accelerate place them
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        
        # Support both old-style (raw state_dict) and new-style (dict with keys) checkpoints
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            start_epoch = checkpoint.get('epoch', 0)
        else:
            model_state = checkpoint
        
        # Clean up any 'module.' prefixes from old DataParallel saves
        clean_state_dict = {k.replace('module.', ''): v for k, v in model_state.items()}
        model.load_state_dict(clean_state_dict)
    elif resume_from:
        accelerator.print(f"Warning: Checkpoint not found at {resume_from}. Starting from scratch.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Prepare everything with accelerator
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Restore optimizer state AFTER accelerator.prepare() so device placement is correct
    if resume_from and os.path.exists(resume_from) and isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            accelerator.print("Restored optimizer state from checkpoint.")
        except Exception as e:
            accelerator.print(f"Warning: Could not restore optimizer state: {e}. Using fresh optimizer.")

    model.train()
    accelerator.print(f"Starting training for {model_type} DDPM...")
    accelerator.print(f"Dataset size: {len(dataset)} images")
    accelerator.print(f"Batch size per device: {batch_size}, Epochs: {epochs}")
    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Checkpoints will be saved to '{checkpoint_dir}' every {save_every} epochs.")

    if accelerator.is_main_process:
        send_ntfy(ntfy_topic, f"🚀 {model_type} STARTED",
                  f"Dataset: {len(dataset)} imgs | Batch: {batch_size} | Epochs: {epochs}",
                  tags="rocket")

    # --- Data sanity check: verify first batch looks correct ---
    if accelerator.is_main_process:
        sample_batch = next(iter(dataloader))
        if model_type == 'conditional':
            sample_images, sample_conds = sample_batch
            print(f"\n[DEBUG] Data sanity check:")
            print(f"  Image batch shape: {sample_images.shape}, dtype: {sample_images.dtype}")
            print(f"  Image value range: [{sample_images.min().item():.3f}, {sample_images.max().item():.3f}] (expected ~[-1, 1])")
            print(f"  Image mean: {sample_images.mean().item():.4f}, std: {sample_images.std().item():.4f}")
            print(f"  Condition shape: {sample_conds.shape}, range: [{sample_conds.min().item():.3f}, {sample_conds.max().item():.3f}]")
            print(f"  Contains NaN: {torch.isnan(sample_images).any().item()}, Contains Inf: {torch.isinf(sample_images).any().item()}")
        else:
            sample_images = sample_batch
            print(f"\n[DEBUG] Data sanity check:")
            print(f"  Image batch shape: {sample_images.shape}, dtype: {sample_images.dtype}")
            print(f"  Image value range: [{sample_images.min().item():.3f}, {sample_images.max().item():.3f}] (expected ~[-1, 1])")
            print(f"  Image mean: {sample_images.mean().item():.4f}, std: {sample_images.std().item():.4f}")
            print(f"  Contains NaN: {torch.isnan(sample_images).any().item()}, Contains Inf: {torch.isinf(sample_images).any().item()}")
        
        # GPU info
        if torch.cuda.is_available():
            print(f"\n[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[DEBUG] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"[DEBUG] Learning rate: {learning_rate}")
        print("")

    # Track consecutive NaN batches — if too many in a row, training is diverging
    max_consecutive_nan = 10
    consecutive_nan_count = 0

    # Track loss history for trend monitoring
    prev_avg_loss = None

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()

            # Clear cache to prevent fragmentation on small VRAM GPUs
            if torch.cuda.is_available() and accelerator.is_main_process:
                torch.cuda.empty_cache()
                
            epoch_loss = 0.0
            epoch_loss_min = float('inf')
            epoch_loss_max = float('-inf')
            valid_steps = 0
            max_grad_norm_epoch = 0.0
            skipped_steps_epoch = 0
            
            for step, batch in enumerate(dataloader):
                with accelerator.accumulate(model):
                    if model_type == 'conditional':
                        clean_images, conditions = batch
                        # Conditions need reshaping
                        conditions = conditions.view(clean_images.shape[0], 1, 2)
                    else:
                        clean_images = batch

                    # Sanity check: skip batches with NaN/Inf in input data
                    if torch.isnan(clean_images).any() or torch.isinf(clean_images).any():
                        accelerator.print(f"  [WARN] Epoch {epoch+1}, step {step}: NaN/Inf in input data, skipping batch.")
                        continue

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
                    
                    # NaN loss detection: skip this batch to prevent poisoning the optimizer state
                    if torch.isnan(loss) or torch.isinf(loss):
                        consecutive_nan_count += 1
                        accelerator.print(
                            f"  [WARN] Epoch {epoch+1}, step {step}: NaN/Inf loss detected "
                            f"({consecutive_nan_count}/{max_consecutive_nan} consecutive). Skipping batch."
                        )
                        optimizer.zero_grad()
                        if consecutive_nan_count >= max_consecutive_nan:
                            msg = f"{max_consecutive_nan} consecutive NaN batches at epoch {epoch+1}. Training is diverging — aborting."
                            accelerator.print(f"  [ERROR] {msg}")
                            send_ntfy(ntfy_topic, f"🔴 {model_type} CRASHED", msg, priority="urgent", tags="rotating_light")
                            return model
                        continue
                    else:
                        consecutive_nan_count = 0
                    
                    # Backward pass handled by accelerate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm = grad_norm.item()
                        if math.isfinite(grad_norm):
                            max_grad_norm_epoch = max(max_grad_norm_epoch, grad_norm)
                        else:
                            skipped_steps_epoch += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Accumulate loss stats
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    epoch_loss_min = min(epoch_loss_min, loss_val)
                    epoch_loss_max = max(epoch_loss_max, loss_val)
                    valid_steps += 1

            # Log diagnostics (only on main process to prevent duplicate prints)
            if accelerator.is_main_process:
                avg_loss = epoch_loss / max(valid_steps, 1)
                epoch_time = time.time() - epoch_start_time
                
                # Main log line
                grad_norm_str = f"{max_grad_norm_epoch:.2f}"
                if skipped_steps_epoch > 0:
                    grad_norm_str += f" ({skipped_steps_epoch} steps skipped due to overflow)"
                
                print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f} | "
                      f"Min/Max: {epoch_loss_min:.4f}/{epoch_loss_max:.4f} | "
                      f"Grad Norm: {grad_norm_str} | "
                      f"Steps: {valid_steps}/{len(dataloader)} | "
                      f"Time: {epoch_time:.1f}s")
                
                # Loss trend warning
                if prev_avg_loss is not None:
                    loss_change = (avg_loss - prev_avg_loss) / max(prev_avg_loss, 1e-8)
                    if loss_change > 0.5:  # Loss jumped by more than 50%
                        print(f"  [WARN] Loss spiked by {loss_change*100:.0f}% from previous epoch!")
                    if avg_loss > 1.0:
                        print(f"  [WARN] Loss is unusually high ({avg_loss:.4f}). Model may be unstable.")
                prev_avg_loss = avg_loss
                
                # GPU memory (every 50 epochs to avoid clutter)
                if torch.cuda.is_available() and (epoch + 1) % 50 == 0:
                    print(f"  [DEBUG] GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                          f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB, "
                          f"Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

                # Checkpointing — save optimizer state alongside model for safe resumption
                if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                    checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_epoch_{epoch+1}.pt")
                    
                    # Unwrap model before saving to ensure clean state dict
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"--> Saved checkpoint: {checkpoint_path}")

                    # Generate and send samples
                    try:
                        sample_dir = os.path.join(checkpoint_dir, "samples_during_training")
                        rgb_collage, depth_collage = generate_samples_during_training(
                            unwrapped_model, epoch + 1, sample_dir, accelerator.device, model_type
                        )
                        
                        send_ntfy(ntfy_topic, f"🖼️ {model_type} Samples (RGB)", 
                                 f"Epoch {epoch+1} visual check", tags="art", image_path=rgb_collage)
                        send_ntfy(ntfy_topic, f"📐 {model_type} Samples (Depth)", 
                                 f"Epoch {epoch+1} depth check", tags="triangular_ruler", image_path=depth_collage)
                    except Exception as e:
                        print(f"  [WARN] Failed to generate/send samples: {e}")
                        send_ntfy(ntfy_topic, f"💾 {model_type} checkpoint",
                                  f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f}",
                                  tags="floppy_disk")

    except KeyboardInterrupt:
        accelerator.print("\n[INFO] Training interrupted by user. Saving current state...")
        # We use epoch + 1 as the current epoch being worked on
        current_epoch = epoch + 1
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_interrupted_epoch_{current_epoch}.pt")
        
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"--> Saved interrupted checkpoint: {checkpoint_path}")
            print(f"To resume training, use: --resume_from {checkpoint_path}")
            send_ntfy(ntfy_topic, f"⏸️ {model_type} INTERRUPTED",
                      f"Training stopped at epoch {current_epoch}. Progress saved.",
                      tags="pause_button")
        
        return model

    accelerator.print("\nTraining complete.")
    send_ntfy(ntfy_topic, f"✅ {model_type} DONE",
              f"Training finished! {epochs} epochs, final loss: {avg_loss:.4f}",
              priority="high", tags="white_check_mark")
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
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable to drastically reduce VRAM usage at the cost of speed.")
    parser.add_argument("--ntfy_topic", type=str, default=None, help="ntfy.sh topic for push notifications (e.g. 'dreamsea_adam'). Install ntfy app on phone and subscribe to the same topic.")

    args = parser.parse_args()

    try:
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
            multi_gpu=args.multi_gpu,
            gradient_checkpointing=args.gradient_checkpointing,
            ntfy_topic=args.ntfy_topic
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n[FATAL] {error_msg}")
        send_ntfy(args.ntfy_topic, "\ud83d\udd34 DreamSea CRASHED", error_msg, priority="urgent", tags="rotating_light")
        raise
