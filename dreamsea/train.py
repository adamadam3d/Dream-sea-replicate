import os
import argparse
import time
import traceback
import urllib.request
import urllib.parse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from dreamsea.models import ConditionalDDPM, UnconditionalDDPM, SRDDPM
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

def generate_sr_samples_during_training(model, dataset, epoch, output_dir, device,
                                         num_samples=4, num_inference_steps=250):
    """Visual check for the SR model: one row per sample, laid out as
    [bilinear x4 input | SR output | ground-truth HR]."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    model.eval()

    def to_pil(tensor):
        t = (tensor.detach().cpu() + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        img_np = (t.numpy() * 255).astype(np.uint8)
        if img_np.ndim == 3:  # RGB
            return Image.fromarray(np.transpose(img_np, (1, 2, 0)), mode='RGB')
        return Image.fromarray(img_np, mode='L')  # Depth

    rows_rgb, rows_depth = [], []
    print(f"  [INFO] Generating {num_samples} SR samples for visual check...")
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            hr, lr_up = dataset[i]
            hr, lr_up = hr.to(device), lr_up.to(device)
            x = torch.randn(1, 4, hr.shape[-2], hr.shape[-1], device=device)
            for t in scheduler.timesteps:
                pred = model(torch.cat([x, lr_up.unsqueeze(0)], dim=1), t)
                x = scheduler.step(pred, t, x).prev_sample
            triplet = [lr_up.cpu(), x[0].cpu(), hr.cpu()]
            rows_rgb.append([to_pil(t_[:3]) for t_ in triplet])
            rows_depth.append([to_pil(t_[3]) for t_ in triplet])

    w, h = rows_rgb[0][0].size
    collage_rgb = Image.new('RGB', (3 * w, len(rows_rgb) * h))
    collage_depth = Image.new('L', (3 * w, len(rows_depth) * h))
    for r, (rgb_row, depth_row) in enumerate(zip(rows_rgb, rows_depth)):
        for c in range(3):
            collage_rgb.paste(rgb_row[c], (c * w, r * h))
            collage_depth.paste(depth_row[c], (c * w, r * h))

    rgb_path = output_dir / f"sr_collage_epoch_{epoch}_rgb.png"
    depth_path = output_dir / f"sr_collage_epoch_{epoch}_depth.png"
    collage_rgb.save(rgb_path)
    collage_depth.save(depth_path)
    return str(rgb_path), str(depth_path)


def build_checkpoint(epoch, unwrapped_model, optimizer, ema_model=None):
    """Assemble a checkpoint dict with raw weights, optimizer state and, when EMA
    is enabled, a named EMA state dict that inference loaders can use directly."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if ema_model is not None:
        # EMAModel only tracks parameters, in parameters() order. Overlay the
        # shadow params on a copy of the full state dict so buffers stay intact
        # and every key keeps its module-qualified name.
        ema_state = {k: v.detach().cpu().clone() for k, v in unwrapped_model.state_dict().items()}
        for (name, _), shadow in zip(unwrapped_model.named_parameters(), ema_model.shadow_params):
            ema_state[name] = shadow.detach().cpu().clone()
        ckpt['ema_model_state_dict'] = ema_state
        ckpt['ema_optimization_step'] = ema_model.optimization_step
    return ckpt

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

class SRPairDataset(Dataset):
    """HR/LR training pairs for the x4 SR stage, cut on the fly from the
    full-resolution RGBD tensors saved by preprocess_sr_dataset.py.

    Each item is a random crop_size crop (with random horizontal flip):
      hr    — the crop, scaled to [-1, 1]
      lr_up — the same crop downsampled x`factor` ('area' mode = antialiased)
              and bilinearly upsampled back, plus conditioning augmentation
              (random gaussian noise) so the SR model stays robust to the base
              generator's imperfect outputs (Ho et al., Cascaded Diffusion
              Models). Every crop position is a distinct training example, so
              a small photo set still yields a large effective dataset.
    """
    def __init__(self, data_dir, crop_size=224, factor=4, aug_noise_max=0.05):
        self.data_dir = Path(data_dir)
        sr_dir = self.data_dir / "sr_rgbd"
        if not sr_dir.exists():
            raise FileNotFoundError(
                f"SR RGBD directory not found at {sr_dir}. "
                f"Run preprocess_sr_dataset.py first.")
        self.files = list(sr_dir.glob("*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {sr_dir}")
        self.crop_size = crop_size
        self.factor = factor
        self.aug_noise_max = aug_noise_max

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = torch.load(self.files[idx], weights_only=True).float()  # (4, H, W) in [0, 1]
        cs = self.crop_size
        _, H, W = img.shape
        if H < cs or W < cs:  # safety net; preprocessing already filters these out
            img = F.interpolate(img.unsqueeze(0), size=(max(H, cs), max(W, cs)),
                                mode='bilinear', align_corners=False).squeeze(0)
            _, H, W = img.shape

        y = torch.randint(0, H - cs + 1, (1,)).item()
        x = torch.randint(0, W - cs + 1, (1,)).item()
        hr = img[:, y:y + cs, x:x + cs]
        if torch.rand(1).item() < 0.5:
            hr = torch.flip(hr, dims=[-1])
        hr = hr * 2.0 - 1.0

        lr = F.interpolate(hr.unsqueeze(0),
                           size=(cs // self.factor, cs // self.factor), mode='area')
        lr_up = F.interpolate(lr, size=(cs, cs),
                              mode='bilinear', align_corners=False).squeeze(0)
        if self.aug_noise_max > 0:
            lr_up = lr_up + torch.randn_like(lr_up) * (torch.rand(1).item() * self.aug_noise_max)
        return hr, lr_up


def train_ddpm(data_dir, model_type='conditional', epochs=500, batch_size=16,
               checkpoint_dir='checkpoints', save_every=50, resume_from=None, 
               device='cuda' if torch.cuda.is_available() else 'cpu', multi_gpu=False,
               learning_rate=1e-4, gradient_accumulation_steps=1, mixed_precision='fp16',
               gradient_checkpointing=False, ntfy_topic=None, ema_decay=0.9999,
               cond_dropout_prob=0.0):
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
    elif model_type == 'sr':
        model = SRDDPM()
        dataset = SRPairDataset(data_dir)
    else:
        raise ValueError("model_type must be 'conditional', 'unconditional' or 'sr'")

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

    # EMA of model weights — sampling/inference uses this averaged copy, which
    # gives markedly cleaner generations than the raw training weights.
    use_ema = ema_decay > 0
    ema_model = None
    if use_ema:
        unwrapped = accelerator.unwrap_model(model)
        resume_ema_state = None
        if resume_from and os.path.exists(resume_from) and isinstance(checkpoint, dict):
            resume_ema_state = checkpoint.get('ema_model_state_dict')
        if resume_ema_state is not None:
            # Seed the EMA shadow params from the checkpoint by temporarily loading
            # the EMA weights into the model (EMAModel clones params on construction).
            raw_state = {k: v.clone() for k, v in unwrapped.state_dict().items()}
            unwrapped.load_state_dict(resume_ema_state)
            ema_model = EMAModel(unwrapped.parameters(), decay=ema_decay, use_ema_warmup=True)
            unwrapped.load_state_dict(raw_state)
            ema_model.optimization_step = checkpoint.get('ema_optimization_step', 0)
            accelerator.print(f"Restored EMA weights from checkpoint (EMA step {ema_model.optimization_step}).")
        else:
            ema_model = EMAModel(unwrapped.parameters(), decay=ema_decay, use_ema_warmup=True)

    model.train()
    accelerator.print(f"Starting training for {model_type} DDPM...")
    accelerator.print(f"Dataset size: {len(dataset)} images")
    accelerator.print(f"Batch size per device: {batch_size}, Epochs: {epochs}")
    accelerator.print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"EMA: {f'enabled (decay={ema_decay}, warmup on)' if use_ema else 'disabled'}")
    if model_type == 'conditional':
        accelerator.print(f"CFG condition dropout: {f'{cond_dropout_prob:.2f}' if cond_dropout_prob > 0 else 'disabled (legacy)'}")
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
        elif model_type == 'sr':
            sample_images, sample_lr = sample_batch
            print(f"\n[DEBUG] Data sanity check:")
            print(f"  HR batch shape: {sample_images.shape}, dtype: {sample_images.dtype}")
            print(f"  HR value range: [{sample_images.min().item():.3f}, {sample_images.max().item():.3f}] (expected ~[-1, 1])")
            print(f"  LR-up batch shape: {sample_lr.shape}, range: [{sample_lr.min().item():.3f}, {sample_lr.max().item():.3f}]")
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
                        # Conditions need reshaping. Clone so the in-place CFG
                        # dropout below never mutates the dataloader's batch tensor.
                        conditions = conditions.reshape(clean_images.shape[0], 1, 2).clone()
                        # Classifier-free guidance dropout: with probability
                        # cond_dropout_prob, replace a sample's condition with the
                        # null token (zeros). The PCA latents are unit-variance and
                        # centered on ~0, so zero is the marginal/"no information"
                        # condition; training the model to predict the dataset mean
                        # there turns the zero latent into a true unconditional
                        # branch, which is what CFG / guided SDS extrapolate from.
                        # Set to 0 to reproduce the legacy no-dropout behavior.
                        if cond_dropout_prob > 0:
                            drop = torch.rand(clean_images.shape[0], device=clean_images.device) < cond_dropout_prob
                            conditions[drop] = 0.0
                    elif model_type == 'sr':
                        clean_images, lr_up = batch
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
                    elif model_type == 'sr':
                        # SR3-style conditioning: the clean LR-upsampled patch rides
                        # along as extra input channels at every timestep.
                        noise_pred = model(torch.cat([noisy_images, lr_up], dim=1), timesteps)
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

                    # Update the EMA copy after each real optimizer step (not on
                    # intermediate gradient-accumulation micro-steps)
                    if use_ema and accelerator.sync_gradients:
                        ema_model.step(accelerator.unwrap_model(model).parameters())

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
                    torch.save(
                        build_checkpoint(epoch + 1, unwrapped_model, optimizer, ema_model),
                        checkpoint_path
                    )
                    print(f"--> Saved checkpoint: {checkpoint_path}")

                    # Generate and send samples — with EMA weights swapped in when
                    # enabled, since those are what inference will load
                    try:
                        sample_dir = os.path.join(checkpoint_dir, "samples_during_training")
                        if use_ema:
                            ema_model.store(unwrapped_model.parameters())
                            ema_model.copy_to(unwrapped_model.parameters())
                        if model_type == 'sr':
                            rgb_collage, depth_collage = generate_sr_samples_during_training(
                                unwrapped_model, dataset, epoch + 1, sample_dir, accelerator.device
                            )
                        else:
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
                    finally:
                        if use_ema:
                            ema_model.restore(unwrapped_model.parameters())
                        # Sampling puts the live model in eval mode; switch back so
                        # training resumes in train mode.
                        unwrapped_model.train()

    except KeyboardInterrupt:
        accelerator.print("\n[INFO] Training interrupted by user. Saving current state...")
        # We use epoch + 1 as the current epoch being worked on
        current_epoch = epoch + 1
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_interrupted_epoch_{current_epoch}.pt")
        
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                build_checkpoint(current_epoch, unwrapped_model, optimizer, ema_model),
                checkpoint_path
            )
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
    parser.add_argument("-i", "--data_dir", type=str, required=True,
                        help="Directory containing preprocessed 'rgbd' and 'conditions' folders "
                             "(for conditional/unconditional), or an 'sr_rgbd' folder from "
                             "preprocess_sr_dataset.py (for the 'sr' model).")
    parser.add_argument("-t", "--model_type", type=str, choices=['conditional', 'unconditional', 'sr'],
                        default='conditional',
                        help="Which model to train. 'sr' = x4 super-resolution cascade stage.")
    parser.add_argument("-e", "--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("-a", "--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("-m", "--mixed_precision", type=str, choices=['no', 'fp16'], default='fp16', help="Whether to use mixed precision (fp16).")
    parser.add_argument("-c", "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("-s", "--save_every", type=int, default=50, help="Save a checkpoint every N epochs.")
    parser.add_argument("-r", "--resume_from", type=str, default=None, help="Path to a checkpoint .pt file to resume training from.")
    parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    parser.add_argument("-p", "--multi_gpu", action="store_true", help="Enable multi-GPU training if multiple GPUs are available.")
    parser.add_argument("-x", "--gradient_checkpointing", action="store_true", help="Enable to drastically reduce VRAM usage at the cost of speed.")
    parser.add_argument("-n", "--ntfy_topic", type=str, default=None, help="ntfy.sh topic for push notifications (e.g. 'dreamsea_adam'). Install ntfy app on phone and subscribe to the same topic.")
    parser.add_argument("-w", "--ema_decay", type=float, default=0.9999, help="Decay for the EMA copy of model weights used for sampling/inference. Set to 0 to disable EMA.")
    parser.add_argument("-g", "--cond_dropout_prob", type=float, default=0.0,
                        help="Classifier-free guidance dropout probability for the conditional model: "
                             "fraction of samples whose condition is replaced with the null (zero) token "
                             "each step. Default 0 (off): we do not use CFG at inference, and dropout makes "
                             "the [0,0] condition a clean mean-attractor while weakly-used real conditions "
                             "degrade — forcing the model to always use the condition gives stronger type "
                             "control. Set >0 only if you intend to use CFG. Ignored for the unconditional model.")

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
            ntfy_topic=args.ntfy_topic,
            ema_decay=args.ema_decay,
            cond_dropout_prob=args.cond_dropout_prob
        )
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"\n[FATAL] {error_msg}")
        send_ntfy(args.ntfy_topic, "\ud83d\udd34 DreamSea CRASHED", error_msg, priority="urgent", tags="rotating_light")
        raise
