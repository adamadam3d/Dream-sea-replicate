import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, UNet2DModel

class ConditionalDDPM(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, sample_size=224,
                 cond_dim=2, cond_embed_dim=256):
        super().__init__()
        # The raw condition is a low-dim PCA-of-DINOv2 vector (cond_dim, e.g. 2).
        # Feeding it bare into cross-attention (cross_attention_dim=cond_dim) gives
        # the UNet almost nothing to attend to, so the model collapses to the
        # marginal and effectively ignores the condition (no "type" control). We
        # instead embed it through a small MLP into a richer cross_attention_dim,
        # giving the UNet a learnable nonlinear basis. The LayerNorm makes the
        # embedding robust to the raw PCA magnitude — the saved conditions have
        # per-axis std ~14, far from unit scale, which a bare Linear handles poorly.
        # cond_embed_dim=0 reproduces the legacy bare conditioning (only needed to
        # load pre-embedding checkpoints; new training should keep the MLP on).
        self.cond_dim = cond_dim
        self.cond_embed_dim = cond_embed_dim
        if cond_embed_dim and cond_embed_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, cond_embed_dim),
                nn.LayerNorm(cond_embed_dim),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim),
            )
            cross_attention_dim = cond_embed_dim
        else:
            self.cond_embed = None
            cross_attention_dim = cond_dim

        # The UNet2DConditionModel is used for conditional generation.
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=cross_attention_dim,
        )

    def forward(self, sample, timestep, encoder_hidden_states):
        """
        Forward pass for the UNet conditional model.
        encoder_hidden_states is the raw condition of shape (batch_size,
        sequence_length, cond_dim) — typically (B, 1, 2). It is embedded to
        (B, 1, cond_embed_dim) before cross-attention when the MLP is enabled.
        """
        if self.cond_embed is not None:
            encoder_hidden_states = self.cond_embed(encoder_hidden_states)
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


class UnconditionalDDPM(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, sample_size=224):
        super().__init__()
        # Standard UNet2DModel for unconditional generation.
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, sample, timestep):
        """
        Forward pass for the UNet unconditional model.
        """
        return self.unet(sample, timestep).sample


class SRDDPM(nn.Module):
    """x4 super-resolution DDPM (SR3-style) — stage 2 of the generation cascade.

    Denoises a high-res RGBD patch conditioned on its bilinearly-upsampled
    low-res counterpart, concatenated channel-wise onto the noisy input
    (4 noise + 4 LR = 8 in_channels).

    Architecture follows SR3+ (Sahak et al. 2023): a conv-only UNet with NO
    self-attention — attention hurts generalization across inference
    resolutions/aspect ratios, and this stage always runs far above its
    training-crop size. It is also slimmer than the base generator (4 levels):
    super-resolution is a local texture task whose global structure is pinned
    by the LR conditioning, so scene-scale capacity would only overfit the
    small dataset. Fully convolutional: trained on 224 crops, runs at any
    input size divisible by 8.
    """
    def __init__(self, in_channels=8, out_channels=4, sample_size=224):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, sample, timestep):
        """sample is the channel-wise concat [noisy_hr (4ch), lr_upsampled (4ch)]."""
        return self.unet(sample, timestep).sample
