import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, UNet2DModel

class ConditionalDDPM(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, sample_size=224):
        super().__init__()
        # The UNet2DConditionModel is used for conditional generation.
        # We need it to accept a conditioning embedding.
        # cross_attention_dim is set to 2 because we use the 2D PCA-reduced DINOv2 features.
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
            cross_attention_dim=2, # Dimension of the PCA conditioning
        )

    def forward(self, sample, timestep, encoder_hidden_states):
        """
        Forward pass for the UNet conditional model.
        encoder_hidden_states should have shape (batch_size, sequence_length, cross_attention_dim).
        In our case, it will be (batch_size, 1, 2) since we use a 2D vector.
        """
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
