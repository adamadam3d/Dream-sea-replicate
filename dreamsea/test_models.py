import torch
from dreamsea.models import ConditionalDDPM, UnconditionalDDPM

def test_conditional_ddpm_forward():
    """Test the forward pass of the ConditionalDDPM model."""
    # Initialize the model with smaller sample size for fast testing
    in_channels = 4
    out_channels = 4
    sample_size = 32
    model = ConditionalDDPM(in_channels=in_channels, out_channels=out_channels, sample_size=sample_size)

    batch_size = 2
    # Create dummy sample: (batch_size, in_channels, height, width)
    sample = torch.randn(batch_size, in_channels, sample_size, sample_size)
    # Timestep: (batch_size,)
    timestep = torch.tensor([10, 10])
    # Encoder hidden states: (batch_size, sequence_length, cross_attention_dim)
    # From models.py: sequence_length=1, cross_attention_dim=2
    encoder_hidden_states = torch.randn(batch_size, 1, 2)

    # Run forward pass
    output = model(sample, timestep, encoder_hidden_states)

    # Assert output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, out_channels, sample_size, sample_size)

def test_unconditional_ddpm_forward():
    """Test the forward pass of the UnconditionalDDPM model."""
    in_channels = 3
    out_channels = 3
    sample_size = 32
    model = UnconditionalDDPM(in_channels=in_channels, out_channels=out_channels, sample_size=sample_size)

    batch_size = 2
    sample = torch.randn(batch_size, in_channels, sample_size, sample_size)
    timestep = torch.tensor([5, 5])

    # Run forward pass
    output = model(sample, timestep)

    # Assert output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, out_channels, sample_size, sample_size)
