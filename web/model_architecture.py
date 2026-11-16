from typing import List
import torch
import torch.nn as nn


class StackedAutoencoder(nn.Module):
    """Arsitektur simetris Stacked Autoencoder (39 -> 64 -> 32 -> 16 -> 32 -> 64 -> 39)."""

    def __init__(
        self,
        input_dim: int = 39,
        encoder_dims: List[int] = None,
        latent_dim: int = 16,
        decoder_dims: List[int] = None,
    ):
        super().__init__()
        encoder_dims = encoder_dims or [64, 32]
        decoder_dims = decoder_dims or [32, 64]

        # Encoder
        enc_layers = []
        prev = input_dim
        for d in encoder_dims:
            enc_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        enc_layers.extend([nn.Linear(prev, latent_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for d in decoder_dims:
            dec_layers.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        dec_layers.extend([nn.Linear(prev, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))