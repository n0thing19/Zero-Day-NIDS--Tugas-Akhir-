"""
Stacked Autoencoder — arsitektur identik dengan
02_hybrid_multi_strategy_training.ipynb

Encoder: 39 → 64 → 32 → 16 (ReLU)
Decoder: 16 → 32 → 64 → 39 (ReLU + Sigmoid)
"""
import numpy as np
import torch
import torch.nn as nn


def to_2d_float32(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class StackedAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=39,
        encoder_dims=None,
        latent_dim=16,
        decoder_dims=None,
        output_activation="sigmoid",
    ):
        super().__init__()
        encoder_dims = encoder_dims or [64, 32]
        decoder_dims = decoder_dims or [32, 64]
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev = input_dim
        for d in encoder_dims:
            encoder_layers.append(nn.Linear(prev, d))
            encoder_layers.append(nn.ReLU())
            prev = d
        encoder_layers.append(nn.Linear(prev, latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev = latent_dim
        for d in decoder_dims:
            decoder_layers.append(nn.Linear(prev, d))
            decoder_layers.append(nn.ReLU())
            prev = d
        decoder_layers.append(nn.Linear(prev, input_dim))
        if output_activation == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def get_latent_features(self, x, batch_size=4096):
        self.eval()
        arr = to_2d_float32(x)
        if len(arr) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        dev = next(self.parameters()).device
        parts = []
        for i in range(0, len(arr), batch_size):
            b = torch.from_numpy(arr[i:i+batch_size]).to(dev)
            parts.append(self.encoder(b).detach().cpu().numpy())
        return np.vstack(parts).astype(np.float32)

    @torch.no_grad()
    def get_reconstruction_error(self, x, batch_size=4096):
        self.eval()
        arr = to_2d_float32(x)
        if len(arr) == 0:
            return np.empty(0, dtype=np.float32)
        dev = next(self.parameters()).device
        parts = []
        for i in range(0, len(arr), batch_size):
            b = torch.from_numpy(arr[i:i+batch_size]).to(dev)
            recon = self(b)
            mse = torch.mean((b - recon) ** 2, dim=1)
            parts.append(mse.detach().cpu().numpy())
        return np.concatenate(parts).astype(np.float32)