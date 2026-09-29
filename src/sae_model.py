"""
Stacked Autoencoder (SAE) PyTorch model architecture, training, and feature extraction.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from .data_loader import stream_train_batches_scaled


def to_2d_float32_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32, copy=False)
    else:
        arr = np.array(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class StackedAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dims: Optional[List[int]] = None,
        latent_dim: int = 16,
        decoder_dims: Optional[List[int]] = None,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        encoder_dims = encoder_dims or [64, 32]
        decoder_dims = decoder_dims or [32, 64]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.output_activation = output_activation

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        if output_activation == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
        elif output_activation == "tanh":
            decoder_layers.append(nn.Tanh())
        elif output_activation in {"linear", "identity", "none"}:
            pass
        else:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def get_latent_features(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        arr = to_2d_float32_array(x)
        if arr.shape[0] == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        device = next(self.parameters()).device
        parts = []
        for start in range(0, arr.shape[0], batch_size):
            b = arr[start : start + batch_size]
            b_tensor = torch.from_numpy(b).to(device)
            z = self.encoder(b_tensor)
            parts.append(z.detach().cpu().numpy())
        return np.vstack(parts).astype(np.float32)

    @torch.no_grad()
    def get_reconstruction_error(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        arr = to_2d_float32_array(x)
        if arr.shape[0] == 0:
            return np.empty(0, dtype=np.float32)

        device = next(self.parameters()).device
        parts = []
        for start in range(0, arr.shape[0], batch_size):
            b = arr[start : start + batch_size]
            b_tensor = torch.from_numpy(b).to(device)
            recon = self(b_tensor)
            mse = torch.mean((b_tensor - recon) ** 2, dim=1)
            parts.append(mse.detach().cpu().numpy())
        return np.concatenate(parts).astype(np.float32)

    @torch.no_grad()
    def get_reconstruction_residuals(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        arr = to_2d_float32_array(x)
        if arr.shape[0] == 0:
            return np.empty((0, self.input_dim), dtype=np.float32)

        device = next(self.parameters()).device
        parts = []
        for start in range(0, arr.shape[0], batch_size):
            b = arr[start : start + batch_size]
            b_tensor = torch.from_numpy(b).to(device)
            recon = self(b_tensor)
            residuals = (b_tensor - recon) ** 2
            parts.append(residuals.detach().cpu().numpy())
        return np.vstack(parts).astype(np.float32)


def train_sae(
    model: StackedAutoencoder,
    train_files: List[Path],
    feature_columns: List[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    max_batches_per_epoch: Optional[int] = None,
    log_every: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "batches_per_epoch": [],
        "samples_per_epoch": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        batch_count = 0

        print(f"\n===== SAE Training Epoch {epoch + 1}/{epochs} =====")
        for X_batch_np in stream_train_batches_scaled(
            files=train_files,
            feature_columns=feature_columns,
            mins=mins,
            maxs=maxs,
            batch_size=batch_size,
            shuffle_files=True,
            shuffle_rows=True,
            seed=42 + epoch,
        ):
            X_tensor = torch.from_numpy(X_batch_np).to(device)
            optimizer.zero_grad()
            recon = model(X_tensor)
            loss = criterion(recon, X_tensor)
            loss.backward()
            optimizer.step()

            bs = X_tensor.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            batch_count += 1

            if batch_count % log_every == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_count} | Samples {total_samples} | Loss: {total_loss / total_samples:.6f}")

            if max_batches_per_epoch is not None and batch_count >= max_batches_per_epoch:
                break

        epoch_loss = total_loss / max(total_samples, 1)
        history["train_loss"].append(epoch_loss)
        history["batches_per_epoch"].append(batch_count)
        history["samples_per_epoch"].append(total_samples)
        print(f"Epoch {epoch + 1} Done | Samples: {total_samples} | Loss: {epoch_loss:.6f}")

    return history


def save_sae(
    model: StackedAutoencoder,
    weights_path: Path,
    config_path: Path,
    history: Optional[Dict[str, Any]] = None,
    history_path: Optional[Path] = None,
) -> None:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)

    config = {
        "input_dim": model.input_dim,
        "encoder_dims": model.encoder_dims,
        "latent_dim": model.latent_dim,
        "decoder_dims": model.decoder_dims,
        "output_activation": model.output_activation,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    if history is not None and history_path is not None:
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)


def load_sae(
    weights_path: Path,
    config_path: Path,
    device: Optional[torch.device] = None,
) -> StackedAutoencoder:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = StackedAutoencoder(
        input_dim=cfg["input_dim"],
        encoder_dims=cfg.get("encoder_dims", [64, 32]),
        latent_dim=cfg.get("latent_dim", 16),
        decoder_dims=cfg.get("decoder_dims", [32, 64]),
        output_activation=cfg.get("output_activation", "sigmoid"),
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model
