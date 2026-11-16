import json
import time
import joblib
import numpy as np
import torch

from config import ARTIFACTS_DIR, DEFAULT_TAU
from model_architecture import StackedAutoencoder


class ModelEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.load_all_artifacts()

    def load_all_artifacts(self):
        config_path = ARTIFACTS_DIR / "hybrid_pipeline_config.json"
        scaler_path = ARTIFACTS_DIR / "minmax_params.json"

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler = json.load(f)

        self.feature_columns = self.config["feature_columns"]
        self.mins = np.array(scaler["min"], dtype=np.float32)
        self.maxs = np.array(scaler["max"], dtype=np.float32)
        self.sae_bounds = tuple(self.config["normalization_bounds"]["sae_bounds"])
        self.if_bounds = tuple(self.config["normalization_bounds"]["if_bounds_A"])

        # Model PyTorch SAE
        sae_path = ARTIFACTS_DIR / "sae_autoencoder_final.pth"
        self.sae = StackedAutoencoder(input_dim=len(self.feature_columns), latent_dim=16).to(self.device)
        self.sae.load_state_dict(torch.load(sae_path, map_location=self.device))
        self.sae.eval()

        # Model Isolation Forest
        if_path = ARTIFACTS_DIR / "iforest_latent_final.joblib"
        self.iforest = joblib.load(if_path)

        # Referensi Kalibrasi ECDF
        self.sae_ref = np.sort(np.load(ARTIFACTS_DIR / "sae_calib_norm_ref.npy").astype(np.float64))
        self.ifz_ref = np.sort(np.load(ARTIFACTS_DIR / "ifz_calib_norm_ref.npy").astype(np.float64))

        self.is_loaded = True
        print(f"[OK] Engine AI siap. Device: {self.device} | Features: {len(self.feature_columns)}")

    def scale_minmax(self, X: np.ndarray) -> np.ndarray:
        denom = np.where(np.isclose(self.maxs - self.mins, 0.0), 1.0, self.maxs - self.mins)
        return np.clip((X - self.mins) / denom, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def normalize_bounds(val: float, bounds: tuple) -> float:
        denom = bounds[1] - bounds[0]
        if denom <= 0:
            return 0.0
        return float(np.clip((val - bounds[0]) / denom, 0.0, 1.0))

    @staticmethod
    def ecdf_rank(val: float, sorted_ref: np.ndarray) -> float:
        idx = np.searchsorted(sorted_ref, val, side="right")
        return float(idx / len(sorted_ref))

    @torch.no_grad()
    def predict_single_flow(self, raw_features: np.ndarray, tau: float = DEFAULT_TAU) -> dict:
        t_start = time.perf_counter()

        x_scaled = self.scale_minmax(raw_features.reshape(1, -1))
        tensor_x = torch.from_numpy(x_scaled).to(self.device)

        # Level 1: SAE Reconstruction & XAI per-fitur dalam 1 kali jalan
        latent_z = self.sae.encoder(tensor_x)
        recon_x = self.sae.decoder(latent_z)

        feat_errors = (tensor_x - recon_x) ** 2
        sae_raw = float(torch.mean(feat_errors, dim=1).cpu().item())

        # Ekstrak Top-3 Fitur paling anomali (MSE terbesar)
        feat_err_np = feat_errors.cpu().numpy()[0]
        top_indices = np.argsort(feat_err_np)[::-1][:3]
        top_features = [
            {"feature": self.feature_columns[idx], "error": round(float(feat_err_np[idx]), 6)}
            for idx in top_indices
        ]

        # Level 2: Isolation Forest pada Laten Z (16-D)
        z_np = latent_z.cpu().numpy().astype(np.float32)
        if_raw = float(-self.iforest.score_samples(z_np)[0])

        # Normalisasi & ECDF
        sae_norm = self.normalize_bounds(sae_raw, self.sae_bounds)
        if_norm = self.normalize_bounds(if_raw, self.if_bounds)
        sae_cdf = self.ecdf_rank(sae_norm, self.sae_ref)
        if_cdf = self.ecdf_rank(if_norm, self.ifz_ref)

        # Max-Rank Fusion
        max_rank = max(sae_cdf, if_cdf)
        is_anomaly = max_rank > tau

        latency_ms = (time.perf_counter() - t_start) * 1000

        return {
            "saeRaw": round(sae_raw, 6),
            "ifRaw": round(if_raw, 4),
            "saeCdf": round(sae_cdf, 4),
            "ifCdf": round(if_cdf, 4),
            "maxRank": round(max_rank, 4),
            "verdict": "ANOMALY" if is_anomaly else "NORMAL",
            "latencyMs": round(latency_ms, 2),
            "topFeatures": top_features,
        }


# Singleton instance yang siap diimpor oleh routes
ENGINE = ModelEngine()