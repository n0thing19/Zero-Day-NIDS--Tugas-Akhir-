"""
Pipeline inferensi — mereplikasi persis alur di notebook:
1. MinMax scaling (mins/maxs dari minmax_params.json)
2. SAE → Z (latent) + MSE (reconstruction error)
3. IF(Z) score (Isolation Forest pada latent Z saja)
4. Normalisasi dengan bounds dari config
5. CDF rank terhadap calibration refs
6. Max-Rank fusion: H = max(CDF_SAE, CDF_IF)
7. Threshold τ (P97 default = 0.9832)
"""
import json
import numpy as np
import torch
import joblib
from pathlib import Path
from model_def import StackedAutoencoder


class NIDSPipeline:
    def __init__(self, artifacts_dir: str = "artifacts", device: str = "auto"):
        self.artifacts_dir = Path(artifacts_dir)

        # --- Device ---
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # --- Load config ---
        with open(self.artifacts_dir / "hybrid_pipeline_config.json", "r") as f:
            self.config = json.load(f)

        # --- Load MinMax params ---
        with open(self.artifacts_dir / "minmax_params.json", "r") as f:
            scaler = json.load(f)
        self.feature_columns = scaler["feature_columns"]
        self.mins = np.array(scaler["min"], dtype=np.float32)
        self.maxs = np.array(scaler["max"], dtype=np.float32)

        # --- Load SAE ---
        self.sae = StackedAutoencoder(
            input_dim=self.config["feature_count"],
            encoder_dims=[64, 32],
            latent_dim=self.config["latent_dim"],
            decoder_dims=[32, 64],
            output_activation="sigmoid",
        ).to(self.device)
        self.sae.load_state_dict(
            torch.load(
                self.artifacts_dir / "sae_autoencoder_final.pth",
                map_location=self.device,
            )
        )
        self.sae.eval()

        # --- Load IForest (Eksperimen A: hanya pada Z) ---
        self.iforest = joblib.load(
            self.artifacts_dir / "iforest_latent_final.joblib"
        )

        # --- Load CDF calibration refs ---
        self.sae_calib_ref = np.load(
            self.artifacts_dir / "sae_calib_norm_ref.npy"
        )
        self.ifz_calib_ref = np.load(
            self.artifacts_dir / "ifz_calib_norm_ref.npy"
        )

        # --- Bounds ---
        self.sae_bounds = tuple(
            self.config["normalization_bounds"]["sae_bounds"]
        )
        self.if_bounds = tuple(
            self.config["normalization_bounds"]["if_bounds_A"]
        )

        # --- Threshold default (P97) ---
        self.tau = self.config["thresholds"]["percentile_97_optimal"]["tau_val"]

    # ---------- helpers ----------
    def minmax_transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        denom = self.maxs - self.mins
        denom = np.where(np.isclose(denom, 0.0), 1.0, denom).astype(np.float32)
        return np.clip((X - self.mins) / denom, 0.0, 1.0).astype(np.float32)

    def normalize_with_bounds(self, scores, bounds):
        s = np.asarray(scores, dtype=np.float64)
        lo, hi = bounds
        denom = hi - lo
        if np.isclose(denom, 0.0):
            return np.zeros_like(s, dtype=np.float32)
        return np.clip((s - lo) / denom, 0.0, 1.0).astype(np.float32)

    def cdf_against(self, scores, ref):
        ref_sorted = np.sort(np.asarray(ref, dtype=np.float64))
        s = np.asarray(scores, dtype=np.float64)
        return np.searchsorted(ref_sorted, s, side="right") / len(ref_sorted)

    # ---------- core inference ----------
    def predict(self, X_raw):
        """
        X_raw: np.ndarray shape (n, 39) — fitur mentah NetFlow
        Returns: dict dengan skor & verdict per sampel
        """
        X_raw = np.asarray(X_raw, dtype=np.float32)
        if X_raw.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Jumlah fitur harus {len(self.feature_columns)}, "
                f"didapat {X_raw.shape[1]}"
            )

        # 1. Scaling
        X_scaled = self.minmax_transform(X_raw)

        # 2. SAE
        Z = self.sae.get_latent_features(X_scaled)
        mse = self.sae.get_reconstruction_error(X_scaled)

        # 3. IF(Z) — hanya latent, TIDAK pakai MSE
        if_raw = -self.iforest.score_samples(Z)

        # 4. Normalisasi dengan bounds
        mse_norm = self.normalize_with_bounds(mse, self.sae_bounds)
        if_norm = self.normalize_with_bounds(if_raw, self.if_bounds)

        # 5. CDF rank
        c_sae = self.cdf_against(mse_norm, self.sae_calib_ref)
        c_if = self.cdf_against(if_norm, self.ifz_calib_ref)

        # 6. Max-Rank fusion
        H = np.maximum(c_sae, c_if)

        # 7. Verdict
        verdict = (H > self.tau).astype(int)

        return {
            "c_sae": c_sae,
            "c_if": c_if,
            "H": H,
            "verdict": verdict,
            "Z": Z,
            "mse": mse,
            "if_raw": if_raw,
        }

    def set_threshold(self, percentile: int):
        """Ganti threshold ke persentil lain (95 atau 97)."""
        key = f"percentile_{percentile}_optimal" if percentile == 97 else f"percentile_{percentile}"
        if key in self.config["thresholds"]:
            self.tau = self.config["thresholds"][key]["tau_val"]
            return self.tau
        raise KeyError(f"Persentil {percentile} tidak ada di config")