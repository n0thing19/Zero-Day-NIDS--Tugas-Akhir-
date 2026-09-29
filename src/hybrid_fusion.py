"""
Hybrid Fusion module: Normalization, threshold calibration, and decision fusion strategies.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np

from .sae_model import StackedAutoencoder
from .iforest_model import CascadedIForest


def normalize_with_bounds(
    scores: np.ndarray, bounds: Tuple[float, float]
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    min_score, max_score = bounds

    if np.isclose(max_score, min_score):
        return np.zeros_like(scores, dtype=np.float32)

    norm = (scores - min_score) / (max_score - min_score)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


class HybridDetector:
    def __init__(
        self,
        sae_model: StackedAutoencoder,
        iforest_model: CascadedIForest,
        sae_bounds: Optional[Tuple[float, float]] = None,
        iforest_bounds: Optional[Tuple[float, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        fusion_strategy: str = "or_rule",
        alpha: float = 0.5,
    ):
        self.sae_model = sae_model
        self.iforest_model = iforest_model
        self.sae_bounds = sae_bounds
        self.iforest_bounds = iforest_bounds
        self.thresholds = thresholds or {}
        self.fusion_strategy = fusion_strategy
        self.alpha = alpha

    def fit_bounds(
        self, sae_train_raw: np.ndarray, iforest_train_raw: np.ndarray
    ) -> None:
        self.sae_bounds = (
            float(np.min(sae_train_raw)),
            float(np.max(sae_train_raw)),
        )
        self.iforest_bounds = (
            float(np.min(iforest_train_raw)),
            float(np.max(iforest_train_raw)),
        )

    def calibrate_thresholds(
        self,
        sae_calib_raw: np.ndarray,
        iforest_calib_raw: np.ndarray,
        percentile: float = 95.0,
    ) -> Dict[str, float]:
        if self.sae_bounds is None or self.iforest_bounds is None:
            raise ValueError("Bounds must be fitted before threshold calibration.")

        sae_calib_norm = normalize_with_bounds(sae_calib_raw, self.sae_bounds)
        if_calib_norm = normalize_with_bounds(iforest_calib_raw, self.iforest_bounds)

        tau_sae = float(np.percentile(sae_calib_norm, percentile))
        tau_if = float(np.percentile(if_calib_norm, percentile))

        safe_tau_sae = tau_sae if tau_sae > 1e-6 else 1e-6
        safe_tau_if = tau_if if tau_if > 1e-6 else 1e-6
        max_calib = np.maximum(
            sae_calib_norm / safe_tau_sae,
            if_calib_norm / safe_tau_if,
        )
        tau_max = float(np.percentile(max_calib, percentile))

        w_calib = self.alpha * sae_calib_norm + (1.0 - self.alpha) * if_calib_norm
        tau_w = float(np.percentile(w_calib, percentile))

        self.thresholds = {
            "tau_sae": tau_sae,
            "tau_iforest": tau_if,
            "tau_max_normalized": tau_max,
            "tau_weighted_average": tau_w,
            "percentile": percentile,
        }
        return self.thresholds

    def compute_scores(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.sae_bounds is None or self.iforest_bounds is None:
            raise ValueError("Bounds must be fitted before computing scores.")

        sae_raw = self.sae_model.get_reconstruction_error(X)
        if_raw = self.iforest_model.score_samples(X, self.sae_model)

        sae_norm = normalize_with_bounds(sae_raw, self.sae_bounds)
        if_norm = normalize_with_bounds(if_raw, self.iforest_bounds)

        tau_sae = self.thresholds.get("tau_sae", 0.5)
        tau_if = self.thresholds.get("tau_iforest", 0.5)
        safe_tau_sae = tau_sae if tau_sae > 1e-6 else 1e-6
        safe_tau_if = tau_if if tau_if > 1e-6 else 1e-6

        max_norm = np.maximum(
            sae_norm / safe_tau_sae,
            if_norm / safe_tau_if,
        )
        weighted = self.alpha * sae_norm + (1.0 - self.alpha) * if_norm

        return {
            "sae_raw": sae_raw,
            "iforest_raw": if_raw,
            "sae_norm": sae_norm,
            "iforest_norm": if_norm,
            "hybrid_max_norm": max_norm,
            "hybrid_weighted": weighted,
        }

    def predict(
        self,
        X: np.ndarray,
        strategy: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        strat = strategy or self.fusion_strategy
        scores = self.compute_scores(X)

        tau_sae = self.thresholds["tau_sae"]
        tau_if = self.thresholds["tau_iforest"]

        pred_sae = (scores["sae_norm"] > tau_sae).astype(int)
        pred_if = (scores["iforest_norm"] > tau_if).astype(int)

        if strat == "or_rule":
            y_pred = (pred_sae | pred_if).astype(int)
        elif strat == "max_normalized":
            tau_max = self.thresholds.get("tau_max_normalized", 1.0)
            y_pred = (scores["hybrid_max_norm"] > tau_max).astype(int)
        elif strat == "weighted_average":
            tau_w = self.thresholds.get("tau_weighted_average", 0.5)
            y_pred = (scores["hybrid_weighted"] > tau_w).astype(int)
        else:
            raise ValueError(f"Unknown fusion strategy: {strat}")

        return y_pred, scores

    def save_state(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sae_bounds": self.sae_bounds,
            "iforest_bounds": self.iforest_bounds,
            "thresholds": self.thresholds,
            "fusion_strategy": self.fusion_strategy,
            "alpha": self.alpha,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    def load_state(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.sae_bounds = tuple(payload["sae_bounds"])
        self.iforest_bounds = tuple(payload["iforest_bounds"])
        self.thresholds = payload["thresholds"]
        self.fusion_strategy = payload.get("fusion_strategy", "or_rule")
        self.alpha = payload.get("alpha", 0.5)
