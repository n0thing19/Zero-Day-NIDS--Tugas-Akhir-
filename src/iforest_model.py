"""
Cascaded Isolation Forest trained on SAE latent representations and reconstruction errors.
"""

from pathlib import Path
from typing import Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from .sae_model import StackedAutoencoder


class CascadedIForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str = "auto",
        n_jobs: int = -1,
        random_state: int = 42,
        feature_mode: str = "latent_mse",
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.feature_mode = feature_mode

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination="auto",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def extract_features(
        self, X: np.ndarray, sae_model: StackedAutoencoder
    ) -> np.ndarray:
        Z = sae_model.get_latent_features(X)

        if self.feature_mode == "latent_mse":
            mse = sae_model.get_reconstruction_error(X).reshape(-1, 1)
            return np.hstack([Z, mse])
        elif self.feature_mode == "latent_residual":
            res = sae_model.get_reconstruction_residuals(X)
            return np.hstack([Z, res])
        elif self.feature_mode == "latent_only":
            return Z
        else:
            raise ValueError(f"Unknown feature_mode: {self.feature_mode}")

    def fit(self, X: np.ndarray, sae_model: StackedAutoencoder) -> "CascadedIForest":
        cascaded_features = self.extract_features(X, sae_model)
        self.model.fit(cascaded_features)
        return self

    def score_samples(
        self, X: np.ndarray, sae_model: StackedAutoencoder
    ) -> np.ndarray:
        cascaded_features = self.extract_features(X, sae_model)
        raw_scores = self.model.score_samples(cascaded_features)
        return -raw_scores.astype(np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_mode": self.feature_mode,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "CascadedIForest":
        data = joblib.load(path)
        instance = cls(
            n_estimators=data["n_estimators"],
            random_state=data["random_state"],
            feature_mode=data["feature_mode"],
        )
        instance.model = data["model"]
        return instance
