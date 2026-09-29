"""
Configuration module for Hybrid SAE + Isolation Forest NIDS pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    # Base Directories & Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default_factory=lambda: Path("notebooks/prepared_data"))
    artifacts_dir: Path = field(default_factory=lambda: Path("notebooks/artifacts"))
    results_dir: Path = field(default_factory=lambda: Path("notebooks/results"))

    # Parquet partitions
    label_column: str = "Label"
    attack_column: str = "Attack"
    seed: int = 42

    # SAE Hyperparameters
    encoder_dims: List[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16
    decoder_dims: List[int] = field(default_factory=lambda: [32, 64])
    output_activation: str = "sigmoid"

    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 5
    max_batches_per_epoch: Optional[int] = None
    device: str = "auto"

    # Cascaded Isolation Forest Hyperparameters
    iforest_n_estimators: int = 100
    iforest_max_samples: str = "auto"
    iforest_n_jobs: int = -1
    # feature_mode: 'latent_mse', 'latent_residual', 'latent_only'
    iforest_feature_mode: str = "latent_mse"
    max_train_samples_for_iforest: int = 50000
    iforest_calibration_ratio: float = 0.2

    # Hybrid & Decision Thresholds
    threshold_percentile: float = 95.0
    fusion_strategy: str = "or_rule"
    alpha: float = 0.5

    # Evaluation Options
    eval_batch_size: int = 256
    eval_max_benign_samples: Optional[int] = 25000
    eval_max_attack_samples: Optional[int] = 25000

    def resolve_paths(self):
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        if not self.artifacts_dir.is_absolute():
            self.artifacts_dir = self.base_dir / self.artifacts_dir
        if not self.results_dir.is_absolute():
            self.results_dir = self.base_dir / self.results_dir

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "benign_train_parts"

    @property
    def test_benign_dir(self) -> Path:
        return self.data_dir / "benign_test_parts"

    @property
    def unseen_attack_dir(self) -> Path:
        return self.data_dir / "unseen_attack_test_parts"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "partition_metadata.json"

    @property
    def scaler_path(self) -> Path:
        return self.artifacts_dir / "minmax_params.json"

    @property
    def sae_weights_path(self) -> Path:
        return self.artifacts_dir / "sae_autoencoder.pth"

    @property
    def sae_config_path(self) -> Path:
        return self.artifacts_dir / "sae_config.json"

    @property
    def iforest_model_path(self) -> Path:
        return self.artifacts_dir / "cascaded_iforest.joblib"
