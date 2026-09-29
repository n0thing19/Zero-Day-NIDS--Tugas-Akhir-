"""
Data loading, batch streaming, and scaling utilities for NIDS parquet datasets.
"""

import json
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


def load_partition_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Partition metadata not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scaler_params(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Scaler parameters not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mins = np.array(data["min"], dtype=np.float32)
    maxs = np.array(data["max"], dtype=np.float32)
    return mins, maxs


def save_scaler_params(path: Path, mins: np.ndarray, maxs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "feature_count": int(len(mins)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def minmax_transform(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = maxs - mins
    denom = np.where(np.isclose(denom, 0.0), 1.0, denom)
    scaled = (X - mins) / denom
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def stream_train_batches_scaled(
    files: List[Path],
    feature_columns: List[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    batch_size: int = 256,
    shuffle_files: bool = True,
    shuffle_rows: bool = True,
    seed: int = 42,
) -> Generator[np.ndarray, None, None]:
    rng = np.random.default_rng(seed)
    file_list = list(files)

    if shuffle_files:
        rng.shuffle(file_list)

    for file_path in file_list:
        df = pd.read_parquet(file_path, columns=feature_columns)
        if df.shape[0] == 0:
            continue

        X = df.to_numpy(dtype=np.float32)
        finite_mask = np.isfinite(X).all(axis=1)
        if not finite_mask.all():
            X = X[finite_mask]

        if X.shape[0] == 0:
            continue

        X_scaled = minmax_transform(X, mins, maxs)

        if shuffle_rows:
            perm = rng.permutation(X_scaled.shape[0])
            X_scaled = X_scaled[perm]

        for start in range(0, X_scaled.shape[0], batch_size):
            yield X_scaled[start : start + batch_size]


def load_evaluation_dataset(
    files: List[Path],
    feature_columns: List[str],
    mins: np.ndarray,
    maxs: np.ndarray,
    label_column: str = "Label",
    attack_column: str = "Attack",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    file_list = list(files)
    rng.shuffle(file_list)

    X_list, y_list, attack_list = [], [], []
    total = 0

    for f in file_list:
        if max_samples is not None and total >= max_samples:
            break

        cols_to_read = feature_columns + [label_column]
        try:
            df = pd.read_parquet(f, columns=cols_to_read + [attack_column])
        except Exception:
            df = pd.read_parquet(f, columns=cols_to_read)
            df[attack_column] = np.where(df[label_column] == 0, "Benign", "Attack")

        if df.shape[0] == 0:
            continue

        X = df[feature_columns].to_numpy(dtype=np.float32)
        finite_mask = np.isfinite(X).all(axis=1)
        if not finite_mask.all():
            X = X[finite_mask]
            df = df.loc[finite_mask]

        if X.shape[0] == 0:
            continue

        X_scaled = minmax_transform(X, mins, maxs)
        y = df[label_column].astype(int).to_numpy()
        attacks = df[attack_column].astype(str).to_numpy()

        perm = rng.permutation(len(X_scaled))
        X_scaled = X_scaled[perm]
        y = y[perm]
        attacks = attacks[perm]

        if max_samples is not None:
            remain = max_samples - total
            if remain <= 0:
                break
            X_scaled = X_scaled[:remain]
            y = y[:remain]
            attacks = attacks[:remain]

        X_list.append(X_scaled)
        y_list.append(y)
        attack_list.append(attacks)
        total += len(X_scaled)

    if len(X_list) == 0:
        return (
            np.empty((0, len(feature_columns)), dtype=np.float32),
            np.empty(0, dtype=int),
            np.empty(0, dtype=object),
        )

    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(attack_list),
    )
