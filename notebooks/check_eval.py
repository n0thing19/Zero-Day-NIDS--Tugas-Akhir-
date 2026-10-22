import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve
import torch
import torch.nn as nn

# ==============================================================================
# 1. PATH & REPRODUCIBILITY
# ==============================================================================
NOTEBOOKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent

DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
ARTIFACTS_DIR = NOTEBOOKS_DIR / "artifacts"

META_PATH = DATA_DIR / "partition_metadata.json"
SCALER_PATH = ARTIFACTS_DIR / "minmax_params.json"
SAE_WEIGHTS_PATH = ARTIFACTS_DIR / "sae_autoencoder.pth"  # atau sae_autoencoder_final.pth
TRAIN_DIR = DATA_DIR / "benign_train_parts"
TEST_BENIGN_DIR = DATA_DIR / "benign_test_parts"
UNSEEN_ATTACK_DIR = DATA_DIR / "unseen_attack_test_parts"

SEED = 42
MAX_TRAIN_SAMPLES_IFOREST = 50_000
IFOREST_CALIB_RATIO = 0.20
IFOREST_ESTIMATORS = 100
LATENT_DIM = 16
EVAL_BATCH_SIZE = 256
EVAL_MAX_BENIGN_SAMPLES = 25_000
EVAL_MAX_ATTACK_SAMPLES = 25_000

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 2. DEFINISI MODEL & FUNGSI HELPER
# ==============================================================================
def minmax_transform_np(X: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    denom = maxs - mins
    denom = np.where(np.isclose(denom, 0.0), 1.0, denom).astype(np.float32)
    return np.clip((X - mins) / denom, 0.0, 1.0).astype(np.float32)

def normalize_with_bounds(scores: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    min_val, max_val = bounds
    denom = max_val - min_val
    if np.isclose(denom, 0.0):
        return np.zeros_like(s, dtype=np.float32)
    return np.clip((s - min_val) / denom, 0.0, 1.0).astype(np.float32)

def cdf_against(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(np.asarray(ref, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(ref_sorted, s, side="right") / len(ref_sorted)

class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim: int, encoder_dims: List[int] = None, latent_dim: int = 16, decoder_dims: List[int] = None):
        super().__init__()
        encoder_dims = encoder_dims or [64, 32]
        decoder_dims = decoder_dims or [32, 64]
        self.latent_dim = latent_dim

        enc = []
        prev = input_dim
        for d in encoder_dims:
            enc.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        enc.extend([nn.Linear(prev, latent_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for d in decoder_dims:
            dec.extend([nn.Linear(prev, d), nn.ReLU()])
            prev = d
        dec.extend([nn.Linear(prev, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def get_latent_features(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        dev = next(self.parameters()).device
        latents = []
        for start in range(0, len(arr), batch_size):
            b_tensor = torch.from_numpy(arr[start : start + batch_size]).to(dev)
            latents.append(self.encoder(b_tensor).detach().cpu().numpy())
        return np.vstack(latents).astype(np.float32)

    @torch.no_grad()
    def get_reconstruction_error(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.eval()
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(arr) == 0:
            return np.empty(0, dtype=np.float32)
        dev = next(self.parameters()).device
        errors = []
        for start in range(0, len(arr), batch_size):
            b_tensor = torch.from_numpy(arr[start : start + batch_size]).to(dev)
            recon = self(b_tensor)
            mse = torch.mean((b_tensor - recon) ** 2, dim=1)
            errors.append(mse.detach().cpu().numpy())
        return np.concatenate(errors).astype(np.float32)

# ==============================================================================
# 3. MEMUAT METADATA, BOBOT SAE, DAN KALIBRASI LATEN
# ==============================================================================
print("=== [1/4] MEMUAT ARTEFAK & KALIBRASI DASAR ===")

with open(META_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)["feature_columns"]

with open(SCALER_PATH, "r", encoding="utf-8") as f:
    scaler_info = json.load(f)
mins = np.array(scaler_info["min"], dtype=np.float32)
maxs = np.array(scaler_info["max"], dtype=np.float32)

sae_model = StackedAutoencoder(input_dim=len(feature_columns), latent_dim=LATENT_DIM).to(device)
weights_file = SAE_WEIGHTS_PATH if SAE_WEIGHTS_PATH.exists() else (ARTIFACTS_DIR / "sae_autoencoder_final.pth")
sae_model.load_state_dict(torch.load(weights_file, map_location=device))
sae_model.eval()

# Ekstraksi 50.000 sampel latih normal untuk memastikan kalibrasi persis Cell 3 & 10
train_files = sorted(TRAIN_DIR.glob("*.parquet"))
rng_train = np.random.default_rng(SEED)
shuffled_train = list(train_files)
rng_train.shuffle(shuffled_train)

Z_list, S_list = [], []
total_collected = 0
for fpath in shuffled_train:
    if total_collected >= MAX_TRAIN_SAMPLES_IFOREST:
        break
    df = pd.read_parquet(fpath, columns=feature_columns)
    if df.empty:
        continue
    X = df.to_numpy(dtype=np.float32)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if len(X) == 0:
        continue
    X_scaled = minmax_transform_np(X, mins, maxs)
    perm = rng_train.permutation(len(X_scaled))
    X_scaled = X_scaled[perm]
    rem = MAX_TRAIN_SAMPLES_IFOREST - total_collected
    X_batch = X_scaled[:rem]

    Z_list.append(sae_model.get_latent_features(X_batch))
    S_list.append(sae_model.get_reconstruction_error(X_batch))
    total_collected += len(X_batch)

Z_all = np.vstack(Z_list)
sae_train_raw = np.concatenate(S_list)

calib_len = int(len(Z_all) * IFOREST_CALIB_RATIO)
Z_fit = Z_all[calib_len:]
Z_calib = Z_all[:calib_len]
sae_calib_raw = sae_train_raw[:calib_len]

# Melatih IForest Exp A (Murni 16 Dimensi Laten Z)
iforest_A = IsolationForest(
    n_estimators=IFOREST_ESTIMATORS,
    contamination="auto",
    random_state=SEED,
    n_jobs=-1,
)
iforest_A.fit(Z_fit)

sae_bounds = (float(np.min(sae_train_raw)), float(np.max(sae_train_raw)))
if_calib_raw_A = -iforest_A.score_samples(Z_calib)
if_bounds_A = (float(np.min(if_calib_raw_A)), float(np.max(if_calib_raw_A)))

sae_calib_norm = normalize_with_bounds(sae_calib_raw, sae_bounds)
if_calib_norm_A = normalize_with_bounds(if_calib_raw_A, if_bounds_A)

# Transformasi CDF Kalibrasi
sae_calib_c = cdf_against(sae_calib_norm, sae_calib_norm)
ifz_calib_c = cdf_against(if_calib_norm_A, if_calib_norm_A)
fc_max = np.maximum(sae_calib_c, ifz_calib_c)
tau_97 = float(np.percentile(fc_max, 97))

print(f"[OK] Kalibrasi selesai. Threshold Max-Rank P97: {tau_97:.4f}")

# ==============================================================================
# 4. PENGUMPULAN SKOR TEST SET PERSIS CELL 4 & CELL 10
# ==============================================================================
print("\n=== [2/4] MENGEKSTRAK SKOR DATA UJI (N = 50.000) ===")

test_benign_files = sorted(TEST_BENIGN_DIR.glob("*.parquet"))
attack_subfolders = sorted([d for d in UNSEEN_ATTACK_DIR.iterdir() if d.is_dir()])
samples_per_category = EVAL_MAX_ATTACK_SAMPLES // len(attack_subfolders) if len(attack_subfolders) > 0 else 5000

def collect_scores_single_pass(files: List[Path], max_samples: int, rng_gen) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = list(files)
    rng_gen.shuffle(shuffled)
    s_l, i_l, y_l = [], [], []
    tot = 0
    for fpath in shuffled:
        if tot >= max_samples:
            break
        df = pd.read_parquet(fpath, columns=feature_columns + ["Label"])
        if df.empty:
            continue
        X = df[feature_columns].to_numpy(dtype=np.float32)
        mask = np.isfinite(X).all(axis=1)
        if not mask.all():
            X = X[mask]
            df = df.loc[mask]
        if len(X) == 0:
            continue
        X_scaled = minmax_transform_np(X, mins, maxs)
        y = df["Label"].astype(int).to_numpy()
        rem = max_samples - tot
        bx = X_scaled[:rem]
        by = y[:rem]

        for start in range(0, len(bx), EVAL_BATCH_SIZE):
            batch_x = bx[start : start + EVAL_BATCH_SIZE]
            batch_y = by[start : start + EVAL_BATCH_SIZE]

            zb = sae_model.get_latent_features(batch_x)
            s_raw = sae_model.get_reconstruction_error(batch_x)
            i_raw = -iforest_A.score_samples(zb)

            s_l.append(normalize_with_bounds(s_raw, sae_bounds))
            i_l.append(normalize_with_bounds(i_raw, if_bounds_A))
            y_l.append(batch_y)
        tot += len(bx)
    return np.concatenate(s_l), np.concatenate(i_l), np.concatenate(y_l)

# Ekstraksi Benign (25.000)
rng_benign = np.random.default_rng(SEED)
benign_sae, benign_if, benign_y = collect_scores_single_pass(test_benign_files, EVAL_MAX_BENIGN_SAMPLES, rng_benign)

# Ekstraksi Attack (5 x 5.000 dengan generator tunggal yang terus berjalan)
rng_attack = np.random.default_rng(SEED + 1)
atk_s_l, atk_i_l, atk_y_l, atk_cat_l = [], [], [], []
for folder in attack_subfolders:
    files_in_f = sorted(folder.glob("*.parquet"))
    if len(files_in_f) == 0:
        continue
    s_arr, i_arr, y_arr = collect_scores_single_pass(files_in_f, samples_per_category, rng_attack)
    atk_s_l.append(s_arr)
    atk_i_l.append(i_arr)
    atk_y_l.append(y_arr)
    atk_cat_l.append(np.full(len(y_arr), folder.name, dtype=object))

sae_test_norm = np.concatenate([benign_sae, np.concatenate(atk_s_l)])
ifz_test_norm = np.concatenate([benign_if, np.concatenate(atk_i_l)])
y_true = np.concatenate([benign_y, np.concatenate(atk_y_l)])
attack_names = np.concatenate([np.full(len(benign_y), "Benign", dtype=object), np.concatenate(atk_cat_l)])

# ==============================================================================
# 5. TRANSFORMASI CDF & KALKULASI METRIK
# ==============================================================================
print("\n=== [3/4] MENGHITUNG FUSI MAX-RANK P97 ===")

sae_test_c = cdf_against(sae_test_norm, sae_calib_norm)
ifz_test_c = cdf_against(ifz_test_norm, if_calib_norm_A)
score_max_rank = np.maximum(sae_test_c, ifz_test_c)

pred_p97 = (score_max_rank > tau_97).astype(int)
cm = confusion_matrix(y_true, pred_p97, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
f1 = f1_score(y_true, pred_p97, pos_label=1, zero_division=0)

fpr_arr, tpr_arr, _ = roc_curve(y_true, score_max_rank)
roc_auc = auc(fpr_arr, tpr_arr)

# ==============================================================================
# 6. HASIL AKHIR & BREAKDOWN PER KATEGORI
# ==============================================================================
print("\n" + "=" * 90)
print("HASIL EVALUASI MASTER: MAX-RANK (P97)")
print("=" * 90)
print(f"Threshold : {tau_97:.4f}")
print(f"FPR (%)   : {fpr:.2f}%")
print(f"Recall (%): {rec:.2f}%")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("-" * 90)
print(f"Confusion Matrix -> TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
print("=" * 90)

print("\n=== BREAKDOWN PER KATEGORI SERANGAN ===")
benign_mask = (y_true == 0)
for cat in sorted(set(attack_names[y_true == 1])):
    mask = benign_mask | (attack_names == cat)
    y_sub = y_true[mask]
    p_sub = pred_p97[mask]
    cm_sub = confusion_matrix(y_sub, p_sub, labels=[0, 1])
    _, _, fn_s, tp_s = cm_sub.ravel()
    rec_s = (tp_s / (tp_s + fn_s)) * 100
    print(f"- {cat:<15}: TP = {tp_s:,}/{tp_s + fn_s:,} | Recall = {rec_s:.2f}%")