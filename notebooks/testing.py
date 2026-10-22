import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

# ==============================================================================
# 1. KONFIGURASI PATH
# ==============================================================================
NOTEBOOKS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = NOTEBOOKS_DIR / "artifacts"

# Muat array skor eksak hasil sesi notebook
sae_test_norm = np.load(ARTIFACTS_DIR / "sae_test_norm_exact.npy")
ifz_test_norm = np.load(ARTIFACTS_DIR / "ifz_test_norm_exact.npy")
y_true = np.load(ARTIFACTS_DIR / "y_true_exact.npy")
attack_names = np.load(ARTIFACTS_DIR / "attack_test_exact.npy", allow_pickle=True)

# Muat skor referensi kalibrasi CDF normal
sae_calib_ref = np.load(ARTIFACTS_DIR / "sae_calib_norm_ref.npy")
ifz_calib_ref = np.load(ARTIFACTS_DIR / "ifz_calib_norm_ref.npy")

# ==============================================================================
# 2. TRANSFORMASI CDF EMPIRIS & FUSI MAX-RANK
# ==============================================================================
def cdf_against(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(np.asarray(ref, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(ref_sorted, s, side="right") / len(ref_sorted)

sae_test_c = cdf_against(sae_test_norm, sae_calib_ref)
ifz_test_c = cdf_against(ifz_test_norm, ifz_calib_ref)

# Operasi Fusi Max-Rank
score_max_rank = np.maximum(sae_test_c, ifz_test_c)

# Threshold Persentil ke-97
tau_97 = 0.983203

# ==============================================================================
# 3. EVALUASI BINER
# ==============================================================================
pred_p97 = (score_max_rank > tau_97).astype(int)

cm = confusion_matrix(y_true, pred_p97, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
f1 = f1_score(y_true, pred_p97, pos_label=1, zero_division=0)

fpr_arr, tpr_arr, _ = roc_curve(y_true, score_max_rank)
roc_auc = auc(fpr_arr, tpr_arr)

# ==============================================================================
# 4. CETAK HASIL
# ==============================================================================
print("=" * 85)
print("HASIL EVALUASI MASTER: MAX-RANK (P97)")
print("=" * 85)
print(f"Threshold : {tau_97:.4f}")
print(f"FPR (%)   : {fpr:.2f}%")
print(f"Recall (%): {rec:.2f}%")
print(f"Precision : {prec:.2f}%")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("-" * 85)
print(f"Confusion Matrix -> TP: {tp:,} | FP: {fp:,} | TN: {tn:,} | FN: {fn:,}")
print("=" * 85)

print("\n=== BREAKDOWN PER KATEGORI SERANGAN ===")
benign_mask = (y_true == 0)
for cat in sorted(set(attack_names[y_true == 1])):
    mask = benign_mask | (attack_names == cat)
    y_sub = y_true[mask]
    p_sub = pred_p97[mask]
    cm_sub = confusion_matrix(y_sub, p_sub, labels=[0, 1])
    _, _, fn_s, tp_s = cm_sub.ravel()
    rec_s = (tp_s / (tp_s + fn_s)) * 100 if (tp_s + fn_s) > 0 else 0.0
    print(f"- {cat:<15}: TP = {tp_s:,}/{tp_s + fn_s:,} | Recall = {rec_s:.2f}%")