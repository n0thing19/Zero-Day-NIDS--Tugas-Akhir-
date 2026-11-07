from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ==============================================================================
# 1. KONFIGURASI PATH & PEMUATAN DATA
# ==============================================================================
NOTEBOOKS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = NOTEBOOKS_DIR / "artifacts"
RESULTS_DIR = NOTEBOOKS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Muat skor pengujian eksak dan skor referensi kalibrasi normal
sae_test_norm = np.load(ARTIFACTS_DIR / "sae_test_norm_exact.npy")
ifz_test_norm = np.load(ARTIFACTS_DIR / "ifz_test_norm_exact.npy")
y_true = np.load(ARTIFACTS_DIR / "y_true_exact.npy")

sae_calib_ref = np.load(ARTIFACTS_DIR / "sae_calib_norm_ref.npy")
ifz_calib_ref = np.load(ARTIFACTS_DIR / "ifz_calib_norm_ref.npy")

# ==============================================================================
# 2. TRANSFORMASI CDF EMPIRIS & FUSI MAX-RANK
# ==============================================================================
def cdf_against(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_sorted = np.sort(np.asarray(ref, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(ref_sorted, s, side="right") / len(ref_sorted)

# Skala CDF terhadap referensi kalibrasi masing-masing
sae_calib_c = cdf_against(sae_calib_ref, sae_calib_ref)
ifz_calib_c = cdf_against(ifz_calib_ref, ifz_calib_ref)

sae_test_c = cdf_against(sae_test_norm, sae_calib_ref)
ifz_test_c = cdf_against(ifz_test_norm, ifz_calib_ref)

# Fusi Max-Rank
score_max_calib = np.maximum(sae_calib_c, ifz_calib_c)
score_max_test = np.maximum(sae_test_c, ifz_test_c)

# ==============================================================================
# 3. PENGUJIAN AMBANG BATAS PERSENTIL 80% - 99%
# ==============================================================================
percentiles = np.arange(80.0, 99.5, 0.5)
records = []

for pct in percentiles:
    tau = float(np.percentile(score_max_calib, pct))
    y_pred = (score_max_test > tau).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0.0
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5

    records.append({
        "Percentile": pct,
        "Threshold": tau,
        "FPR": fpr,
        "Recall": rec,
        "FNR": fnr,
        "Precision": prec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "F1_x100": f1 * 100,
        "Diff_FPR_FNR": abs(fpr - fnr)
    })

df_metrics = pd.DataFrame(records)

# Cari titik EER dan F1 Optimum
eer_idx = df_metrics["Diff_FPR_FNR"].idxmin()
eer_row = df_metrics.loc[eer_idx]
eer_pct = eer_row["Percentile"]
eer_val = (eer_row["FPR"] + eer_row["FNR"]) / 2

f1_idx = df_metrics["F1-Score"].idxmax()
f1_row = df_metrics.loc[f1_idx]
f1_opt_pct = f1_row["Percentile"]

# Simpan CSV dengan format kolom sesuai permintaan
target_cols = [
    "Percentile",
    "Threshold",
    "FPR",
    "Recall",
    "FNR",
    "Precision",
    "F1-Score",
    "ROC-AUC"
]
csv_output_path = RESULTS_DIR / "percentile_metrics_analysis.csv"
df_metrics[target_cols].to_csv(csv_output_path, index=False)
print(f"[OK] Data metrik berhasil disimpan ke: {csv_output_path}")

# ==============================================================================
# 4. PLOTTING GRAFIK KOMPARATIF DENGAN PENANDA PROPOSED (97%)
# ==============================================================================
PROPOSED_PCT = 97.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Subplot 1: Trade-off Error Rates (FPR vs FNR)
ax1.plot(df_metrics["Percentile"], df_metrics["FPR"], marker="o", markersize=4, 
         color="#1f77b4", linewidth=1.5, label="FPR (False Positive Rate)")
ax1.plot(df_metrics["Percentile"], df_metrics["FNR"], marker="s", markersize=4, 
         color="#d62728", linewidth=1.5, label="FNR (False Negative Rate)")
ax1.axvline(x=eer_pct, color="gray", linestyle="--", linewidth=1.5, 
            label=f"EER ~ {eer_pct:.1f}% (Error ~ {eer_val:.2f}%)")

# Garis Proposed pada 97%
ax1.axvline(x=PROPOSED_PCT, color="#b20a2c", linestyle="-.", linewidth=2, 
            label=f"Proposed ({PROPOSED_PCT:.0f}%)")
ax1.text(PROPOSED_PCT + 0.3, ax1.get_ylim()[1] * 0.85, "Proposed (97%)", 
         color="#b20a2c", fontsize=10, fontweight="bold", rotation=90, verticalalignment="top")

ax1.set_title("Trade-off Error Rates (FPR vs FNR)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Error Rate (%)", fontsize=11, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper center", fontsize=10, framealpha=0.9)

# Subplot 2: Kurva Precision, Recall, dan F1-Score
ax2.plot(df_metrics["Percentile"], df_metrics["Precision"], marker="^", markersize=4, 
         color="#2ca02c", linewidth=1.5, label="Precision (%)")
ax2.plot(df_metrics["Percentile"], df_metrics["Recall"], marker="v", markersize=4, 
         color="#ff7f0e", linewidth=1.5, label="Recall (%)")
ax2.plot(df_metrics["Percentile"], df_metrics["F1_x100"], marker="d", markersize=4, 
         color="#9467bd", linewidth=1.5, label="F1-Score (x100)")
ax2.axvline(x=f1_opt_pct, color="#7f7f7f", linestyle=":", linewidth=1.5, 
            label=f"Optimum F1 ({f1_opt_pct:.1f}%)")

# Garis Proposed pada 97%
ax2.axvline(x=PROPOSED_PCT, color="#b20a2c", linestyle="-.", linewidth=2, 
            label=f"Proposed ({PROPOSED_PCT:.0f}%)")
ax2.text(PROPOSED_PCT + 0.3, ax2.get_ylim()[0] + 15, "Proposed (97%)", 
         color="#b20a2c", fontsize=10, fontweight="bold", rotation=90, verticalalignment="bottom")

ax2.set_title("Kurva Precision, Recall, dan F1-Score", fontsize=12, fontweight="bold")
ax2.set_xlabel("Threshold Persentil CDF (%)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Skor Performa (%)", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)

# Format sumbu X
ax2.set_xticks(np.arange(80, 100, 2))
ax2.set_xlim(79.5, 99.5)

plt.tight_layout()
chart_path = RESULTS_DIR / "tradeoff_eer_curve_with_proposed.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"[OK] Grafik berhasil dibuat dan disimpan di: {chart_path}")