"""
================================================================================
THRESHOLD SWEEP EXTENDED METRICS CALCULATOR & VISUALIZATION
================================================================================
Menghitung TP, FP, TN, FN, Recall, Precision, F1-Score, dan F0.5-Score
berdasarkan proporsi 25.000 Benign (Kelas 0) dan 25.000 Attack (Kelas 1).
================================================================================
"""

import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. KONFIGURASI DAN PATH
# ------------------------------------------------------------------------------
BASE_DIR = Path("i:/skripsi/NIDS")
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
RESULTS_DIR = NOTEBOOKS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_INPUT_PATH = RESULTS_DIR / "threshold_sweep_80_to_99.csv"
CSV_OUTPUT_PATH = RESULTS_DIR / "threshold_sweep_comprehensive_metrics.csv"
PLOT_OUTPUT_PATH = RESULTS_DIR / "metrics_comprehensive_sweep.png"

# Ukuran dataset uji
N_BENIGN = 25_000   # Negatif Aktual
N_ATTACK = 25_000   # Positif Aktual

# Data fallback jika file CSV belum ada di direktori
RAW_CSV_DATA = """Threshold (%),Threshold (Val),FPR (%),FNR (%)
80.0,0.8,27.692,5.024
80.5,0.805,26.928,5.028
81.0,0.81,26.576,5.028
81.5,0.815,25.888,5.124
82.0,0.82,25.176,5.204
82.5,0.825,24.632,5.236
83.0,0.83,23.88,5.244
83.5,0.835,23.164,5.396
84.0,0.84,22.372,5.476
84.5,0.845,21.784,5.544
85.0,0.85,21.232,5.644
85.5,0.855,20.808,5.712
86.0,0.86,20.5,7.56
86.5,0.865,20.16,7.776
87.0,0.87,19.596,7.992
87.5,0.875,18.828,8.0
88.0,0.88,18.124,8.508
88.5,0.885,17.32,9.092
89.0,0.89,16.74,9.136
89.5,0.895,15.852,9.156
90.0,0.9,15.404,9.216
90.5,0.905,14.704,9.34
91.0,0.91,13.892,9.376
91.5,0.915,13.28,9.976
92.0,0.92,12.472,10.188
92.5,0.925,11.884,10.592
93.0,0.93,11.176,10.68
93.5,0.935,10.568,12.052
94.0,0.94,9.908,12.14
94.5,0.945,8.984,12.144
95.0,0.95,8.156,12.224
95.5,0.955,7.436,12.32
96.0,0.96,6.668,12.492
96.5,0.965,5.884,12.624
97.0,0.97,5.144,12.724
97.5,0.975,4.292,12.784
98.0,0.98,3.396,12.904
98.5,0.985,2.648,14.576
99.0,0.99,1.608,15.896"""


def load_dataset() -> pd.DataFrame:
    if CSV_INPUT_PATH.exists():
        return pd.read_csv(CSV_INPUT_PATH)
    return pd.read_csv(io.StringIO(RAW_CSV_DATA))


def main():
    df = load_dataset()

    # --------------------------------------------------------------------------
    # 2. PERHITUNGAN CONFUSION MATRIX DAN METRIK OPERASIONAL
    # --------------------------------------------------------------------------
    # FP = (FPR / 100) * Benign_Total
    # FN = (FNR / 100) * Attack_Total
    # TP = Attack_Total - FN
    # TN = Benign_Total - FP
    df["FP"] = np.round((df["FPR (%)"] / 100.0) * N_BENIGN).astype(int)
    df["FN"] = np.round((df["FNR (%)"] / 100.0) * N_ATTACK).astype(int)
    df["TP"] = N_ATTACK - df["FN"]
    df["TN"] = N_BENIGN - df["FP"]

    # Recall (%) = TP / (TP + FN) * 100 = 100 - FNR
    df["Recall (%)"] = np.round(100.0 - df["FNR (%)"], 3)

    # Precision (%) = TP / (TP + FP) * 100
    df["Precision (%)"] = np.round((df["TP"] / (df["TP"] + df["FP"])) * 100.0, 3)

    # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    p = df["Precision (%)"] / 100.0
    r = df["Recall (%)"] / 100.0
    df["F1-Score"] = np.round(2 * (p * r) / (p + r), 4)

    # F0.5-Score (Mengutamakan Precision / Penekanan False Alarm 2x lipat)
    df["F0.5-Score"] = np.round((1 + 0.5**2) * (p * r) / ((0.5**2 * p) + r), 4)

    # Simpan hasil komprehensif
    df.to_csv(CSV_OUTPUT_PATH, index=False)

    # --------------------------------------------------------------------------
    # 3. IDENTIFIKASI THRESHOLD KRUSIAL
    # --------------------------------------------------------------------------
    idx_max_f1 = df["F1-Score"].idxmax()
    idx_max_f05 = df["F0.5-Score"].idxmax()
    idx_eer = np.abs(df["FPR (%)"] - df["FNR (%)"]).idxmin()

    best_f1_row = df.loc[idx_max_f1]
    best_f05_row = df.loc[idx_max_f05]
    eer_row = df.loc[idx_eer]

    print("=" * 95)
    print("HASIL ANALISIS LENGKAP METRIK SWEEP THRESHOLD (80.0% - 99.0%)")
    print("=" * 95)
    print(f"File hasil tersimpan di: {CSV_OUTPUT_PATH.resolve()}\n")

    display_cols = [
        "Threshold (%)", "TP", "FP", "TN", "FN",
        "FPR (%)", "Recall (%)", "Precision (%)", "F1-Score", "F0.5-Score"
    ]
    print(df[display_cols].to_string(index=False))

    print("\n" + "=" * 95)
    print("RINGKASAN THRESHOLD OPTIMAL")
    print("=" * 95)
    print(f"1. F1-Score Tertinggi (Harmonic Mean Optimal):")
    print(f"   -> Threshold: {best_f1_row['Threshold (%)']}% | F1: {best_f1_row['F1-Score']:.4f} | "
          f"Precision: {best_f1_row['Precision (%)']:.2f}% | Recall: {best_f1_row['Recall (%)']:.2f}% | "
          f"FPR: {best_f1_row['FPR (%)']:.2f}%")

    print(f"\n2. F0.5-Score Tertinggi (Penekanan False Alarm / SOC Priority):")
    print(f"   -> Threshold: {best_f05_row['Threshold (%)']}% | F0.5: {best_f05_row['F0.5-Score']:.4f} | "
          f"Precision: {best_f05_row['Precision (%)']:.2f}% | Recall: {best_f05_row['Recall (%)']:.2f}% | "
          f"FPR: {best_f05_row['FPR (%)']:.2f}%")

    print(f"\n3. Titik Keseimbangan EER (|FPR - FNR| Terkecil):")
    print(f"   -> Threshold: {eer_row['Threshold (%)']}% | Selisih: {abs(eer_row['FPR (%)'] - eer_row['FNR (%)']):.3f}% | "
          f"FPR: {eer_row['FPR (%)']:.2f}% | FNR: {eer_row['FNR (%)']:.2f}%")
    print("=" * 95)

    # --------------------------------------------------------------------------
    # 4. PLOTTING MULTI-METRIK
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=300, sharex=True)

    # Subplot 1: Error Rates (FPR vs FNR)
    ax1.plot(df["Threshold (%)"], df["FPR (%)"], marker="o", markersize=3.5, color="#1f77b4", label="FPR (False Positive Rate)")
    ax1.plot(df["Threshold (%)"], df["FNR (%)"], marker="s", markersize=3.5, color="#d62728", label="FNR (False Negative Rate)")
    ax1.axvline(eer_row["Threshold (%)"], color="gray", linestyle="--", alpha=0.7, label=f"EER ~ {eer_row['Threshold (%)']}%")
    ax1.set_ylabel("Error Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Trade-off Error Rates (FPR vs FNR)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper center", frameon=True)

    # Subplot 2: Performance Metrics (Precision, Recall, F1-Score)
    ax2.plot(df["Threshold (%)"], df["Precision (%)"], marker="^", markersize=3.5, color="#2ca02c", label="Precision (%)")
    ax2.plot(df["Threshold (%)"], df["Recall (%)"], marker="v", markersize=3.5, color="#ff7f0e", label="Recall (%)")
    ax2.plot(df["Threshold (%)"], df["F1-Score"] * 100.0, marker="D", markersize=3.5, color="#9467bd", linewidth=2.0, label="F1-Score (x100)")
    ax2.axvline(best_f1_row["Threshold (%)"], color="#9467bd", linestyle=":", alpha=0.8, label=f"Optimum F1 ({best_f1_row['Threshold (%)']}%)")
    ax2.set_xlabel("Threshold Persentil CDF (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Skor Performa (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Kurva Precision, Recall, dan F1-Score", fontsize=12, fontweight="bold")
    ax2.set_xticks(np.arange(80, 100, 2))
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH)
    plt.close()
    print(f"\nGrafik metrik komprehensif berhasil disimpan ke: {PLOT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()