"""
Halaman Evaluasi — Metrik Final, Breakdown per Attack, dan Threshold Lab
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluasi Model & Threshold Lab")

# ============================================================
# TAB 1: METRIK MASTER
# ============================================================
tab_metrics, tab_breakdown, tab_lab = st.tabs([
    "📈 Metrik Komparatif", "🎯 Breakdown per Attack", "🎚️ Threshold Lab"
])

with tab_metrics:
    st.subheader("Tabel Master Evaluasi (N = 50.000 Test Samples)")
    
    # Data diambil persis dari hasil notebook Cell 20
    data = [
        {"Model": "Baseline 1: SAE Alone", "FPR": 4.40, "Recall": 66.73, "F1": 0.7799, "AUC": 0.8813},
        {"Model": "Baseline 2: IForest Alone", "FPR": 5.21, "Recall": 69.40, "F1": 0.7950, "AUC": 0.8672},
        {"Model": "Eksperimen A: IF(Z)", "FPR": 5.17, "Recall": 87.55, "F1": 0.9086, "AUC": 0.9609},
        {"Model": "Proposed: Max-Rank (P97)", "FPR": 2.93, "Recall": 90.50, "F1": 0.9358, "AUC": 0.9834},
        {"Model": "Proposed: Max-Rank (P95)", "FPR": 4.93, "Recall": 93.24, "F1": 0.9410, "AUC": 0.9834},
    ]
    df_master = pd.DataFrame(data)
    
    st.dataframe(df_master, use_container_width=True)

    fig = px.bar(
        df_master,
        x="Model",
        y=["FPR", "Recall"],
        barmode="group",
        title="Perbandingan FPR vs Recall",
        color_discrete_map={"FPR": "#e74c3c", "Recall": "#3498db"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: BREAKDOWN PER ATTACK
# ============================================================
with tab_breakdown:
    st.subheader("Recall per Kategori Zero-Day Attack")
    
    # Data dari notebook Cell 16 (Persentil 97)
    data_p97 = [
        {"Attack": "Bot", "Recall": 100.00, "F1": 0.9318},
        {"Attack": "Brute Force", "Recall": 87.62, "F1": 0.8664},
        {"Attack": "DDoS", "Recall": 87.96, "F1": 0.8683},
        {"Attack": "DoS", "Recall": 87.64, "F1": 0.8665},
        {"Attack": "Reconnaissance", "Recall": 89.30, "F1": 0.8757},
    ]
    df_breakdown = pd.DataFrame(data_p97)

    fig = px.bar(
        df_breakdown,
        x="Attack",
        y="Recall",
        title="Recall per Kategori (Threshold Persentil 97%)",
        color="Recall",
        color_continuous_scale="RdYlGn",
        text="Recall"
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_yaxes(range=[0, 110])
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 **Insight:** Model sangat kuat mendeteksi serangan volumetrik (Bot 100%, DDoS ~88%). Serangan stealthy seperti Brute Force dan Reconnaissance tertangkap di kisaran 87-89% berkat kontribusi Isolation Forest pada latent space.")

# ============================================================
# TAB 3: THRESHOLD LAB (INTERAKTIF)
# ============================================================
with tab_lab:
    st.subheader("🎚️ Threshold Sensitivity Lab (Conformal Prediction)")
    st.markdown("""
    Geser slider di bawah ini untuk mensimulasikan perubahan ambang batas ($\\tau$). 
    Berdasarkan teori **Conformal Prediction**, jika kita mengambil persentil ke-$P$ dari distribusi skor benign, 
    kita memiliki jaminan teoretis bahwa **False Positive Rate (FPR) marginal akan $\le (100 - P)\%$**.
    """)

    pct = st.slider("Pilih Persentil Threshold (P)", 90.0, 99.9, 97.0, 0.1)
    
    # Load calibration refs untuk menghitung tau secara real-time
    try:
        sae_ref = np.load("artifacts/sae_calib_norm_ref.npy")
        ifz_ref = np.load("artifacts/ifz_calib_norm_ref.npy")
        
        # Hitung CDF rank
        sae_cdf = np.searchsorted(np.sort(sae_ref), sae_ref, side="right") / len(sae_ref)
        if_cdf = np.searchsorted(np.sort(ifz_ref), ifz_ref, side="right") / len(ifz_ref)
        
        # Max-Rank Fusion pada calibration set
        max_rank_calib = np.maximum(sae_cdf, if_cdf)
        
        # Hitung tau
        tau_sim = float(np.percentile(max_rank_calib, pct))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Persentil (P)", f"{pct}%")
        col2.metric("Threshold (τ)", f"{tau_sim:.4f}")
        col3.metric("Jaminan FPR Maks", f"≤ {100 - pct:.1f}%")
        
        st.success(f"Pada persentil **{pct}%**, sistem akan men-flag flow sebagai attack jika skor Hybrid > **{tau_sim:.4f}**. Secara teoretis, ini menjamin alarm palsu (FPR) tidak akan melebihi **{100 - pct:.1f}%**.")
        
    except Exception as e:
        st.error(f"Gagal memuat artefak kalibrasi: {e}")