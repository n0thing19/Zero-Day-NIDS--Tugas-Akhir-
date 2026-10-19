"""
Halaman Tentang — Metodologi, Rumus, dan Informasi Dataset
"""
import streamlit as st

st.set_page_config(page_title="Tentang", page_icon="📚", layout="wide")
st.title("📚 Tentang Metode & Metodologi")

# ============================================================
# METODOLOGI
# ============================================================
st.subheader("🎯 Arsitektur Hybrid NIDS")
st.markdown("""
Sistem ini mendeteksi *zero-day attack* pada trafik jaringan (NetFlow) tanpa memerlukan label serangan saat training. 
Metode ini menggabungkan dua paradigma deteksi anomali:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 1. Stacked Autoencoder (SAE)")
    st.markdown("""
    - **Fungsi:** Mempelajari manifold distribusi trafik normal.
    - **Arsitektur:** Encoder kompresi $39 \to 64 \to 32 \to 16$ (Latent $Z$).
    - **Output:** Reconstruction Error (MSE). Jika input adalah attack, SAE gagal merekonstruksi $\to$ MSE tinggi.
    """)

with col2:
    st.markdown("#### 2. Isolation Forest pada Latent Z")
    st.markdown("""
    - **Fungsi:** Mendeteksi outlier di ruang laten $Z \in \mathbb{R}^{16}$.
    - **Keunggulan:** IF bekerja sangat baik pada ruang dimensi rendah yang sudah dipadatkan oleh SAE.
    - **Pembuktian:** Eksperimen menunjukkan IF pada $Z$ murni (AUC 0.961) jauh lebih baik daripada IF pada $[Z, MSE]$ (AUC 0.867).
    """)

st.divider()

# ============================================================
# RUMUS FUSI
# ============================================================
st.subheader("🧮 Max-Rank Fusion & Conformal Threshold")
st.markdown("""
Skor mentah dari SAE ($S_1$) dan IForest ($S_2$) memiliki skala berbeda. Kami menormalisasinya menggunakan **Empirical CDF (Rank)** terhadap data kalibrasi benign, lalu menggabungkannya dengan aturan **MAX**:
""")

st.latex(r'''
    c_i = F_{benign}(S_i) \in [0,1]
''')
st.latex(r'''
    H = \max(c_{SAE}, c_{IF})
''')

st.markdown("""
**Aturan Keputusan:**
Sebuah flow diklasifikasikan sebagai *Zero-Day Attack* jika:
""")
st.latex(r'''
    H > \tau_{97}
''')
st.markdown("""
Di mana $\tau_{97}$ adalah persentil ke-97 dari distribusi $H$ pada data benign. Pendekatan ini memberikan jaminan teoretis *Conformal Prediction* bahwa False Positive Rate (FPR) $\le 3\%$.
""")

st.divider()

# ============================================================
# DATASET
# ============================================================
st.subheader("🗄️ Informasi Dataset")
st.markdown("""
- **Nama Dataset:** NF-UQ-NIDS-v2
- **Format:** NetFlow (39 extended features)
- **Skenario:** Zero-Day / Unseen Attack
- **Data Training:** 100% Trafik Benign (Normal)
- **Data Testing:** Campuran Benign + 5 Kategori Serangan Unseen (Bot, Brute Force, DDoS, DoS, Reconnaissance)
""")

st.divider()
st.caption("Dikembangkan untuk keperluan Skripsi • 2026")