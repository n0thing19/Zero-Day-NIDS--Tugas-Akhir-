"""
Halaman Simulasi — Upload CSV atau Input Manual
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Tambahkan parent folder ke path agar bisa import pipeline
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline import NIDSPipeline

st.set_page_config(page_title="Simulasi", page_icon="🧪", layout="wide")
st.title("🧪 Simulasi Deteksi Zero-Day Attack")

# ============================================================
# LOAD PIPELINE (cached)
# ============================================================
@st.cache_resource
def load_pipeline():
    return NIDSPipeline(artifacts_dir=r"I:\skripsi\NIDS\notebooks\artifacts", device="auto")

pipe = load_pipeline()

# ============================================================
# SIDEBAR — PENGATURAN
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold_mode = st.radio(
        "Mode Threshold",
        ["Persentil 97 (FPR rendah)", "Persentil 95 (Recall tinggi)"],
        index=0,
    )
    if "97" in threshold_mode:
        pipe.set_threshold(97)
        st.caption(f"τ = {pipe.tau:.4f} | FPR ≤ 3%")
    else:
        pipe.set_threshold(95)
        st.caption(f"τ = {pipe.tau:.4f} | Recall lebih tinggi")

    st.divider()
    st.caption("Dataset: NF-UQ-NIDS-v2\nFeatures: 39 NetFlow")

# ============================================================
# TAB: UPLOAD vs MANUAL
# ============================================================
tab_upload, tab_manual = st.tabs(["📤 Upload File", "️ Input Manual"])

# ---------- UPLOAD ----------
with tab_upload:
    st.subheader("Upload File CSV dengan 39 Kolom Fitur NetFlow")
    uploaded = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"📊 {len(df)} baris terbaca")

        # Validasi kolom
        missing = [c for c in pipe.feature_columns if c not in df.columns]
        if missing:
            st.error(f"❌ Kolom hilang: {missing[:5]}...")
            st.stop()

        X = df[pipe.feature_columns].to_numpy(dtype=np.float32)
        if np.isnan(X).any():
            st.error("❌ Data mengandung NaN. Lengkapi semua fitur.")
            st.stop()

        # Inferensi
        result = pipe.predict(X)

        # Tampilkan hasil
        df_result = df.copy()
        df_result["SAE_Rank"] = np.round(result["c_sae"], 4)
        df_result["IF_Rank"] = np.round(result["c_if"], 4)
        df_result["Hybrid_Score"] = np.round(result["H"], 4)
        df_result["Verdict"] = np.where(
            result["verdict"] == 1, "🚨 ATTACK", "✅ NORMAL"
        )

        st.dataframe(df_result, use_container_width=True)

        # Statistik
        n_attack = int(result["verdict"].sum())
        n_normal = len(result["verdict"]) - n_attack
        c1, c2 = st.columns(2)
        c1.metric("✅ Normal", n_normal)
        c2.metric("🚨 Attack", n_attack)

        # Distribusi skor
        fig = px.histogram(
            df_result,
            x="Hybrid_Score",
            color="Verdict",
            barmode="overlay",
            title="Distribusi Skor Hybrid",
        )
        fig.add_vline(
            x=pipe.tau,
            line_dash="dash",
            line_color="red",
            annotation_text=f"τ = {pipe.tau:.4f}",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- MANUAL ----------
with tab_manual:
    st.subheader("Input Manual 39 Fitur NetFlow")
    st.caption(
        "Isi nilai fitur NetFlow (contoh: durasi, bytes, packets, dst.). "
        "Gunakan format angka desimal."
    )

    # Buat dataframe kosong dengan 39 kolom fitur
    editor = st.data_editor(
        pd.DataFrame(0.0, index=[0], columns=pipe.feature_columns),
        num_rows="fixed",
        key="manual_editor",
    )

    if st.button("🔍 Analisis", type="primary"):
        X = editor[pipe.feature_columns].to_numpy(dtype=np.float32)
        if np.isnan(X).any():
            st.error(" Ada nilai kosong. Lengkapi semua fitur.")
            st.stop()

        result = pipe.predict(X)

        st.divider()
        st.subheader("Hasil Analisis")

        col1, col2, col3 = st.columns(3)
        col1.metric("SAE Rank (CDF)", f"{result['c_sae'][0]:.4f}")
        col2.metric("IF(Z) Rank (CDF)", f"{result['c_if'][0]:.4f}")
        col3.metric("Hybrid Score", f"{result['H'][0]:.4f}")

        if result["verdict"][0] == 1:
            st.error("🚨 TERDETEKSI SEBAGAI ZERO-DAY ATTACK")
        else:
            st.success("✅ Trafik Tergolong NORMAL")

        st.caption(
            f"Threshold aktif: τ = {pipe.tau:.4f}. "
            "Sampel diklasifikasikan sebagai attack jika Hybrid Score > τ."
        )