"""
Halaman Explainability — Mengapa sebuah flow diflag sebagai anomaly?
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline import NIDSPipeline

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Explainability: Mengapa Diflag?")

# ============================================================
# LOAD PIPELINE
# ============================================================
@st.cache_resource
def load_pipeline():
    return NIDSPipeline(artifacts_dir="artifacts", device="auto")

pipe = load_pipeline()

# ============================================================
# INPUT SAMPEL
# ============================================================
st.subheader("1. Masukkan Sampel NetFlow")
st.caption("Masukkan 39 nilai fitur NetFlow (dipisah koma) atau gunakan contoh.")

# Contoh data dummy (bisa diganti dengan sampel asli dari dataset Anda)
# Untuk demo, kita pakai nilai acak yang di-scale [0,1]
default_vals = ",".join([f"{np.random.rand():.4f}" for _ in range(len(pipe.feature_columns))])

sample_input = st.text_area(
    "Paste 39 nilai fitur (dipisah koma):",
    value=default_vals,
    height=80,
)

if st.button("🔍 Analisis Sampel", type="primary"):
    try:
        vals = [float(v.strip()) for v in sample_input.split(",")]
        if len(vals) != len(pipe.feature_columns):
            st.error(f"Jumlah fitur harus {len(pipe.feature_columns)}, didapat {len(vals)}")
            st.stop()
        
        X_raw = np.array([vals], dtype=np.float32)
        result = pipe.predict(X_raw)

        # ============================================================
        # VISUALISASI SKOR
        # ============================================================
        st.subheader("2. Breakdown Skor Anomali")
        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["c_sae"][0],
                title={"text": "SAE Rank (CDF)"},
                gauge={"axis": {"range": [0, 1]}, "bar": {"color": "blue"}},
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["c_if"][0],
                title={"text": "IF(Z) Rank (CDF)"},
                gauge={"axis": {"range": [0, 1]}, "bar": {"color": "orange"}},
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result["H"][0],
                title={"text": "Hybrid Score (Max-Rank)"},
                delta={"reference": pipe.tau},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "green"},
                    "steps": [
                        {"range": [0, pipe.tau], "color": "lightgreen"},
                        {"range": [pipe.tau, 1], "color": "red"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": pipe.tau,
                    },
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # VERDICT
        # ============================================================
        if result["verdict"][0] == 1:
            st.error(f"🚨 TERDETEKSI SEBAGAI ZERO-DAY ATTACK (Score {result['H'][0]:.4f} > τ {pipe.tau:.4f})")
        else:
            st.success(f"✅ Trafik Normal (Score {result['H'][0]:.4f} ≤ τ {pipe.tau:.4f})")

        st.divider()

        # ============================================================
        # TOP-K FITUR PENYUMBANG ERROR (XAI)
        # ============================================================
        st.subheader("3. Top-10 Fitur dengan Reconstruction Error Tertinggi")
        st.caption("Fitur ini paling sulit direkonstruksi oleh SAE, menandakan anomali pada dimensi tersebut.")

        # Hitung MSE per fitur
        X_scaled = pipe.minmax_transform(X_raw)
        Z = pipe.sae.get_latent_features(X_scaled)
        
        with torch.no_grad():
            Z_tensor = torch.from_numpy(Z).to(pipe.device)
            X_recon_tensor = pipe.sae.decoder(Z_tensor)
            X_scaled_tensor = torch.from_numpy(X_scaled).to(pipe.device)
            
            # .detach() memastikan tensor tidak membutuhkan gradient
            mse_per_feature = torch.mean(
                (X_scaled_tensor - X_recon_tensor) ** 2, dim=0
            ).detach().cpu().numpy()
            
        # Ambil top 10
        top_idx = np.argsort(mse_per_feature)[::-1][:10]
        top_features = [pipe.feature_columns[i] for i in top_idx]
        top_errors = mse_per_feature[top_idx]

        fig_bar = go.Figure(go.Bar(
            x=top_errors,
            y=top_features,
            orientation='h',
            marker_color='crimson',
            text=[f"{v:.6f}" for v in top_errors],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title="Kontribusi Fitur terhadap Anomali",
            xaxis_title="Mean Squared Error (MSE)",
            yaxis_title="Nama Fitur",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")