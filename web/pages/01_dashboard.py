"""
Halaman Dashboard — Ringkasan model, metrik, dan pipeline
"""
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")
st.title(" Dashboard Utama")

# ============================================================
# KPI CARDS
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("F1-Score (P97)", "0.936", "+0.027 vs IF(Z)")
col2.metric("ROC-AUC", "0.983", "+0.022 vs IF(Z)")
col3.metric("FPR", "2.93%", "-2.24% vs IF(Z)")
col4.metric("Recall", "90.50%", "+2.95% vs IF(Z)")

st.divider()

# ============================================================
# PIPELINE DIAGRAM
# ============================================================
st.subheader("🔄 Alur Pipeline Deteksi")

cols = st.columns(7)
with cols[0]:
    st.info("**1. Input**\n39 fitur NetFlow mentah")
with cols[1]:
    st.info("**2. Scaling**\nMinMax [0, 1]")
with cols[2]:
    st.info("**3. SAE Encoder**\nZ ∈ ℝ¹⁶")
with cols[3]:
    st.info("**4. Dual Score**\nMSE + IF(Z)")
with cols[4]:
    st.info("**5. CDF Rank**\nNormalisasi rank")
with cols[5]:
    st.info("**6. Fusion**\nMax-Rank")
with cols[6]:
    st.success("**7. Decision**\nτ = P97")

st.divider()

# ============================================================
# METRIK PER KATEGORI ATTACK
# ============================================================
st.subheader("🎯 Recall per Kategori Attack (Threshold P97)")

import pandas as pd
breakdown_data = {
    "Attack": ["Bot", "Brute Force", "DDoS", "DoS", "Reconnaissance"],
    "Recall (%)": [100.00, 87.62, 87.96, 87.64, 89.30],
    "F1-Score": [0.9318, 0.8664, 0.8683, 0.8665, 0.8757],
    "FPR (%)": [2.93, 2.93, 2.93, 2.93, 2.93],
}
df_breakdown = pd.DataFrame(breakdown_data)

fig = go.Figure(
    go.Bar(
        x=df_breakdown["Attack"],
        y=df_breakdown["Recall (%)"],
        text=[f"{v}%" for v in df_breakdown["Recall (%)"]],
        textposition="outside",
        marker_color=["#2ecc71" if v >= 90 else "#f39c12" for v in df_breakdown["Recall (%)"]],
    )
)
fig.update_layout(
    title="Recall per Kategori Zero-Day Attack",
    yaxis_title="Recall (%)",
    yaxis_range=[0, 110],
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_breakdown, use_container_width=True)

st.divider()

# ============================================================
# INFO MODEL
# ============================================================
st.subheader(" Informasi Model")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
        **Arsitektur:**
        - SAE: 39 → 64 → 32 → **16** → 32 → 64 → 39
        - IForest: 100 trees, dilatih pada latent Z (16-dim)
        - Fusion: Max-Rank pada skala CDF
        - Threshold: Conformal P97 (τ = 0.9832)
        """
    )
with col_b:
    st.markdown(
        """
        **Dataset:**
        - NF-UQ-NIDS-v2 (NetFlow)
        - 39 extended NetFlow features
        - Zero-day: Bot, Brute Force, DDoS, DoS, Reconnaissance
        - Train: hanya benign | Test: benign + unseen attacks
        """
    )