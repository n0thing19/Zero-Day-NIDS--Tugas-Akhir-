"""
Zero-Day NIDS Simulator — Streamlit Multi-Page App
Entry point. Jalankan: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Zero-Day NIDS Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ Zero-Day Attack Detection Simulator")
st.markdown(
    """
    **Hybrid Stacked Autoencoder + Isolation Forest** dengan 
    **Max-Rank Fusion** untuk deteksi *zero-day attack* pada NetFlow.
    
    Silakan pilih halaman di sidebar:
    - 🏠 **Dashboard** — Ringkasan model, metrik, dan pipeline
    - 🧪 **Simulasi** — Upload file atau input manual untuk deteksi
    """
)

st.info(
    " Gunakan sidebar di sebelah kiri untuk navigasi antar halaman."
)