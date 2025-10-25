import streamlit as st
import googleNet
import yolo
import documentation

st.set_page_config(page_title="Medical Image Analysis", layout="wide", page_icon="🏥")

st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to:", ["📚 Documentation", "🩸 Blood Cell Classification", "🧠 Brain Tumor Detection"])

if page == "📚 Documentation":
    documentation.app()
elif page == "🩸 Blood Cell Classification":
    googleNet.app()
elif page == "🧠 Brain Tumor Detection":
    yolo.app()