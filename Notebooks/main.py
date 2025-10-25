import streamlit as st
import googleNet
import yolo

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["App blood cells", "App brain cells"])

if page == "App blood cells":
    googleNet.app()
elif page == "App brain cells":
    yolo.app()