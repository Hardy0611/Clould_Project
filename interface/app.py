import streamlit as st
import os
st.title("Image Upload")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)