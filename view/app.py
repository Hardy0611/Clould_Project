import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controllers.allocation_controller import dynamic_allocate, run_model, simulate_cloud_model

st.title("Image Upload")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
    model1, model2, select_layer = dynamic_allocate()
    result = run_model(model1, bytes_data)
    emotion = simulate_cloud_model(model2, result)
    st.title(emotion)