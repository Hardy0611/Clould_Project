import streamlit as st
import sys
import os
import requests, json
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controllers.allocation_controller import dynamic_allocate, run_model

url = "https://nxrqlgpnzzzh3vgkxhvy37asde0atgvb.lambda-url.us-east-1.on.aws/"

st.title("Image Upload")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
    model1, stage = dynamic_allocate()
    result = run_model(model1, bytes_data)
    data = {"input": str(result.tolist()), "stage": stage}

    # send to AWS lambda function
    header = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=header)
    # print(response.text)
    emotion = response.json()['emotion']
    print(emotion)
    st.title(emotion)