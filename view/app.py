import streamlit as st
import sys
import os
import requests, json
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controllers.allocation_controller import dynamic_allocate, run_model, get_label

url = "https://fsiz1djvx9.execute-api.ap-southeast-1.amazonaws.com/Prod/send"

st.title("Image Upload")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
    model1, model2, spilt = dynamic_allocate()
    result = run_model(model1, bytes_data)
    data = {"result": result.tolist(), "case": spilt}

    # send to AWS lambda function
    response = requests.post(url, data=json.dumps(data))
    print(response.text)
    print(response.status_code)
    response_data = json.loads(response.text)
    response_data = np.array(response_data.get("message"))

    emotion = get_label(response_data)
    st.title(emotion)