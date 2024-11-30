import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import psutil 
import numpy as np 
from PIL import Image
import io

def split_model(model, select_layer):
    split_layer = model.get_layer(select_layer).input
    model1 = Model(inputs=model.layers[0].input, outputs=split_layer)

    return model1

def dynamic_allocate():
    model = load_model("./data/model/face_model.h5")
    layers = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
    cpu_usage = psutil.cpu_percent()
    stage = 99
    if cpu_usage == 0:
        select_layer = layers[3]
        stage = 3
    elif cpu_usage < 70 and cpu_usage > 0: 
        select_layer = layers[2]
        stage = 2
    elif cpu_usage >= 70 and cpu_usage < 95: 
        select_layer = layers[1]
        stage = 1
    else:
        select_layer = layers[0]
        stage = 0

    model1 = split_model(model, select_layer)
    return model1, stage

def run_model(model, img_bytes): 
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1).astype('float32')
    result = model.predict(img_array)
    return result