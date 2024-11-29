import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import psutil 
import numpy as np 
from PIL import Image
import io

def split_model(model, select_layer):
    split_layer = model.get_layer(select_layer)
    model1 = Model(inputs=model.layers[0].input, outputs=split_layer.output)

    model2_input = split_layer.output
    model2_layers = model.layers[model.layers.index(split_layer) + 1:]
    x = model2_input
    for layer in model2_layers:
        x = layer(x)

    model2 = Model(inputs=model2_input, outputs=x)

    return model1, model2

def dynamic_allocate():
    model = load_model("/Users/tszhofan/Documents/Coding/Clould_Project/data/model/cnn_emotion_detection.h5")
    layers = ["dropout", "dropout_1", "dropout_2"]
    cpu_usage = psutil.cpu_percent()
    spilt = 99
    if cpu_usage == 0:
        select_layer = layers[2]
        spilt = 2
    elif cpu_usage < 10 and cpu_usage > 0: 
        select_layer = layers[1]
        spilt = 1
    else: 
        select_layer = layers[0]
        spilt = 0

    model1, model2 = split_model(model, select_layer)
    return model1, model2, spilt

def run_model(model, img_bytes): 
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1).astype('float32')
    result = model.predict(img_array)
    return result

def get_label(emotions_pred): 
    emotions = ['fearful','disgust','surprise','happy', 'neutral','sad', 'angry']
    idx = np.argmax(emotions_pred)
    return emotions[idx]