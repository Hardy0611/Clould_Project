import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import psutil 

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

def dynamic_allocate(spilt):
    model = load_model("/Users/tszhofan/Documents/Coding/Clould_Project/data/model/cnn_emotion_detection.h5")
    layers = ["dropout", "dropout_1", "dropout_2"]
    cpu_usage = psutil.cpu_percent()
    if spilt == 2:
        select_layer = layers[2]
    elif spilt == 1: 
        select_layer = layers[1]
    elif spilt == 0: 
        select_layer = layers[0]
    else:
        return None, None, None

    model1, model2 = split_model(model, select_layer)
    return model2

def run_model(input_data, spilt): 
    model = dynamic_allocate(spilt)
    result = model.predict(input_data)
    return result