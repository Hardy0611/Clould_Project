from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input
import os

def get_path(file: str) -> str:
    return os.path.join(os.path.dirname(__file__), file)

def check_models_loaded() -> bool:
    for i in range(6):
        if not os.path.exists(get_path(f"models/face_model_{i}.h5")):
            return False

    return True

def generate_model() -> None:
    
    if check_models_loaded():
        return
    
    if not os.path.exists(get_path("models/face_model.h5")):
        print(get_path("models/face_model.h5"))
        raise FileNotFoundError("face_model.h5 not found")
    
    model = load_model(get_path("models/face_model.h5"))
    splits = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'dense']
    models = [model]
    
    for split in splits:
        seq_model = Sequential()
        layer = model.get_layer(split)
        split_model = Model(layer.input, model.outputs)
        for layer in split_model.layers:
            seq_model.add(layer)
        
        models.append(seq_model) 
    
    for i, model in enumerate(models):
        model.save(get_path(f"models/face_model_{i}.h5"))

def get_model(index) -> Model:
    return load_model(get_path(f"models/face_model_{index}.h5"))

if __name__ == "__main__":
    generate_model()