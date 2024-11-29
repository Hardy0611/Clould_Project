from typing import List
from package import get_model
import logging
import numpy as np
import ast
import json
import gc

logger = logging.getLogger()
logger.setLevel(logging.INFO)

models_cache = {}
LABELS = ['Angry', 'Disgusted', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def run_model(event, context):
    
    try:
        logging.info(f"Function started")
        if 'body' not in event.keys():
            logging.error(f"Missing body")
            return {"response": 400, "error": "Missing body"}
        
        event = json.loads(event['body'])
        
        if 'input' not in event.keys() or 'stage' not in event.keys():
            logging.error(f"Missing input or stage")
            return {"response": 400, "error": "Missing input or stage"}
        
        logging.info(f"Parsing inputs")
        
        input: List[List[List[List[float]]]] = ast.literal_eval(event['input'])
        stage: int = int(event['stage'])
        
        logging.info(f"Inputs parsed")
        
        if stage < 0 or stage > 5:
            logging.error(f"Passed stage of {stage}")
            return {"response": 400, "error": "Invalid stage"}
        
        if type(input) != list:
            logging.error(f"Passed input of {type(input)}")
            return {"response": 400, "error": "Invalid input"}
        
        array: np.ndarray | None = np.array(input)
        
        logging.info(f"Getting model")
        
        if stage not in models_cache:
            models_cache[stage] = get_model(stage)
        
        model = models_cache[stage]
        
        if model.inputs[0].shape[1:] != array.shape[1:]:
            logging.error(f"Passed shape of {array.shape} for model with shape {model.inputs[0].shape}")
            return {"response": 400, "error": "Invalid input shape"}
        
        logging.info(f"Predicting")
        
        prediction = model.predict(array)
        
        # clear memory
        event.clear()
        del input
        array = None
        gc.collect()
        
        return {"response": 200, "predictions": str(prediction.tolist()), 'labels': LABELS, "emotion": LABELS[np.argmax(prediction)]}
    
    except Exception as e:
        logging.error(f"Error: {e}")
        return {"response": 500, "error": "Internal server error"}

    
    
    
