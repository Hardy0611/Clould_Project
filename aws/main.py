from typing import List
from lib import get_model
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def run_model(event, context):
    
    try:
        input: List[List[List[List[float]]]] = event['input']
        stage: int = event['stage']
        
        if stage < 0 or stage > 5:
            logging.error(f"Passed stage of {stage}")
            return {"response": 400, "error": "Invalid stage"}
        
        if type(input) != list:
            logging.error(f"Passed input of {type(input)}")
            return {"response": 400, "error": "Invalid input"}
        
        array: np.ndarray = np.array(input)
        
        model = get_model(stage)
        
        if model.inputs[0].shape[1:] != array.shape[1:]:
            logging.error(f"Passed shape of {array.shape} for model with shape {model.inputs[0].shape}")
            return {"response": 400, "error": "Invalid input shape"}
        
        prediction = model.predict(array)
        
        return {"response": 200, "prediction": prediction.tolist()}
    
    except Exception as e:
        logging.error(f"Error: {e}")
        # return {"response": 500, "error": "Internal server error"}
        raise e
    
    
    