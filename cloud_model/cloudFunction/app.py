import json
from allocation_controller import run_model
import numpy as np

def lambda_handler(event, context):
    data = json.loads(event['body'])
    input_data = data['result']
    input_data = np.array(input_data)
    spilt = data['case']
    result = run_model(input_data, spilt)
    result = result.tolist()

    return {
        "statusCode": 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',  # Allow all origins
            'Access-Control-Allow-Methods': 'OPTIONS, POST, GET',  # Allowed methods
            'Access-Control-Allow-Headers': 'Content-Type',  # Allowed headers
        },
        "body": json.dumps({
            "message": result
        }),
    }
