import json

def lambda_handler(event, context):
    output = {
        "taskInput": event['dataObject']
    }
    return output
