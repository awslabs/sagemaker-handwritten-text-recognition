import os
import json
import numpy as np
import mxnet as mx

from utils import resize_image
from run_inference import load_nets, run_pipeline

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu()

def model_fn(model_dir):    
    nets = load_nets(os.path.join(model_dir, "model"), ctx)
    return nets

def transform_fn(model, request_body, content_type, accept_type):
    request_data = json.loads(request_body)
        
    resized_img = np.array(request_data)
    decoded, line_bbs = run_pipeline(model, resized_img, ctx)
    
    response_body = {"decoded": decoded,
                    "line_bbs": line_bbs.tolist()
                    }
                              
    return response_body, content_type