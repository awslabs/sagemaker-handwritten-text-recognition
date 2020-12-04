import os
import mxnet as mx
import numpy as np

try:
    from src.utils import resize_image, handwriting_recognition_transform
    from src.utils import decode as decoded_characters
except ModuleNotFoundError:
    from utils import resize_image, handwriting_recognition_transform
    from utils import decode as decoded_characters

def load_net(directory, ctx, path_filter, num_symbols=None):    
    if num_symbols is None:
        symbol_path = os.path.join(directory, path_filter+"-symbol.json")
        params_path = os.path.join(directory, path_filter+"-0000.params")
        net = mx.gluon.nn.SymbolBlock.imports(symbol_path, ['data'], 
                                            params_path, ctx=ctx)
        return [net]
    else:
        nets = []
        for i in range(num_symbols):
            symbol_path = os.path.join(directory, path_filter+"{}-symbol.json".format(i))
            params_path = os.path.join(directory, path_filter+"{}-0000.params".format(i))
            net = mx.gluon.nn.SymbolBlock.imports(symbol_path, ['data'],
                                                  params_path, ctx=ctx)
            nets.append(net)
        return nets
        
def load_nets(out_directory, ctx):
    body_net = load_net(out_directory, ctx, path_filter="ws_body")
    downsamples_net = load_net(out_directory, ctx, path_filter="ws_downsamples", num_symbols=3)
    class_preds_net = load_net(out_directory, ctx, path_filter="ws_class_preds", num_symbols=6)
    box_preds_net = load_net(out_directory, ctx, path_filter="ws_box_preds", num_symbols=6)
    htr_net = load_net(out_directory, ctx, path_filter="htr")
    
    return {"body": body_net, 
            "downsamples": downsamples_net,
            "class_preds": class_preds_net,
            "box_preds": box_preds_net,
            "htr": htr_net
           }

def _flatten_prediction(pred):
    '''
    Helper function to flatten the predicted bounding boxes and categories
    '''
    return mx.nd.flatten(mx.nd.transpose(pred, axes=(0, 2, 3, 1)))

def preprocess_image(img, ctx):
    '''
    Converts the image into the correct size for the word segmentation network.
    Parameters:
    ----------
    Image: np.array
        Gray scale image containing handwritten text. At least 700x700
    
    ctx: mxnet context
    
    Returns:
    --------
    image: nd.array
        Resized and normalised image.
    '''
    image = mx.nd.array(img).expand_dims(axis=2)
    image = mx.image.resize_short(image, 350)
    image = image.transpose([2, 0, 1])/255.

    image = image.as_in_context(ctx)
    image = image.expand_dims(0)
    return image

def run_word_segmentation_net(nets, input_image, ctx, min_c=0.1, overlap_thres=0.1, topk=600):
    x = nets["body"][0](input_image)
    
    anchor_sizes = [[.1, .2], [.2, .3], [.2, .4], [.3, .4], [.3, .5], [.4, .6]]
    anchor_ratios = [[1, 3, 5], [1, 3, 5], [1, 6, 8], [1, 4, 7], [1, 6, 8], [1, 5, 7]]

    num_anchors = len(anchor_sizes)
    num_classes = 2
    
    # Run SSD
    default_anchors = []
    predicted_boxes = []
    predicted_classes = []

    for i in range(num_anchors):
        default_anchors.append(mx.contrib.ndarray.MultiBoxPrior(x, sizes=anchor_sizes[i], ratios=anchor_ratios[i]))
        predicted_boxes.append(_flatten_prediction(nets["box_preds"][i](x)))
        predicted_classes.append(_flatten_prediction(nets["class_preds"][i](x)))
        if i < len(nets["downsamples"]):
            x = nets["downsamples"][i](x)
        elif i == 3:
            x = mx.nd.Pooling(x, global_pool=True, pool_type='max', kernel=(4, 4))
    
    anchors = mx.nd.concat(*default_anchors, dim=1)
    box_preds = mx.nd.concat(*predicted_boxes, dim=1)
    class_preds = mx.nd.concat(*predicted_classes, dim=1)
    class_preds = mx.nd.reshape(class_preds, shape=(0, -1, num_classes + 1))
        
    # Do SSD prediction
    bb = np.zeros(shape=(13, 5))
    bb = mx.nd.array(bb)
    bb = bb.as_in_context(ctx)
    bb = bb.expand_dims(axis=0)
    
    class_preds = mx.nd.transpose(class_preds, axes=(0, 2, 1))
    box_target, box_mask, cls_target = mx.contrib.ndarray.MultiBoxTarget(
        anchors, bb, class_preds)

    cls_probs = mx.nd.SoftmaxActivation(class_preds, mode='channel')
    predicted_bb = mx.contrib.ndarray.MultiBoxDetection(*[cls_probs, box_preds, anchors], force_suppress=True, clip=False)
    predicted_bb = mx.contrib.ndarray.box_nms(predicted_bb, overlap_thresh=overlap_thres, valid_thresh=min_c, topk=topk)
    predicted_bb = predicted_bb.asnumpy()
    predicted_bb = predicted_bb[0, predicted_bb[0, :, 0] != -1]
    predicted_bb = predicted_bb[:, 2:]
    predicted_bb[:, 2] = predicted_bb[:, 2] - predicted_bb[:, 0]
    predicted_bb[:, 3] = predicted_bb[:, 3] - predicted_bb[:, 1]

    return predicted_bb

def _clip_value(value, max_value):
    '''
    Helper function to make sure that "value" will not be greater than max_value
    or lower than 0.
    '''
    output = value
    if output < 0:
        output = 0
    if output > max_value:
        output = max_value
    return int(output)

def crop_images(image, bbs):
    word_images = []
    for bb in bbs:
        (x, y, w, h) = bb
        image_h, image_w = image.shape[-2:]
        (x, y, w, h) = (x * image_w, y * image_h, w * image_w, h * image_h)
        x1 = _clip_value(x, max_value=image_w)
        x2 = _clip_value(x + w, max_value=image_w)
        y1 = _clip_value(y, max_value=image_h)
        y2 = _clip_value(y + h, max_value=image_h)

        word_image = image[y1:y2, x1:x2]    
        if word_image.shape[0] > 0 and word_image.shape[1] > 0:
            word_images.append(word_image)
    return word_images

def run_handwriting_recognition(nets, word_images, ctx):
    word_image_size = (60, 175)
    decoded = []
    for word_image in word_images:
        word_image = handwriting_recognition_transform(word_image, word_image_size)
        word_character_probs = nets["htr"][0](word_image.as_in_context(ctx))
        arg_max = word_character_probs.topk(axis=2).asnumpy()
        decoded.append(decoded_characters(arg_max)[0])
    return decoded

def run_pipeline(nets, resized_img, ctx, expand_bb=1.0):
    preprocessed_img = preprocess_image(resized_img, ctx)
    bbs = run_word_segmentation_net(nets, preprocessed_img, ctx)

    bb = bbs.copy()
    new_w = (1 + expand_bb) * bb[:, 2]
    new_h = (1 + expand_bb) * bb[:, 3]
    
    bb[:, 0] = bb[:, 0] - (new_w - bb[:, 2])/2
    bb[:, 1] = bb[:, 1] - (new_h - bb[:, 3])/2
    bb[:, 2] = new_w
    bb[:, 3] = new_h
    bbs = bb 
    
    word_images = crop_images(resized_img, bbs)

    decoded = run_handwriting_recognition(nets, word_images, ctx)
    
    return decoded, bbs