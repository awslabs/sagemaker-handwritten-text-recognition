import mxnet as mx
import cv2
import numpy as np
from skimage.draw import line_aa

def crop_image(image, bb):
    ''' Helper function to crop the image by the bounding box (in percentages)
    '''
    (x, y, w, h) = bb
    x = x * image.shape[1]
    y = y * image.shape[0]
    w = w * image.shape[1]
    h = h * image.shape[0]
    (x1, y1, x2, y2) = (x, y, x + w, y + h)
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    return image[y1:y2, x1:x2]

def resize_image(image, desired_size, allignment="centre"):
    ''' Helper function to resize an image while keeping the aspect ratio.
    Parameter
    ---------

    image: np.array
        The image to be resized.

    desired_size: (int, int)
        The (height, width) of the resized image

    Return
    ------

    image: np.array
        The image of size = desired_size

    border_bb: (int, int, int, int)
        (x, y, w, h) in percentages of the borders added into the image to keep the aspect ratio
    '''
    original_image_size = image.shape[:2]
    size = image.shape[:2]

    if size[0] > desired_size[0] or size[1] > desired_size[1]:
        ratio_w = float(desired_size[0]) / size[0]
        ratio_h = float(desired_size[1]) / size[1]
        ratio = min(ratio_w, ratio_h)
        new_size = tuple([int(x * ratio) for x in size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        size = image.shape

    delta_w = max(0, desired_size[1] - size[1])
    delta_h = max(0, desired_size[0] - size[0])
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)

    if allignment == "centre":
        left, right = delta_w // 2, delta_w - (delta_w // 2)
    if allignment == "left":
        left = 5
        right = delta_w - 5

    color = image[0][0]
    if color < 230:
        color = 230
    bordered_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=float(color))
    border_bb = [left, top,
                 bordered_image.shape[1] - right - left,
                 bordered_image.shape[0] - top - bottom]

    border_bb = [border_bb[0] / bordered_image.shape[1],
                 border_bb[1] / bordered_image.shape[0],
                 border_bb[2] / bordered_image.shape[1],
                 border_bb[3] / bordered_image.shape[0]]

    bordered_image[bordered_image > 230] = 255
    return bordered_image, border_bb

def decode(prediction):
    '''
    Returns the string given one-hot encoded vectors.
    '''
    alphabet_encoding = r'0123456789abcdefghijklmnopqrstuvwxyz'
    alphabet_dict = {alphabet_encoding[i]:i for i in range(len(alphabet_encoding))}

    results = []
    for word in prediction:
        result = []
        for i, index in enumerate(word):
            if i < len(word) - 1 and word[i] == word[i+1] and word[-1] != -1: #Hack to decode label as well
                continue
            if index == len(alphabet_dict) or index == -1:
                continue
            else:
                result.append(alphabet_encoding[int(index)])
        results.append(result)
    words = [''.join(word) for word in results]
    return words

def handwriting_recognition_transform(image, line_image_size):
    '''
    Resize and normalise the image to be fed into the network.
    '''
    image, _ = resize_image(image, line_image_size)

    image = mx.nd.array(image)/255.
    image = (image - 0.942532484060557) / 0.15926149044640417
    image = image.expand_dims(0).expand_dims(0)
    return image

def transform_bb_after_resize(bb, border_bb, input_image_size, resized_image_size):
    (x1, y1, x2, y2) = (bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3])

    border_bb = [border_bb[0] * resized_image_size[1],
                border_bb[1] * resized_image_size[0],
                border_bb[2] * resized_image_size[1],
                border_bb[3] * resized_image_size[0]]

    (x1, y1, x2, y2) = (x1 * border_bb[2],
                        y1 * border_bb[3],
                        x2 * border_bb[2],
                        y2 * border_bb[3])

    x_offset = border_bb[0]
    y_offset = border_bb[1]

    new_x1 = (x1 + x_offset) / resized_image_size[1]
    new_y1 = (y1 + y_offset) / resized_image_size[0]
    new_x2 = (x2 + x_offset) / resized_image_size[1]
    new_y2 = (y2 + y_offset) / resized_image_size[0]

    new_bbs = np.zeros(shape=bb.shape)
    new_bbs[:, 0] = new_x1
    new_bbs[:, 1] = new_y1
    new_bbs[:, 2] = new_x2
    new_bbs[:, 3] = new_y2

    return new_bbs

def transform_bb_after_cropping(bb, crop_bb, input_image_size, cropped_image_size):
    (x1, y1, x2, y2) = (bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3])
    (x1, y1, x2, y2) = (
        x1 * input_image_size[1], y1 * input_image_size[0],
        x2 * input_image_size[1], y2 * input_image_size[0])

    new_x1 = (x1 - crop_bb[0] * input_image_size[1]) / cropped_image_size[1]
    new_y1 = (y1 - crop_bb[1] * input_image_size[0]) / cropped_image_size[0]
    new_x2 = (x2 - crop_bb[0] * input_image_size[1]) / cropped_image_size[1]
    new_y2 = (y2 - crop_bb[1] * input_image_size[0]) / cropped_image_size[0]

    new_bbs = np.zeros(shape=bb.shape)
    new_bbs[:, 0] = new_x1
    new_bbs[:, 1] = new_y1
    new_bbs[:, 2] = new_x2
    new_bbs[:, 3] = new_y2

    return new_bbs

def draw_text_on_image(images, text):
    output_image_shape = (images.shape[0], images.shape[1], images.shape[2] * 2, images.shape[3])  # Double the output_image_shape to print the text in the bottom
    
    output_images = np.zeros(shape=output_image_shape)
    for i in range(images.shape[0]):
        white_image_shape = (images.shape[2], images.shape[3])
        white_image = np.ones(shape=white_image_shape)*1.0
        text_image = cv2.putText(white_image, text[i], org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=0.0, thickness=1)
        output_images[i, :, :images.shape[2], :] = images[i]
        output_images[i, :, images.shape[2]:, :] = text_image
    return output_images

def draw_line(image, y1, x1, y2, x2, line_type):
    rr, cc, val = line_aa(y1, x1, y2, x2)
    if line_type == "dotted":
        rr = np.delete(rr, np.arange(0, rr.size, 5))
        cc = np.delete(cc, np.arange(0, cc.size, 5))
    image[rr, cc] = 0
    return image
    
def draw_box(bounding_box, image, line_type, is_xywh=True):
    image_h, image_w = image.shape[-2:]
    if is_xywh:
        (x, y, w, h) = bounding_box
        (x1, y1, x2, y2) = (x, y, x + w, y + h)
    else:
        (x1, y1, x2, y2) = bounding_box
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    if y2 >= image_h:
        y2 = image_h - 1
    if x2 >= image_w:
        x2 = image_w - 1
    if y1 >= image_h:
        y1 = image_h - 1
    if x1 >= image_w:
        x1 = image_w - 1
    if y2 < 0:
        y2 = 0
    if x2 < 0:
        x2 =0
    if y1 < 0:
        y1 = 0
    if x1 < 0:
        x1 = 0

    image = draw_line(image, y1, x1, y2, x1, line_type)
    image = draw_line(image, y2, x1, y2, x2, line_type)
    image = draw_line(image, y2, x2, y1, x2, line_type)
    image = draw_line(image, y1, x2, y1, x1, line_type)
    return image

def draw_boxes_on_image(pred, label, images):
    ''' Function to draw multiple bounding boxes on the images. Predicted bounding boxes will be
    presented with a dotted line and actual boxes are presented with a solid line.
    Parameters
    ----------
    
    pred: [n x [x, y, w, h]]
        The predicted bounding boxes in percentages. 
        n is the number of bounding boxes predicted on an image
    label: [n x [x, y, w, h]]
        The actual bounding boxes in percentages
        n is the number of bounding boxes predicted on an image
    images: [[np.array]]
        The correponding images.
    Returns
    -------
    images: [[np.array]]
        Images with bounding boxes printed on them.
    '''
    image_h, image_w = images.shape[-2:]
    label[:, :, 0], label[:, :, 1] = label[:, :, 0] * image_w, label[:, :, 1] * image_h
    label[:, :, 2], label[:, :, 3] = label[:, :, 2] * image_w, label[:, :, 3] * image_h
    for i in range(len(pred)):
        pred_b = pred[i]
        pred_b[:, 0], pred_b[:, 1] = pred_b[:, 0] * image_w, pred_b[:, 1] * image_h
        pred_b[:, 2], pred_b[:, 3] = pred_b[:, 2] * image_w, pred_b[:, 3] * image_h

        image = images[i, 0]
        for j in range(pred_b.shape[0]):
            image = draw_box(pred_b[j, :], image, line_type="dotted")

        for k in range(label.shape[1]):
            image = draw_box(label[i, k, :], image, line_type="solid")
        images[i, 0, :, :] = image
    return images

def draw_box_on_image(pred, label, images):
    ''' Function to draw bounding boxes on the images. Predicted bounding boxes will be
    presented with a dotted line and actual boxes are presented with a solid line.
    Parameters
    ----------
    
    pred: [[x, y, w, h]]
        The predicted bounding boxes in percentages
    label: [[x, y, w, h]]
        The actual bounding boxes in percentages
    images: [[np.array]]
        The correponding images.
    Returns
    -------
    images: [[np.array]]
        Images with bounding boxes printed on them.
    '''

    image_h, image_w = images.shape[-2:]
    pred[:, 0], pred[:, 1] = pred[:, 0] * image_w, pred[:, 1] * image_h
    pred[:, 2], pred[:, 3] = pred[:, 2] * image_w, pred[:, 3] * image_h

    label[:, 0], label[:, 1] = label[:, 0] * image_w, label[:, 1] * image_h
    label[:, 2], label[:, 3] = label[:, 2] * image_w, label[:, 3] * image_h

    for i in range(images.shape[0]):
        image = images[i, 0]
        image = draw_box(pred[i, :], image, line_type="dotted")
        image = draw_box(label[i, :], image, line_type="solid")
        images[i, 0, :, :] = image
    return images