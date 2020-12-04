import numpy as np
from mxnet.gluon.data import dataset
import json
import os
import cv2

try:
    from src.utils import resize_image, transform_bb_after_resize, transform_bb_after_cropping
except ModuleNotFoundError:
    from utils import resize_image, transform_bb_after_resize, transform_bb_after_cropping

class HandwritingRecognitionDataset(dataset.ArrayDataset):
    PAGE_SIZE = (700, 700)
    LINE_SIZE = (60, 800)
    WORD_SIZE = (60, 175)

    def __init__(self, dir_path, path, annotation_filename, output_type, transform=None):
        assert output_type in ["page", "line", "word"], "output_type can only be 'page', 'line', or 'word'."
        
        self.images_path = os.path.join(dir_path, path)
        self.annotations = self._read_annotations(os.path.join(dir_path, path, annotation_filename),
                                                  output_type)
        self.output_type = output_type
        self.transform = transform
        super(HandwritingRecognitionDataset, self).__init__(self.annotations)
                    
    def __getitem__(self, index):
        item = self.annotations[index]
        
        im = cv2.imread(os.path.join(self.images_path, item['filename']),
                        cv2.IMREAD_GRAYSCALE)
        original_image_size = im.shape
        resized_image, border_bb = resize_image(im, desired_size=self.PAGE_SIZE)
        resized_image_size = resized_image.shape
           
        annotations = item['annotation']
        texts = []
        bbs = []
        for annotation in annotations:
            texts.append(annotation['text'])
            bbs.append(annotation['bb'])
        bbs = np.array(bbs).astype(float)
        if len(bbs.shape) == 3:
            bbs = bbs[0]
        bbs = self._normalise_bb(bbs, original_image_size)
                
        if self.output_type == "line":
            transformed_bb = transform_bb_after_resize(
                bbs, border_bb, original_image_size, resized_image_size)

            line_bb = np.expand_dims(annotations[0]['line_bb'], 0).astype(float)
            line_bb = self._normalise_bb(line_bb, original_image_size) 
            
            transformed_line_bb = transform_bb_after_resize(
                line_bb, border_bb, original_image_size, resized_image_size)
            resized_image, transformed_bb, texts = self._crop_image(
                resized_image, transformed_line_bb, transformed_bb, texts, self.LINE_SIZE)
        
        elif self.output_type == "word":
            transformed_bb = transform_bb_after_resize(
                bbs, border_bb, original_image_size, resized_image_size)

            word_bb = np.expand_dims(annotations[0]['bb'], 0).astype(float)
            word_bb = self._normalise_bb(word_bb, original_image_size) 
        
            transformed_word_bb = transform_bb_after_resize(
                word_bb, border_bb, original_image_size, resized_image_size)

            resized_image, _, texts = self._crop_image(
                resized_image, transformed_word_bb, transformed_bb, [texts], self.WORD_SIZE)
            
            # Word output_type has no bounding boxes
            transformed_bb = np.array([[0, 0, 1, 1]])

        elif self.output_type == "page":

            transformed_bb = transform_bb_after_resize(
                bbs, border_bb, original_image_size, resized_image_size)
        
        if self.transform is not None:
            return self.transform(resized_image, transformed_bb, texts)
        else:
            return self.transform(resized_image, transformed_bb, texts)
    
    def _crop_image(self, resized_image, crop_bb, bb, texts, desired_size):
        assert crop_bb.shape[0] == 1, "There should be only 1 bounding boxes for output line mode"
        x1, y1, x2, y2 = crop_bb[0]
        resized_image_shape = resized_image.shape
        x1 = int(x1 * resized_image_shape[1])
        y1 = int(y1 * resized_image_shape[0])
        x2 = int(x2 * resized_image_shape[1])
        y2 = int(y2 * resized_image_shape[0])

        cropped_image = resized_image[y1:y2, x1:x2]
        cropped_bb = transform_bb_after_cropping(
            bb, crop_bb[0], resized_image.shape, cropped_image.shape)
        
        resized_cropped_image, resized_cropped_bb = resize_image(
            cropped_image, desired_size=desired_size, allignment="left")
        transformed_line_bb = transform_bb_after_resize(
                cropped_bb, resized_cropped_bb, cropped_image.shape, resized_cropped_image.shape)

        return resized_cropped_image, transformed_line_bb, texts[0]
        
    def _read_annotations(self, annotation_path, output_type):
        '''
        Given an annotation path
        '''
        annotations = []
        with open(annotation_path, "r") as w:
            lines = w.readlines()
            for line in lines:
                if len(line) <= 1:
                    continue
                annotation_dict = json.loads(line) #json.loads(line[:-1])
                if output_type == "line":
                    line_annotations = self._read_line_annotation(annotation_dict)
                    for line_annotation in line_annotations:
                        annotations.append(line_annotation)
                elif output_type == "word":
                    word_annotations = self._read_word_annotations(annotation_dict)
                    for word_annotation in word_annotations:
                        annotations.append(word_annotation)
                elif output_type == "page":
                    annotation = self._read_page_annotation(annotation_dict)
                    annotations.append(annotation)
        return annotations
                    
    def _read_line_annotation(self, annotation_dict):
        '''
        Extract the relevant information from the annotation (dict) of the output.manifest. 
        Then convert the bb into lines

        Parameter:
        ----------
        annotation_dict: {}
            line from the output.manifest
            
        Return:
        -------
        line_annotations: [{[]}]
            formatted information.
            Note that bbs are converted from polygons to rectangles.
        '''
        page_annotation = self._read_page_annotation(annotation_dict)
        line_annotation_dict = {}
        for annotation in page_annotation['annotation']:
            line_num = annotation['line_num']
            if line_num not in line_annotation_dict:
                line_annotation_dict[line_num] = []
            line_annotation_dict[line_num].append(annotation) 
                        
        line_annotations = []
        # Sort annotation by line
        for line_num in line_annotation_dict:
            tmp = line_annotation_dict[line_num]
            bb_list = [a['bb'] for a in tmp]
            texts = [a['text'] for a in tmp]
            bb_list, texts = self._sort_texts_on_bb_x(bb_list, texts)
            bb = self._convert_bb_list_to_max_bb(bb_list)
            line_annotations.append({
                "filename": page_annotation["filename"],
                "annotation": [{
                    "text": texts,
                    "line_bb": bb,
                    "bb": bb_list
                }]
            })
            
        return line_annotations
    
    def _read_word_annotations(self, annotation_dict):
        '''
        Extract the relevant information from the annotation (dict) of the output.manifest. 
        Then convert the bb into words

        Parameter:
        ----------
        annotation_dict: {}
            line from the output.manifest
            
        Return:
        -------
        word_annotations: [{[]}]
            formatted information.
            Note that bbs are converted from polygons to rectangles.
        '''
        page_annotation = self._read_page_annotation(annotation_dict)
        word_annotations = []
        for annotation in page_annotation["annotation"]:
            word_annotations.append({
                "filename": page_annotation["filename"],
                "annotation": [{
                    "text": annotation["text"],
                    "bb": annotation['bb']
                }]
            })
            
        return word_annotations
        
    def _read_page_annotation(self, annotation_dict):
        '''
        Extract the relevant information from the annotation (dict) of the output.manifest.

        Parameter:
        ----------
        annotation_dict: {}
            line from the output.manifest
            
        Return:
        -------
        out: {[]}
            formatted information.
            Note that bbs are converted from polygons to rectangles.
        '''
        filename = os.path.basename(annotation_dict["source-ref"])
        out = {"filename": filename}
        
        annotation_list = []
        for annotation in annotation_dict["annotations"]["texts"]:
            tmp = annotation
            tmp["bb"] = self._convert_polygon_to_rects(annotation["bb"])
            annotation_list.append(tmp)
        out["annotation"] = annotation_list
        return out
    
    def _sort_texts_on_bb_x(self, bb_list, texts):
        # Sort positions of texts based on the x position of bb_lists
        assert len(bb_list) == len(texts), "No. bb_list are not identical to No. texts"
        
        bb_x = [a[0] for a in bb_list]
        sorted_bb_x = np.argsort(bb_x)

        return np.array(bb_list)[sorted_bb_x].tolist(), np.array(texts)[sorted_bb_x].tolist(),
  
    def _convert_polygon_to_rects(self, polygon_bb):
        assert len(polygon_bb)==4, "your bounding box should only have 4 coords"
        
        x_sorted = sorted(polygon_bb, key=lambda i: i['x'])
        x_max, x_min = x_sorted[-1]['x'], x_sorted[0]['x']
        
        y_sorted = sorted(polygon_bb, key=lambda i: i['y'])
        y_max, y_min = y_sorted[-1]['y'], y_sorted[0]['y']

        return (x_min, y_min, x_max, y_max)
    
    def _convert_bb_list_to_max_bb(self, bb_list):
        '''
        Helper function to convert a list of bbs into one bb that encompasses all
        the bbs.
        BBs are in the form (x1, y1, x2, y2)
        '''
        max_x = np.max([a[2] for a in bb_list])
        min_x = np.min([a[0] for a in bb_list])
        
        max_y = np.max([a[3] for a in bb_list])
        min_y = np.min([a[1] for a in bb_list])
        
        return (min_x, min_y, max_x, max_y)
    
    def _normalise_bb(self, bbs, image_size):
        '''
        Normalise bbs from absolute lengths to percentages
        '''
        bbs[:, 0] = bbs[:, 0]/image_size[1]
        bbs[:, 1] = bbs[:, 1]/image_size[0]
        bbs[:, 2] = bbs[:, 2]/image_size[1]
        bbs[:, 3] = bbs[:, 3]/image_size[0]
        return bbs
