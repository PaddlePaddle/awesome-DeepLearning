# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from utils import logger
from utils import config
from utils.predictor import Predictor
from utils.get_image_list import get_image_list
from det_preprocess import det_preprocess
from preprocess import create_operators
from utils.draw_bbox import draw_bbox_results

import os
import argparse
import time
import yaml
import ast
from functools import reduce
import cv2
import numpy as np
import paddle
import requests
import base64
import json
class DetPredictor(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"],
                         config["Global"]["det_inference_model_dir"])

        self.preprocess_ops = create_operators(config["DetPreProcess"][
            "transform_ops"])
        self.config = config

    def preprocess(self, img):
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': np.array(
                img.shape[:2], dtype=np.float32),
            'input_shape': self.config["Global"]["image_shape"],
            "scale_factor": np.array(
                [1., 1.], dtype=np.float32)
        }
        
        im, im_info = det_preprocess(img, im_info, self.preprocess_ops)
        print(111111111111111111111)
        print(im)
        inputs = self.create_inputs(im, im_info)
        return inputs

    def create_inputs(self, im, im_info):
        """generate input for different model type
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
            model_arch (str): model type
        Returns:
            inputs (dict): input of model
        """
        inputs = {}
        inputs['image'] = np.array((im, )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info['scale_factor'], )).astype('float32')
        #print(inputs)
        return inputs

    def parse_det_results(self, pred, threshold, label_list):
        max_det_results = self.config["Global"]["max_det_results"]
        keep_indexes = pred[:, 1].argsort()[::-1][:max_det_results]
        results = []
        for idx in keep_indexes:
            single_res = pred[idx]
            class_id = int(single_res[0])
            score = single_res[1]
            bbox = single_res[2:]
            if score < threshold:
                continue
            label_name = label_list[class_id]
            '''
            results.append({
                "class_id": class_id,
                "score": score,
                "bbox": bbox,
                "label_name": label_name,
            })'''
            results.append({
                "bbox": bbox,
                "rec_docs": "background",
                "rec_scores": score,
            })
        return results

    def predict(self, image, threshold=0.5, run_benchmark=False):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2ps
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        inputs = self.preprocess(image)
        print(str(inputs))
        np_boxes = None
        input_names = self.paddle_predictor.get_input_names()
        print(input_names)
        for i in range(len(input_names)):
            input_tensor = self.paddle_predictor.get_input_handle(input_names[
                i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
            print("!!!!!!!",inputs[input_names[i]])
        t1 = time.time()
        print(self.paddle_predictor.run())
        output_names = self.paddle_predictor.get_output_names()
        boxes_tensor = self.paddle_predictor.get_output_handle(output_names[0])

        np_boxes = boxes_tensor.copy_to_cpu()
        t2 = time.time()

        print("Inference: {} ms per batch image".format((t2 - t1) * 1000.0))

        # do not perform postprocess in benchmark mode
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = np.array([])
        else:
            results = np_boxes

        results = self.parse_det_results(results,
                                         self.config["Global"]["threshold"],
                                         self.config["Global"]["labe_list"])
        return results


def main(config):
    det_predictor = DetPredictor(config)
    image_list = get_image_list(config["Global"]["infer_imgs"])

    assert config["Global"]["batch_size"] == 1
    for idx, image_file in enumerate(image_list):
        img = cv2.imread(image_file)[:, :, ::-1]
        output = det_predictor.predict(img)
        print(output)
        draw_bbox_results(img, output, image_file)

    return image_file,output

def cv2_to_base64_img(img):
    data = cv2.imencode('.jpg', img)[1]
    return base64.b64encode(data.tostring()).decode('utf8')
    
def solve_output(output,image_file):
    print(image_file)
    img = cv2.imread(image_file)

    for bbox in output:
        left,top,right,bottom = int(bbox["bbox"][0]),int(bbox["bbox"][1]),int(bbox["bbox"][2]),int(bbox["bbox"][3])
        print(left,top,right,bottom)
        img_crop = img[top:bottom,left:right]
        url = "http://123.157.241.94:36807/ppyolo_mbv3/prediction"
        img2 = {"key": ["image"], "value": [cv2_to_base64_img(img_crop)]}
        r = requests.post(url=url,data=json.dumps(img2), timeout=5)
        r = r.json()
        print(r)
        result = eval(r['value'][0])[0]
        cv2.putText(img,str(round(float(result["scores"][0]),2)),(left,top+30), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
        cv2.putText(img,str(result["label_names"][0]),(left,top+60), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
        cv2.rectangle(img,(left ,top),(right,bottom), (0, 0, 255), 2)    
    cv2.imwrite("./output/ppyolo_result" + image_file[image_file.rfind("/"):],img) 
if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=True)
    image_file,output = main(config)
    #solve_output(output,image_file)
    
