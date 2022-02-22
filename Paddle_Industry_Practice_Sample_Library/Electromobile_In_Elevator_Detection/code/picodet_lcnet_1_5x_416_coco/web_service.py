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
from paddle_serving_server.web_service import WebService, Op
import logging
import numpy as np
import sys
import cv2
from paddle_serving_app.reader import *
import base64
import os
import yaml
import glob
from picodet_postprocess import PicoDetPostProcess
from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine

class PPYoloMbvOp(Op):
    def init_op(self):
        self.feed_dict={}
        deploy_file = 'infer_cfg.yml'
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        preprocess_infos = yml_conf['Preprocess']
        self.preprocess_ops = []
        for op_info in preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            self.preprocess_ops.append(eval(op_type)(**new_op_info))
        #print(self.preprocess_ops)
        
    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        imgs = []
        for key in input_dict.keys():
            data = base64.b64decode(input_dict[key].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_info = {
                'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
                'im_shape': None,
            }
            im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
            im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
            for operator in self.preprocess_ops:
                im, im_info = operator(im, im_info)
            imgs.append({
              "image": im[np.newaxis,:],
              "im_shape": [im_info['im_shape']],#np.array(list(im.shape[1:])).reshape(-1)[np.newaxis,:],
              "scale_factor": [im_info['scale_factor']],#np.array([im_scale_y, im_scale_x]).astype('float32'),
            })
        self.feed_dict = {
            "image": np.concatenate([x["image"] for x in imgs], axis=0),
            "im_shape": np.concatenate([x["im_shape"] for x in imgs], axis=0),
            "scale_factor": np.concatenate([x["scale_factor"] for x in imgs], axis=0)
        }
        #print(self.feed_dict)
        #for key in self.feed_dict.keys():
        # print(key, self.feed_dict[key].shape)
        
        return self.feed_dict, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, log_id,data_id =0):
        #print(fetch_dict)
        np_score_list = []
        np_boxes_list = []
        i = 0
        for value in fetch_dict.values():#range(4):
            if i<4:
                np_score_list.append(value)
            else:
                np_boxes_list.append(value) 
            i=i+1
 
        post_process = PicoDetPostProcess(
            (640,640),
            self.feed_dict['im_shape'],
            self.feed_dict['scale_factor'],
            [8, 16, 32, 64],
            0.5)
        res_dict = {}
        np_boxes, np_boxes_num = post_process(np_score_list, np_boxes_list)
        if len(np_boxes) == 0:
            return res_dict, None, ""
        
        d = []
        for b in range(np_boxes.shape[0]):
            c = {}
            #print(b)
            c["category_id"] = np_boxes[b][0]
            c["bbox"] = [np_boxes[b][2],np_boxes[b][3],np_boxes[b][4],np_boxes[b][5]]
            c["score"] = np_boxes[b][1]
            d.append(c)
        res_dict["bbox_result"] = str(d)
        #fetch_dict["image"] = "234.png"
        #res_dict = {"bbox_result": str(self.img_postprocess(fetch_dict, visualize=False))}
        return res_dict, None, ""


class PPYoloMbv(WebService):
    def get_pipeline_response(self, read_op):
        ppyolo_mbv3_op = PPYoloMbvOp(name="ppyolo_mbv3", input_ops=[read_op])
        return ppyolo_mbv3_op


ppyolo_mbv3_service = PPYoloMbv(name="ppyolo_mbv3")
ppyolo_mbv3_service.prepare_pipeline_config("config.yml")
ppyolo_mbv3_service.run_service()
