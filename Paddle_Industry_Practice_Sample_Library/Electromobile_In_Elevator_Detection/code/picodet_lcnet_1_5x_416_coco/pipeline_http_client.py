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
# from paddle_serving_server.pipeline import PipelineClient
import numpy as np
import requests
import json
import cv2
import base64
import os
from time import *
import threading


def demo(url,data,i):
    begin_time = time()
    r = requests.post(url=url, data=json.dumps(data))
    end_time = time()
    run_time = end_time-begin_time
    print ('线程 %d 时间  %f '%(i,run_time))
    print(r.json())


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

url = "http://127.0.0.1:2009/ppyolo_mbv3/prediction"
with open(os.path.join(".", "test.jpg"), 'rb') as file:
    image_data1 = file.read()
image = cv2_to_base64(image_data1)
category_dict={0.0:"person",1.0:"bicycle",2.0:"motorcycle"}
data = {"key": ["image"], "value": [image]}
r = requests.post(url=url, data=json.dumps(data))
print(r.json())
'''
results = eval(r.json()['value'][0])
img = cv2.imread("test.jpg")
for result in results:
    if result["score"] > 0.5:
        left, right, top, bottom= int(result['bbox'][0]), int(result['bbox'][2]), int(result['bbox'][1]), int(result['bbox'][3])
        cv2.rectangle(img,(left ,top),(right,bottom), (0, 0, 255), 2)
        cv2.putText(img,str(round(result["score"],2)),(left,top-10), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
        print(category_dict[result["category_id"]])
        cv2.putText(img,category_dict[result["category_id"]],(left,top+20), cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
cv2.imwrite("./result.jpg",img)
'''
    
        


