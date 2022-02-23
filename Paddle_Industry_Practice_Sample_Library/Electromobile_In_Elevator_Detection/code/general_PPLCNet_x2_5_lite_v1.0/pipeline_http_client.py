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

url = "http://127.0.0.1:9315/recognition/prediction"
with open(os.path.join(".", "test.jpg"), 'rb') as file:
    image_data1 = file.read()
image = cv2_to_base64(image_data1)

for i in range(1):
    print(i)
    data = {"key": ["image"], "value": [image]}
    r = requests.post(url=url, data=json.dumps(data))
    print(r.json())
    #t = threading.Thread(target=demo, args=(url,data,i,))
    #t.start()




