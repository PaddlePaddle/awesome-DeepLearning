import glob
import numpy as np
import threading
import time
import random
import os
import base64
import cv2
import json
import paddlex as pdx

image_name = 'dataset/JPEGImages/Image_20210615204210757.bmp'
model = pdx.load_model('output/mask_rcnn_r50_fpn/best_model')

img = cv2.imread(image_name)
result = model.predict(img)

keep_results = []
areas = []
f = open('result.txt', 'a')
count = 0
for dt in np.array(result):
    cname, bbox, score = dt['category'], dt['bbox'], dt['score']
    if score < 0.5:
        continue
    keep_results.append(dt)
    count += 1
    f.write(str(dt) + '\n')
    f.write('\n')
    areas.append(bbox[2] * bbox[3])
areas = np.asarray(areas)
sorted_idxs = np.argsort(-areas).tolist()
keep_results = [keep_results[k]
                for k in sorted_idxs] if len(keep_results) > 0 else []
print(keep_results)
print(count)
f.write("the total number is :" + str(int(count)))
f.close()
pdx.det.visualize(
    image_name, result, threshold=0.5, save_dir='./output/mask_rcnn_r50_fpn')
