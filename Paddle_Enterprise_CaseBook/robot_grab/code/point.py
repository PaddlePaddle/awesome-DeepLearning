import cv2
import numpy as np
from PIL import Image

img = cv2.imread('xiaoduxiong_ins_det/JPEGImages/Xiaoduxiong114.jpeg')
result = model.predict(
    'xiaoduxiong_ins_det/JPEGImages/Xiaoduxiong114.jpeg',
    transforms=model.test_transforms)

mask_edge_points = parse_mask_edge_points(result)
img = cv2.drawContours(img, mask_edge_points[0], 0, (0, 0, 255), 3)

cv2.imwrite('./test.png', img)
# Image.fromarray(img.astype('uint8'))
