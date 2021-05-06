# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
'''
 画图展示如何绘制边界框和锚框
'''
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import math


# 定义画矩形框的程序
def draw_rectangle(currentAxis,
                   bbox,
                   edgecolor='k',
                   facecolor='y',
                   fill=False,
                   linestyle='-'):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0] + 1,
        bbox[3] - bbox[1] + 1,
        linewidth=1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        fill=fill,
        linestyle=linestyle)
    currentAxis.add_patch(rect)


# 绘制锚框
def draw_anchor_box(center, length, scales, ratios, img_height, img_width):
    """
    以center为中心，产生一系列锚框
    其中length指定了一个基准的长度
    scales是包含多种尺寸比例的list
    ratios是包含多种长宽比的list
    img_height和img_width是图片的尺寸，生成的锚框范围不能超出图片尺寸之外
    """
    bboxes = []
    for scale in scales:
        for ratio in ratios:
            h = length * scale * math.sqrt(ratio)
            w = length * scale / math.sqrt(ratio)
            x1 = max(center[0] - w / 2., 0.)
            y1 = max(center[1] - h / 2., 0.)
            x2 = min(center[0] + w / 2. - 1.0, img_width - 1.0)
            y2 = min(center[1] + h / 2. - 1.0, img_height - 1.0)
            print(center[0], center[1], w, h)
            bboxes.append([x1, y1, x2, y2])

    for bbox in bboxes:
        draw_rectangle(currentAxis, bbox, edgecolor='b')


plt.figure(figsize=(10, 10))

filename = '000000086956.jpg'
im = imread(filename)
plt.imshow(im)

# 使用xyxy格式表示物体真实框
bbox1 = [214.29, 325.03, 399.82, 631.37]
bbox2 = [40.93, 141.1, 226.99, 515.73]
bbox3 = [247.2, 131.62, 480.0, 639.32]

currentAxis = plt.gca()

draw_rectangle(currentAxis, bbox1, edgecolor='r')
draw_rectangle(currentAxis, bbox2, edgecolor='r')
draw_rectangle(currentAxis, bbox3, edgecolor='r')

filename = '000000086956.jpg'
im = imread(filename)
img_height = im.shape[0]
img_width = im.shape[1]
draw_anchor_box([300., 500.], 100., [2.0], [0.5, 1.0, 2.0], img_height,
                img_width)

################# 以下为添加文字说明和箭头###############################

plt.text(285, 285, 'G1', color='red', fontsize=20)
plt.arrow(
    300,
    288,
    30,
    40,
    color='red',
    width=0.001,
    length_includes_head=True,
    head_width=5,
    head_length=10,
    shape='full')

plt.text(190, 320, 'A1', color='blue', fontsize=20)
plt.arrow(
    200,
    320,
    30,
    40,
    color='blue',
    width=0.001,
    length_includes_head=True,
    head_width=5,
    head_length=10,
    shape='full')

plt.text(160, 370, 'A2', color='blue', fontsize=20)
plt.arrow(
    170,
    370,
    30,
    40,
    color='blue',
    width=0.001,
    length_includes_head=True,
    head_width=5,
    head_length=10,
    shape='full')

plt.text(115, 420, 'A3', color='blue', fontsize=20)
plt.arrow(
    127,
    420,
    30,
    40,
    color='blue',
    width=0.001,
    length_includes_head=True,
    head_width=5,
    head_length=10,
    shape='full')

# draw_anchor_box([200., 200.], 100., [2.0], [0.5, 1.0, 2.0])
plt.show()
