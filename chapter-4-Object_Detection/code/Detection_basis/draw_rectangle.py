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
