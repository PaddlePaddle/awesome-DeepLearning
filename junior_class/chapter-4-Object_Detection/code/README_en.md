# Use AI to identify insects [[简体中文](./README.md)]

## Dependent packages
- os
- numpy
- xml
- opencv
- PIL
- random
- paddlepaddle==2.0.0
- time
- json

## Structure
'''
|-data: store dataset
|-datasets: store scripts for data preprocessing and data reading
    |-transform.py: data preprocessing script
    |-dataset.py: script to read data
|-Detection_basis: implementation script of target detection basic module
    |-box_iou_xywh.py: script to calculate iou using ‘xywh’ format
    |-box_iou_xyxy.py: script to calculate iou using ‘xyxy’ format
    |-draw_anchor_box.py: script to draw anchor box
    |-draw_rectangle.py: script for drawing the box
    |-mAP.py: mAP calculation script
    |-multiclass_nms.py: Multi-category nms script
    |-nms.py: single category nms script
|-net: store network definition scripts
    |-YOLOv3.py: The script defines the network structure of YOLOv3 and the loss function definition
|-train.py: start training script
|-predict.py: script to infer all validation set images
|-predict_one_pic.py: script to reason about a picture

'''

## Dataset preparation
1. Download the [dataset](https://aistudio.baidu.com/aistudio/datasetdetail/19638) to the data directory
2. Unzip the dataset
‘’‘
cd data
unzip -q insects.zip
’‘’

## Train
Start the training directly using the train.py script.
'''
python3 train.py
'''

## Infer
1. Use all images in the validation set for inference
'''
python3 predict.py
'''
2. Use one picture for inference
'''
predict_one_pic.py
'''
