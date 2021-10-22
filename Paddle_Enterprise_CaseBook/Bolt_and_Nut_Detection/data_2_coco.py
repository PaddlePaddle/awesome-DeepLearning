import os,sys
import json
import cv2
import glob
from PIL import Image
import numpy as np
import pandas as pd
import shutil

DATA_DIR = "/home/aistudio/work/dataset/coco"

class Transfer2COCO:

    def __init__(self,is_mode='train'):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.is_mode = is_mode
        if not os.path.exists(DATA_DIR+"/{}".format(self.is_mode)):
            os.makedirs(DATA_DIR+"/{}".format(self.is_mode))

    def to_coco(self, anno_file,img_dir,label_list_path):
        self._init_categories(label_list_path)

        with open(anno_file,'r') as f:
            anno_result= f.readlines()
        
        for item in anno_result:
            items = item.strip().split('\t')
            
            image_file = items[0]
            image_file_path = os.path.join(img_dir,image_file)

            bboxs=[]
            detect_labels=[]
            for anno in items[1:]:
                if len(anno.strip())<1:
                    continue

                object = json.loads(anno.strip())
                detect_name = object['value']
                detect_label = self.name_dict[detect_name]
                coord = object['coordinate']#[[435.478,126.261],[697.043,382.261]]
                box = []
                box.append(coord[0][0])
                box.append(coord[0][1])
                box.append(coord[1][0])
                box.append(coord[1][1])

                bboxs.append(box)
                detect_labels.append(detect_label)

            #这种读取方法更快
            img = Image.open(image_file_path)
            w, h = img.size
            self.images.append(self._image(image_file_path,h, w))

            self._cp_img(image_file_path)#复制文件路径
            if self.img_id % 200 is 0:
                print("处理到第{}张图片".format(self.img_id))
            for bbox, label in zip(bboxs, detect_labels):
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1

        instance = {}
        instance['info'] = 'bolt and nut defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self,label_list_path):
        with open(label_list_path,'r') as f:
            lines = f.readlines()
        self.name_dict={}
        for line in lines:
            items=line.strip().split(' ')
            category = {}
            category['id'] = items[0]
            category['name'] = items[1]
            self.name_dict[items[1]]=items[0]
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)#返回path最后的文件名
        return image

    def _annotation(self,label,bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = []# np.asarray(points).flatten().tolist()
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation["ignore"] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join(DATA_DIR+"/{}".format(self.is_mode), os.path.basename(img_path)))
    
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))#缩进设置为1，元素之间用逗号隔开 ， key和内容之间 用冒号隔开

transfer = Transfer2COCO()
#训练集
img_dir = "/home/aistudio/data/data6045"
anno_dir="/home/aistudio/data/data6045/train.txt"
label_list_file = '/home/aistudio/data/data6045/label_list.txt'

train_instance = transfer.to_coco(anno_dir,img_dir,label_list_file)
if not os.path.exists(DATA_DIR+"/annotations/"):
    os.makedirs(DATA_DIR+"/annotations/")
transfer.save_coco_json(train_instance, DATA_DIR+"/annotations/"+'instances_{}.json'.format("train"))

transfer = Transfer2COCO(is_mode='eval')
#验证集
img_dir = "/home/aistudio/data/data6045"
anno_dir="/home/aistudio/data/data6045/eval.txt"
label_list_file = '/home/aistudio/data/data6045/label_list.txt'

train_instance = transfer.to_coco(anno_dir,img_dir,label_list_file)
if not os.path.exists(DATA_DIR+"/annotations/"):
    os.makedirs(DATA_DIR+"/annotations/")
transfer.save_coco_json(train_instance, DATA_DIR+"/annotations/"+'instances_{}.json'.format("eval"))
