import random
import os
import xml.etree.ElementTree as ET

#生成train.txt和val.txt
random.seed(2020)
data_root_dir = '/home/aistudio/work/fall'

path_list = list()
labels = []#['down','person']#设置你想检测的类别

print("数据总数：",len(os.listdir(data_root_dir))/2)

for img in os.listdir(data_root_dir):
    if not img.endswith(".jpg"):
        continue

    img_path = os.path.join(data_root_dir,img)
    xml_path = os.path.join(data_root_dir,img.replace('jpg', 'xml'))

    # 读取xml获取标签
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 有些数据标注有问题，如图像大小为空0，过滤掉
    size=root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    if width==0 or height==0:
        continue
    
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in labels:
            labels.append(cls_name) 

    path_list.append((img_path, xml_path))

print("有效数据个数：",len(path_list))
random.shuffle(path_list)
ratio = 0.9
train_f = open('/home/aistudio/work/train.txt','w') #生成训练文件
val_f = open('/home/aistudio/work/val.txt' ,'w')#生成验证文件

for i ,content in enumerate(path_list):
    img, xml = content
    text = img + ' ' + xml + '\n'

    if i < len(path_list) * ratio:
        train_f.write(text)
    else:
        val_f.write(text)

train_f.close()
val_f.close()

#生成标签文档

print(labels)

with open('/home/aistudio/work/label_list.txt', 'w') as f:
    for text in labels:
        f.write(text+'\n')