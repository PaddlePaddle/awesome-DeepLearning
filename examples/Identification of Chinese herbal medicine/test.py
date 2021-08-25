# coding=utf-8
import os
import zipfile
from PIL import Image
import time
import paddle
from dataloader import dataset
from model import VGGNet
import matplotlib.pyplot as plt
import numpy as np

def unzip_infer_data(src_path,target_path):
    '''
    解压预测数据集
    '''
    if(not os.path.isdir(target_path + "Chinese Medicine Infer")):     
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


def load_image(img_path):
    '''
    预测图片预处理
    '''
    img = Image.open(img_path) 
    if img.mode != 'RGB': 
        img = img.convert('RGB') 
    img = img.resize((224, 224), Image.BILINEAR)
    img = np.array(img).astype('float32') 
    img = img.transpose((2, 0, 1)) / 255 # HWC to CHW 及归一化
    return img


infer_src_path = '/home/aistudio/data/data105575/Chinese Medicine Infer.zip'
infer_dst_path = '/home/aistudio/data/'
unzip_infer_data(infer_src_path,infer_dst_path)

label_dic = {}
with open('/home/aistudio/label_dict.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    num, name = line.split()
    label_dic[num] = name

# 加载训练过程保存的最后一个模型
model__state_dict = paddle.load('work/checkpoints/save_dir_final.pdparams')
model_predict = VGGNet(num_classes=5)
model_predict.set_state_dict(model__state_dict) 
model_predict.eval()
infer_imgs_path = os.listdir(infer_dst_path+"Chinese Medicine Infer")
# print(infer_imgs_path)

# 预测所有图片
for infer_img_path in infer_imgs_path:
    infer_img = load_image(infer_dst_path+"Chinese Medicine Infer/"+infer_img_path)
    infer_img = infer_img[np.newaxis,:, : ,:]  #reshape(-1,3,224,224)
    infer_img = paddle.to_tensor(infer_img)
    result = model_predict(infer_img)
    lab = np.argmax(result.numpy())
    print("样本: {},被预测为:{}".format(infer_img_path,label_dic[str(lab)]))
    # 显示图片
    #img = Image.open(infer_dst_path+"Chinese Medicine Infer/"+infer_img_path)
    #plt.imshow(img)
    #plt.axis('off')
    #plt.show()
    #sys.stdout.flush()
    #time.sleep(0.5)

