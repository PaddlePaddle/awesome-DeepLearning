import os
import numpy as np
import cv2

from .transform_img import transform_img

def data_loader(datadir,batch_size=10,mode='train'):
    filenames=os.listdir(datadir)
    def reader():
        if mode =='train':
            np.random.shuffle(filenames)
        batch_imgs=[]
        batch_labels=[]
        for name in filenames:
            filepath=os.path.join(datadir,name)
            img=cv2.imread(filepath)
            img=transform_img(img)
            if name[0]=='H' or name[0]=='N':
                label=0
            elif name[0]=='P':
                label=1
            elif name[0]=='V':
                continue
            else:
                raise('Not excepted file name')
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs)==batch_size:
                imgs_array=np.array(batch_imgs).astype('float32')
                labels_array=np.array(batch_labels).astype('float32').reshape(-1,1)
                yield imgs_array,labels_array
                batch_imgs=[]
                batch_labels=[]
        if len(batch_imgs)>0:
            imgs_array=np.array(batch_imgs).astype('float32')
            labels_array=np.array(batch_labels).astype('float32').reshape(-1,1)
            yield imgs_array,labels_array
    return reader