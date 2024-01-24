#!/usr/bin/python3
#coding=utf-8
import os
import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
import dataset
import warnings
warnings.filterwarnings('ignore')
'''
这是生成预测图片的代码
'''


class Test(object):
    def __init__(self, Dataset, datapath, Network, model_path):
        self.datapath = datapath.split("/")[-1]
        self.datapath2 = datapath
        print(datapath)
        self.cfg = Dataset.Config(datapath=datapath, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(
            self.data,
            batch_size=bs,
            shuffle=True,
            num_workers=8,
            use_shared_memory=False)
        # network
        self.net = Network
        self.net.eval()
        self.net.load_dict(paddle.load(model_path))

    # 读取原图
    def read_img(self, path):
        gt_img = self.norm_img(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        gt_img = (gt_img >= 0.5).astype(np.float32)
        return gt_img

    # 归一化原图
    def norm_img(self, im):
        return cv2.normalize(im.astype('float'),
                             None,
                             0.0, 1.0,
                             cv2.NORM_MINMAX)

    # 保存预测图片
    def save(self, img_path, save_path=None):
        with paddle.no_grad():

            for num, (image, mask, (H, W), maskpath) in enumerate(self.loader):
                out = self.net(image)
                pred = F.sigmoid(out)

                k_pred = pred
                for num in range(len(H)):
                    mae_pred = k_pred[num].unsqueeze(0)
                    path = img_path + '/mask/' + maskpath[num] + '.png'
                    mae_mask = paddle.to_tensor(self.read_img(path)).unsqueeze(0).unsqueeze(0)
                    mae_pred = F.interpolate(mae_pred, size=mae_mask.shape[2:], mode='bilinear')

                    if save_path:
                        save_paths = os.path.join(save_path, self.cfg.datapath.split('/')[-1])
                        if not os.path.exists(save_paths):
                            os.makedirs(save_paths)
                        mae_pred = mae_pred[0].transpose((1, 2, 0)) * 255
                        cv2.imwrite(save_paths + '/' + maskpath[num], mae_pred.cpu().numpy())


if __name__=='__main__':
    from models import Res2NetandACFFNet
    from models import Res2NetandFMFNet
    from models import ResNeXtandACFFNet
    from models import SwinTandACFFNet

    bs = 2
    model_list = ['this is the path of the pre-trained model']
    model = Res2NetandACFFNet()
    img_path = 'write your path of test image'
    save_path = 'write the path where you want to save the test mask'
    test = Test(dataset, img_path, model, model_list)
    test.save(img_path, save_path)








