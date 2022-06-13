# coding:utf-8
from paddle.v2.plot import Ploter
import sys
import paddle.v2 as paddle
from PIL import Image
import numpy as np
import os
from vgg import vgg_bn_drop
 
class TestCIFAR:
    # ***********************��ʼ������***************************************
    def __init__(self):
        # ��ʼ��paddpaddle,ֻ����CPU,��GPU�ر�
        paddle.init(use_gpu=False, trainer_count=2)
 
    # **********************��ȡ����***************************************
    def get_parameters(self, parameters_path):
        with open(parameters_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
        return parameters
 
    # ***********************ʹ��ѵ���õĲ�������Ԥ��***************************************
    def to_prediction(self, image_path, parameters, out):
        # ��ȡͼƬ
        def load_image(file):
            im = Image.open(file)
            im = im.resize((32, 32), Image.ANTIALIAS)
            im = np.array(im).astype(np.float32)
            # PIL��ͼƬ�洢˳��ΪH(�߶�)��W(���)��C(ͨ��)��
            # PaddlePaddleҪ������˳��ΪCHW��������Ҫת��˳��
            im = im.transpose((2, 0, 1))
            # CIFARѵ��ͼƬͨ��˳��ΪB(��),G(��),R(��),
            # ��PIL��ͼƬĬ��ͨ��˳��ΪRGB,��Ϊ��Ҫ����ͨ����
            im = im[(2, 1, 0), :, :]  # BGR
            im = im.flatten()
            im = im / 255.0
            return im
 
        # ���ҪԤ���ͼƬ
        test_data = []
        test_data.append((load_image(image_path),))
 
        # ���Ԥ����
        probs = paddle.infer(output_layer=out,
                             parameters=parameters,
                             input=test_data)
        # ����Ԥ����
        lab = np.argsort(-probs)
        # ���ظ�������ֵ�����Ӧ�ĸ���ֵ
        return lab[0][0], probs[0][(lab[0][0])]
 
 
if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # ��ʼԤ��
    out = vgg_bn_drop(3 * 32 * 32)
    parameters = testCIFAR.get_parameters("model/modelCifar.tar")
    image_path = "Cifar/horse2.jpg"
    result,probability = testCIFAR.to_prediction(image_path=image_path, out=out, parameters=parameters)
    print 'Ԥ����Ϊ:%d,���Ŷ�Ϊ:%f' % (result,probability)