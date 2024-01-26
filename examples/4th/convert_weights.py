from __future__ import division, print_function
from network import Yolo
import tensorflow as tf
import numpy as np

ANCHORS = np.asarray([[[12, 16], [19, 36], [40, 28]],
                                   [[36, 75], [76, 55], [72, 146]],
                                   [[142, 110], [192, 243], [459, 401]]])


def convert_weight(weights_file_path, ckpt_file_path):
    yolo = Yolo(num_classes=80, anchors=ANCHORS)
    with tf.Session() as sess:
        inputs = tf.placeholder(tf.float32, [1, 608, 608, 3], name='inputs')
        _, _, _ = yolo.inference(inputs, istraining=False)
        var_list = tf.global_variables()
        saver = tf.train.Saver(var_list=var_list)

        with open(weights_file_path, "rb") as fp:
            np.fromfile(fp, dtype=np.int32, count=5)
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0  # weights权重指针
        i = 0  # TensorFlow图指针
        assign_ops = []  # 保存的节点列表
        while i < len(var_list) - 1:
            var1 = var_list[i]
            var2 = var_list[i + 1]
            print(var1.name)
            # 仅当当前层为卷积层进行操作
            if 'Conv' in var1.name.split('/')[-2]:

                # 若下一层是BatchNorm层
                if 'BatchNorm' in var2.name.split('/')[-2]:
                    gamma, beta, mean, var = var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        # 加载权重并移动ptr指针
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                        ptr += num_params
                    # 加载了4个参数
                    i += 4

                # 若下一层也是卷积层
                elif 'Conv' in var2.name.split('/')[-2]:
                    shape = var2.shape.as_list()
                    num_params = np.prod(shape)
                    # 加载权重并移动ptr指针
                    bias_weights = weights[ptr:ptr + num_params].reshape(shape)
                    assign_ops.append(tf.assign(var2, bias_weights, validate_shape=True))
                    ptr += num_params
                    # 加载了1个参数
                    i += 1

                # 处理当前层
                shape = var1.shape.as_list()
                num_params = np.prod(shape)
                # 加载权重并移动ptr指针
                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                ptr += num_params
                i += 1

        sess.run(assign_ops)
        saver.save(sess, save_path=ckpt_file_path)
        print('TensorFlow model checkpoint has been saved to {}'.format(ckpt_file_path))


if __name__ == '__main__':
    convert_weight('./weights/yolov4.weights', './coco_models/model')
