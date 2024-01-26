from data_voc import Data
from network import Yolo
import tensorflow as tf
import numpy as np
import os


LR_TYPE = 'constant'
LR_INIT = 2e-4
LR_LOWER = 1e-6
PIECEWISE_DOUNDARIES = [1, 2]
PIECEWISE_VALUES = [2e-4, 1e-4, 1e-4]

OPTIMIZER_TYPE = 'momentum'
MOMENTUM = 0.949


ANCHORS = np.asarray([[[10, 13], [16, 30], [32, 23]],
                      [[30, 61], [62, 45], [59, 119]],
                      [[116, 90], [156, 198], [373, 326]]])

DATABASE_ROOT_DIR = './VOCdevkit'
DATABASE_NAMES = ['VOC2007', 'VOC2012']
PRED_NAMES = './data/coco.names'
NUM_CLASSES = 20
MODEL_SAVE_PATH = './voc_models'
MODEL_SAVE_NAME = 'model'

MULTI_SCALE_IMG = False
BATCH_SIZE = 32
NUM_EPOCHS = 300
SAVE_STEPS = 5000
INPUT_WIDTH = 608
INPUT_HEIGHT = 608

CLS_NORMALIZER = 1.0  # 置信度损失系数
IGNORE_THRESH = 0.7  # 与真值 iou / giou 小于这个阈值就认为没有预测物体
PROB_THRESH = 0.25  # 分类概率的阈值
SCORE_THRESH = 0.25  # 分类得分阈值


# 配置学习率
def config_learning_rate(global_step, num_imgs):
    print('message:配置学习率:'' + str(LR_TYPE) + '', 初始学习率:' + str(LR_INIT))
    if LR_TYPE == 'piecewise':
        lr = tf.train.piecewise_constant(tf.cast(global_step * BATCH_SIZE / num_imgs, tf.int32), PIECEWISE_DOUNDARIES, PIECEWISE_VALUES)
    elif LR_TYPE == 'exponential':
        lr = tf.train.exponential_decay(learning_rate=LR_INIT, global_step=tf.cast(global_step * BATCH_SIZE / num_imgs, tf.int32),
                                        decay_steps=10, decay_rate=0.99, staircase=True)
    elif LR_TYPE == 'constant':
        lr = LR_INIT
    else:
        print('error:不支持的学习率类型:'' + str(LR_TYPE) + ''')
        raise ValueError(str(LR_TYPE) + ':不支持的学习率类型')

    return tf.maximum(lr, LR_LOWER)


# 配置优化器
def config_optimizer(learning_rate):
    print('message:配置优化器:'' + str(OPTIMIZER_TYPE) + ''')
    if OPTIMIZER_TYPE == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER_TYPE == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif OPTIMIZER_TYPE == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    else:
        print('error:不支持的优化器类型:'' + str(OPTIMIZER_TYPE) + ''')
        raise ValueError(str(OPTIMIZER_TYPE) + ':不支持的优化器类型')


# 训练
def train():
    yolo = Yolo(NUM_CLASSES, ANCHORS)
    data = Data(DATABASE_ROOT_DIR, DATABASE_NAMES, PRED_NAMES, NUM_CLASSES, BATCH_SIZE, ANCHORS, MULTI_SCALE_IMG, INPUT_WIDTH, INPUT_HEIGHT)

    inputs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, None, 3])
    y1_true = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, None, 3, 4 + 1 + 20])
    y2_true = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, None, 3, 4 + 1 + 20])
    y3_true = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, None, None, 3, 4 + 1 + 20])
    
    feature_y1, feature_y2, feature_y3 = yolo.inference(inputs, istraining=True)
    loss = yolo.get_loss_v4(feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true,
                            CLS_NORMALIZER, IGNORE_THRESH, PROB_THRESH, SCORE_THRESH)
    l2_loss = tf.losses.get_regularization_loss()
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = config_learning_rate(global_step, data.num_imgs)
    optimizer = config_optimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(loss + l2_loss)
        clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            step = eval(step)
            print('message:存在ckpt模型, global_step=' + str(step))
        else:
            step = 0
            print('message:不存在ckpt模型，从头开始训练')

        num_steps = np.ceil(NUM_EPOCHS * data.num_imgs / BATCH_SIZE)
        while step < num_steps:
            batch_img, y1, y2, y3 = next(data)
            _, train_loss, step, lr = sess.run([train_op, loss, global_step, learning_rate],
                                               feed_dict={inputs: batch_img, y1_true: y1, y2_true: y2, y3_true: y3})
            print('step: %6d, loss: %.5f\t, lr:%.5ff\t' % (step, train_loss, lr))

            if (step + 1) % SAVE_STEPS == 0:
                print('message:保存当前模型, step=' + str(step))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME), global_step=step)

        print('message:保存最终模型, step=' + str(step))
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME), global_step=step)


if __name__ == '__main__':
    train()
