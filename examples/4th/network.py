import tensorflow as tf
import numpy as np


slim = tf.contrib.slim


def conv(inputs, out_channels, kernel_size=3, stride=1):
    if stride > 1:
        inputs = padding(inputs, kernel_size)

    # 这里可以自定义激活方式, 默认 relu, 可以实现空洞卷积:rate 参数
    outputs = slim.conv2d(inputs, out_channels, kernel_size, stride=stride,
                          padding=('SAME' if stride == 1 else 'VALID'))
    return outputs


def padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_start = pad_total // 2
    pad_end = pad_total - pad_start
    outputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]])
    return outputs


# YOLO残差模块
def yolo_res_block(inputs, channels, res_num):
    net = conv(inputs, channels * 2, stride=2)
    route = conv(net, channels, kernel_size=1)
    net = conv(net, channels, kernel_size=1)

    for _ in range(res_num):
        tmp = net
        net = conv(net, channels, kernel_size=1)
        net = conv(net, channels)
        net = tmp + net

    net = conv(net, channels, kernel_size=1)
    net = tf.concat([net, route], -1)
    net = conv(net, channels * 2, kernel_size=1)

    return net


# YOLO卷积模块
def yolo_conv_block(inputs, channels, a, b):
    net = inputs
    for _ in range(a):
        net = conv(net, channels / 2, kernel_size=1)
        net = conv(net, channels)

    for _ in range(b):
        channels /= 2
        net = conv(net, channels, kernel_size=1)
    outputs = net
    return outputs


# YOLO最大池化模块
def yolo_maxpool_block(inputs):
    maxpool_5 = tf.nn.max_pool(inputs, [1, 5, 5, 1], [1, 1, 1, 1], 'SAME')
    maxpool_9 = tf.nn.max_pool(inputs, [1, 9, 9, 1], [1, 1, 1, 1], 'SAME')
    maxpool_13 = tf.nn.max_pool(inputs, [1, 13, 13, 1], [1, 1, 1, 1], 'SAME')
    outputs = tf.concat([maxpool_13, maxpool_9, maxpool_5, inputs], -1)
    return outputs


# YOLO上采样模块
def yolo_upsample_block(inputs, in_channels, route):
    shape = tf.shape(inputs)
    out_height, out_width = shape[1] * 2, shape[2] * 2
    inputs = tf.image.resize_nearest_neighbor(inputs, (out_height, out_width))

    route = conv(route, in_channels, kernel_size=1)

    outputs = tf.concat([route, inputs], -1)
    return outputs


def mish(inputs):
    threshold = 20.0
    tmp = inputs
    inputs = tf.where(tf.math.logical_and(tf.less(inputs, threshold), tf.greater(inputs, -threshold)),
                      tf.log(1 + tf.exp(inputs)),
                      tf.zeros_like(inputs))
    inputs = tf.where(tf.less(inputs, -threshold),
                      tf.exp(inputs),
                      inputs)
    outputs = tmp * tf.tanh(inputs)
    return outputs


class Yolo:
    def __init__(self, num_classes, anchors):
        self.num_classes = num_classes
        self.anchors = anchors
        self.width = 608
        self.height = 608
        pass

    def inference(self, inputs, batch_norm_decay=0.9, weight_decay=0.0005, istraining=True, reuse=False):
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': istraining,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                # darknet53 特征
                with slim.arg_scope([slim.conv2d], activation_fn=mish):
                    with tf.variable_scope('Downsample'):
                        net = conv(inputs, 32)
                        # [N, 608, 608, 32]
                        net = conv(net, 64, stride=2)
                        # [N, 304, 304, 64]
                        route = conv(net, 64, kernel_size=1)
                        net = conv(net, 64, kernel_size=1)
                        tmp = net
                        net = conv(net, 32, kernel_size=1)
                        net = conv(net, 64)
                        net = tmp + net
                        net = conv(net, 64, kernel_size=1)
                        net = tf.concat([net, route], -1)
                        # [N, 304, 304, 128]
                        net = conv(net, 64, kernel_size=1)

                        # [N, 304, 304, 64]
                        net = yolo_res_block(net, 64, 2)
                        # [N, 152, 152, 128]
                        net = yolo_res_block(net, 128, 8)
                        # [N, 76, 76, 256]
                        up_route_54 = net

                        net = yolo_res_block(net, 256, 8)
                        # [N, 38, 38, 512]
                        up_route_85 = net

                        net = yolo_res_block(net, 512, 4)

                # ########## leaky_relu 激活 ##########
                with slim.arg_scope([slim.conv2d], activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1)):
                    with tf.variable_scope('leaky_relu'):
                        # [N, 19, 19, 1024]
                        net = yolo_conv_block(net, 1024, 1, 1)
                        # [N, 19, 19, 512]
                        net = yolo_maxpool_block(net)
                        # [N, 19, 19, 2048]
                        net = conv(net, 512, kernel_size=1)
                        net = conv(net, 1024)
                        net = conv(net, 512, kernel_size=1)
                        # [N, 19, 19, 512]
                        route_3 = net

                        net = conv(net, 256, kernel_size=1)
                        # [N, 19, 19, 256]
                        net = yolo_upsample_block(net, 256, up_route_85)
                        # [N, 38, 38, 768]
                        net = yolo_conv_block(net, 512, 2, 1)
                        # [N, 38, 38, 256]
                        route_2 = net

                        net = conv(net, 128, kernel_size=1)
                        # [N, 38, 38, 128]
                        net = yolo_upsample_block(net, 128, up_route_54)
                        # [N, 76, 76, 384]
                        net = yolo_conv_block(net, 256, 2, 1)
                        # [N, 76, 76, 128]
                        route_1 = net

                    with tf.variable_scope('yolo'):
                        net = conv(route_1, 256)
                        # [N, 76, 76, 256]
                        net = slim.conv2d(net, 3 * (4 + 1 + self.num_classes), 1,
                                          stride=1, normalizer_fn=None, activation_fn=None,
                                          biases_initializer=tf.zeros_initializer())
                        # [N, 76, 76, 3 * (5 + num_classes)]
                        feature_y3 = net

                        net = conv(route_1, 256, stride=2)
                        # [N, 38, 38, 256]
                        net = tf.concat([net, route_2], -1)
                        # [N, 38, 38, 512]
                        net = yolo_conv_block(net, 512, 2, 1)
                        # [N, 38, 38, 256]
                        route_147 = net

                        net = conv(net, 512)
                        # [N, 38, 38, 512]
                        net = slim.conv2d(net, 3*(4+1+self.num_classes), 1,
                                          stride=1, normalizer_fn=None, activation_fn=None,
                                          biases_initializer=tf.zeros_initializer())
                        # [N, 38, 38, 3 * (5 + num_classes)]
                        feature_y2 = net

                        net = conv(route_147, 512, stride=2)
                        # [N, 19, 19, 512]
                        net = tf.concat([net, route_3], -1)
                        # [N, 19, 19, 1024]
                        net = yolo_conv_block(net, 1024, 3, 0)
                        # [N, 19, 19, 1024]
                        net = slim.conv2d(net, 3*(4+1+self.num_classes), 1,
                                          stride=1, normalizer_fn=None,
                                          activation_fn=None, biases_initializer=tf.zeros_initializer())
                        # [N, 19, 19, 3 * (5 + num_classes)]
                        feature_y1 = net
        if not istraining:
            return self.get_predict_result(feature_y1, feature_y2, feature_y3)
        return feature_y1, feature_y2, feature_y3

    # 计算最大的 IOU, GIOU
    @staticmethod
    def compute_iou(pre_xy, pre_wh, valid_yi_true):
        """
            pre_xy : [13, 13, 3, 2]
            pre_wh : [13, 13, 3, 2]
            valid_yi_true : [V, 5 + num_classes] or [V, 4]
            return:
                iou, giou : [13, 13, 3, V], V表示V个真值
        """

        # [13, 13, 3, 2] ==> [13, 13, 3, 1, 2]
        pre_xy = tf.expand_dims(pre_xy, -2)
        pre_wh = tf.expand_dims(pre_wh, -2)

        # [V, 2]
        yi_true_xy = valid_yi_true[..., 0:2]
        yi_true_wh = valid_yi_true[..., 2:4]

        # 相交区域左上角坐标 : [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersection_left_top = tf.maximum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 相交区域右下角坐标
        intersection_right_bottom = tf.minimum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))

        # 相交区域宽高 [13, 13, 3, V, 2] == > [13, 13, 3, V, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        
        # 相交区域面积 : [13, 13, 3, V]
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        # 预测box面积 : [13, 13, 3, 1]
        pre_area = pre_wh[..., 0] * pre_wh[..., 1]
        # 真值 box 面积 : [V]
        true_area = yi_true_wh[..., 0] * yi_true_wh[..., 1]
        # [V] ==> [1, V]
        true_area = tf.expand_dims(true_area, axis=0)
        # iou : [13, 13, 3, V]
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)  # 防止除0

        # 并集区域面积 : [13, 13, 3, V, 1] ==> [13, 13, 3, V] 
        combine_area = combine_wh[..., 0] * combine_wh[..., 1]
        # giou : [13, 13, 3, V]
        giou = (intersection_area+1e-10) / combine_area  # 加上一个很小的数字，确保 giou 存在
        
        return iou, giou

    # 计算 CIOU 损失
    @staticmethod
    def __get_ciou_loss(pre_xy, pre_wh, yi_box):
        """
        the formula of CIOU_LOSS is refers to http://bbs.cvmart.net/topics/1436
        计算每一个 box 与真值的 ciou 损失
        pre_xy:[batch_size, 13, 13, 3, 2]
        pre_wh:[batch_size, 13, 13, 3, 2]
        yi_box:[batch_size, 13, 13, 3, 4]
        return:[batch_size, 13, 13, 3, 1]
        """
        # [batch_size, 13, 13, 3, 2]
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        # 上下左右
        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        # 相交区域坐标 : [batch_size, 13, 13, 3,2]
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        # 相交区域宽高 : [batch_size, 13, 13, 3, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 相交区域面积 : [batch_size, 13, 13, 3, 1]
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)

        # 并集区域对角线 : [batch_size, 13, 13, 3, 1]
        C = tf.square(combine_wh[..., 0:1]) + tf.square(combine_wh[..., 1:2])
        # 中心点对角线:[batch_size, 13, 13, 3, 1]
        D = tf.square(yi_true_xy[..., 0:1] - pre_xy[..., 0:1]) + tf.square(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])

        # box面积 : [batch_size, 13, 13, 3, 1]
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]

        # iou : [batch_size, 13, 13, 3, 1]
        iou = intersection_area / (pre_area + true_area - intersection_area)

        pi = 3.14159265358979323846

        # 衡量长宽比一致性的参数:[batch_size, 13, 13, 3, 1]
        v = 4 / (pi * pi) * tf.square(tf.subtract(tf.math.atan(yi_true_wh[..., 0:1] / yi_true_wh[..., 1:2]), tf.math.atan(pre_wh[..., 0:1] / pre_wh[..., 1:2])))

        # trade-off 参数
        # alpha
        alpha = v / (1.0 - iou + v)
        ciou_loss = 1.0 - iou + D / C + alpha * v
        return ciou_loss

    # 得到低iou的地方
    def __get_low_iou_mask(self, pre_xy, pre_wh, yi_true, use_iou=True, ignore_thresh=0.5):
        """
        pre_xy:[batch_size, 19, 19, 3, 2]
        pre_wh:[batch_size, 19, 19, 3, 2]
        yi_true:[batch_size, 19, 19, 3, 5 + num_classes]
        use_iou:是否使用 iou 作为计算标准
        ignore_thresh:iou小于这个值就认为与真值不重合
        return: [batch_size, 19, 19, 3, 1]
        返回低IOU锚框的bool蒙版
        """
        # 置信度:[batch_size, 19, 19, 3, 1]
        conf_yi_true = yi_true[..., 4:5]

        # iou小的地方
        low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        
        def loop_cond(index):
            # 遍历batch中所有图片后停止
            return tf.less(index, tf.shape(yi_true)[0])

        def loop_body(index, mask):
            # 一张图片的全部真值 : [19, 19, 3, num_classes + 5] & [19, 19, 3, 1] == > [V, num_classes + 5]
            valid_yi_true = tf.boolean_mask(yi_true[index], tf.cast(conf_yi_true[index, ..., 0], tf.bool))
            # 计算 iou / giou : [19, 19, 3, V]
            iou, giou = self.compute_iou(pre_xy[index], pre_wh[index], valid_yi_true)

            # [19, 19, 3]
            if use_iou:
                best_giou = tf.reduce_max(iou, axis=-1)
            else:
                best_giou = tf.reduce_max(giou, axis=-1)
            # [19, 19, 3]
            low_iou_mask_tmp = best_giou < ignore_thresh
            # [19, 19, 3, 1]
            low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
            # 写入
            mask = mask.write(index, low_iou_mask_tmp)
            return index + 1, mask

        _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
        # 拼接:[batch_size, 19, 19, 3, 1]
        low_iou_mask = low_iou_mask.stack()
        return low_iou_mask

    # 得到预测物体概率低的地方的掩码
    @staticmethod
    def __get_low_prob_mask(prob, prob_thresh=0.25):
        """
        prob:[batch_size, 19, 19, 3, num_classes]
        prob_thresh:物体概率预测的阈值
        return: bool [batch_size, 19, 19, 3, 1]
        返回低预测概率锚框的bool蒙版
        """
        # [batch_size, 19, 19, 3, 1]
        max_prob = tf.reduce_max(prob, axis=-1, keepdims=True)
        low_prob_mask = max_prob < prob_thresh        
        return low_prob_mask

    # 对预测值进行解码
    def __decode_feature(self, yi_pred, curr_anchors):
        """
        yi_pred:[batch_size, grid_size, grid_size, 3 * (num_classes + 5)]
        curr_anchors:[3, 2], 这一层对应的 anchor, 真实值
        return:
            xy:[batch_size, grid_size, grid_size, 3, 2], 相对于原图的中心坐标
            wh:[batch_size, grid_size, grid_size, 3, 2], 相对于原图的宽高
            conf:[batch_size, grid_size, grid_size, 3, 1]
            prob:[batch_size, grid_size, grid_size, 3, num_classes]
        """
        shape = tf.cast(tf.shape(yi_pred), tf.float32)
        batch_size, grid_size = shape[0], shape[1]
        # [batch_size, grid_size, grid_size, 3, num_classes + 5]
        yi_pred = tf.reshape(yi_pred, [batch_size, grid_size, grid_size, 3, 5 + self.num_classes])
        # xy: [batch_size, grid_size, grid_size, 3, 2]
        # wh: [batch_size, grid_size, grid_size, 3, 2]
        # conf: [batch_size, grid_size, grid_size, 3, 1]
        # prob: [batch_size, grid_size, grid_size, 3, num_classes]
        xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, self.num_classes], axis=-1)

        ''' 计算 x,y 的坐标偏移 '''
        offset_x = tf.range(grid_size, dtype=tf.float32)  # 宽
        offset_y = tf.range(grid_size, dtype=tf.float32)  # 高
        offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
        offset_x = tf.reshape(offset_x, (-1, 1))
        offset_y = tf.reshape(offset_y, (-1, 1))
        # [grid_size * grid_size, 2], 每一行对应一个网格的原点坐标
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)
        # [grid_size, grid_size, 1, 2]
        offset_xy = tf.reshape(offset_xy, [grid_size, grid_size, 1, 2])
        
        xy = tf.math.sigmoid(xy) + offset_xy
        xy = xy / [grid_size, grid_size]

        wh = tf.math.exp(wh) * curr_anchors
        wh = wh / [self.width, self.height]

        return xy, wh, conf, prob

    # 计算损失, yolov4
    def __compute_loss_v4(self, xy, wh, conf, prob, yi_true, cls_normalizer=1.0, ignore_thresh=0.5, prob_thresh=0.25, score_thresh=0.25, iou_normalizer=0.07):
        """
        xy:[batch_size, 19, 19, 3, 2]
        wh:[batch_size, 19, 19, 3, 2]
        conf:[batch_size, 19, 19, 3, 1]
        prob:[batch_size, 19, 19, 3, num_classes]
        yi_true:[batch_size, 19, 19, 3, num_classes]
        cls_normalizer:置信度损失参数
        ignore_thresh:与真值iou阈值
        prob_thresh:最低分类概率阈值
        score_thresh:最低分类得分阈值
        iou_normalizer:iou_loss 系数
        return: 总的损失

        loss_total:总的损失
        xy_loss:中心坐标损失
        wh_loss:宽高损失
        conf_loss:置信度损失
        class_loss:分类损失
        """
        # 低IOU锚框的bool蒙版 [batch_size, 19, 19, 3, 1]
        low_iou_mask = self.__get_low_iou_mask(xy, wh, yi_true, ignore_thresh=ignore_thresh)
        # 低预测概率锚框的bool蒙版 [batch_size, 19, 19, 3, 1]
        low_prob_mask = self.__get_low_prob_mask(prob, prob_thresh=prob_thresh)        
        # 低IOU或低预测概率锚框的bool蒙版 [batch_size, 19, 19, 3, 1]
        low_iou_prob_mask = tf.math.logical_or(low_iou_mask, low_prob_mask)
        low_iou_prob_mask = tf.cast(low_iou_prob_mask, tf.float32)

        batch_size = tf.cast(tf.shape(xy)[0], tf.float32)

        # 所有锚框产生的预测框面积 [batch_size, 19, 19, 3, 1]
        conf_scale = wh[..., 0:1] * wh[..., 1:2]
        conf_scale = tf.where(tf.math.greater(conf_scale, 0), tf.math.sqrt(conf_scale), conf_scale)
        conf_scale = conf_scale * cls_normalizer                                                        
        conf_scale = tf.math.square(conf_scale)

        # 所有不含物体锚框的置信度损失
        no_obj_mask = 1.0 - yi_true[..., 4:5]
        conf_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:, :, :, :, 4:5], logits=conf) * conf_scale * no_obj_mask * low_iou_prob_mask

        # 所有包含物体锚框的置信度损失
        obj_mask = yi_true[..., 4:5]
        conf_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(labels=yi_true[:, :, :, :, 4:5], logits=conf) * np.square(cls_normalizer) * obj_mask

        # 置信度损失
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e3)
        conf_loss = tf.reduce_sum(conf_loss) / batch_size
        conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e4)

        # CIOU损失
        yi_true_ciou = tf.where(tf.math.less(yi_true[..., 0:4], 1e-10), tf.ones_like(yi_true[..., 0:4]), yi_true[..., 0:4])
        pre_xy = tf.where(tf.math.less(xy, 1e-10), tf.ones_like(xy), xy)
        pre_wh = tf.where(tf.math.less(wh, 1e-10), tf.ones_like(wh), wh)
        ciou_loss = self.__get_ciou_loss(pre_xy, pre_wh, yi_true_ciou)
        ciou_loss = tf.where(tf.math.greater(obj_mask, 0.5), ciou_loss, tf.zeros_like(ciou_loss))
        ciou_loss = tf.square(ciou_loss * obj_mask) * iou_normalizer
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e3)
        ciou_loss = tf.reduce_sum(ciou_loss) / batch_size
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e4)

        # xy损失
        xy = tf.clip_by_value(xy, 1e-10, 1e4)
        xy_loss = obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e3)
        xy_loss = tf.reduce_sum(xy_loss) / batch_size
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e4)

        # wh损失
        wh_y_true = tf.where(condition=tf.math.less(yi_true[..., 2:4], 1e-10), x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.math.less(wh, 1e-10), x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e3)
        wh_loss = tf.reduce_sum(wh_loss) / batch_size
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e4)
        
        # prob 损失
        score = prob * conf
        
        high_score_mask = score > score_thresh
        high_score_mask = tf.cast(high_score_mask, tf.float32)
        
        class_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[..., 5:5+self.num_classes],
                                                        logits=prob 
                                                    ) * low_iou_prob_mask * no_obj_mask * high_score_mask
        
        class_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[..., 5:5+self.num_classes],
                                                        logits=prob
                                                    ) * obj_mask

        class_loss = class_loss_no_obj + class_loss_obj        
        class_loss = tf.clip_by_value(class_loss, 0.0, 1e3)
        class_loss = tf.reduce_sum(class_loss) / batch_size
        class_loss = tf.clip_by_value(class_loss, 0.0, 1e4)

        loss_total = xy_loss + wh_loss + conf_loss + class_loss + ciou_loss
        return loss_total

    # 获得损失 yolov4
    def get_loss_v4(self, feature_y1, feature_y2, feature_y3, y1_true, y2_true, y3_true, cls_normalizer=1.0, ignore_thresh=0.5, prob_thresh=0.25, score_thresh=0.25):
        """
        feature_y1:[batch_size, 19, 19, 3 * (5 + num_classes)]
        feature_y2:[batch_size, 38, 38, 3 * (5 + num_classes)]
        feature_y3:[batch_size, 76, 76, 3 * (5 + num_classes)]
        y1_true: y1尺度的
        y2_true: y2尺度的标签
        y3_true: y3尺度的标签
        cls_normalizer:分类损失系数
        ignore_thresh:与真值 iou 阈值
        prob_thresh:分类概率最小值
        score_thresh:分类得分最小值
        return:total_loss
        """
        # y1
        xy, wh, conf, prob = self.__decode_feature(feature_y1, self.anchors[2])
        loss_y1 = self.__compute_loss_v4(xy, wh, conf, prob, y1_true, cls_normalizer=cls_normalizer, ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        # y2
        xy, wh, conf, prob = self.__decode_feature(feature_y2, self.anchors[1])
        loss_y2 = self.__compute_loss_v4(xy, wh, conf, prob, y2_true, cls_normalizer=cls_normalizer, ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        # y3
        xy, wh, conf, prob = self.__decode_feature(feature_y3, self.anchors[0])
        loss_y3 = self.__compute_loss_v4(xy, wh, conf, prob, y3_true, cls_normalizer=cls_normalizer, ignore_thresh=ignore_thresh, prob_thresh=prob_thresh, score_thresh=score_thresh)

        return loss_y1 + loss_y2 + loss_y3

    # 得到预测框
    def __get_boxes(self, feature, anchors):
        xy, wh, conf, prob = self.__decode_feature(feature, anchors)
        conf, prob = tf.sigmoid(conf), tf.sigmoid(prob)
        boxes = tf.concat([xy[..., 0: 1] - wh[..., 0: 1] / 2.0,
                           xy[..., 1: 2] - wh[..., 1: 2] / 2.0,
                           xy[..., 0: 1] + wh[..., 0: 1] / 2.0,
                           xy[..., 1: 2] + wh[..., 1: 2] / 2.0], -1)
        shape = tf.shape(feature)
        # [batch_size, 19 * 19 * 3, 4]
        boxes = tf.reshape(boxes, (shape[0], shape[1] * shape[2] * 3, -1))
        # [batch_size, 19 * 19 * 3, 1]
        conf = tf.reshape(conf, (shape[0], shape[1] * shape[2] * 3, 1))
        # [batch_size, 19 * 19 * 3, num_classes]
        prob = tf.reshape(prob, (shape[0], shape[1] * shape[2] * 3, -1))
        return boxes, conf, prob

    # 得到预测结果
    def get_predict_result(self, feature_y1, feature_y2, feature_y3, score_threshold=0.5, iou_threshold=0.4, max_boxes=100):
        """
        feature_y1: [batch_size, 19, 19, 3 * (num_classes + 5)]
        feature_y2: [batch_size, 38, 38, 3 * (num_classes + 5)]
        feature_y3: [batch_size, 76, 76, 3 * (num_classes + 5)]
        num_classes: 分类数
        score_thresh: NMS分数阈值
        iou_thresh : NMS IOU阈值
        max_box : NMS最多保留目标数
        return:
            boxes:[V, 4]包含[x_min, y_min, x_max, y_max]
            score:[V, 1]
            label:[V, 1]
        """
        boxes_y1, conf_y1, prob_y1 = self.__get_boxes(feature_y1, self.anchors[2])
        boxes_y2, conf_y2, prob_y2 = self.__get_boxes(feature_y2, self.anchors[1])
        boxes_y3, conf_y3, prob_y3 = self.__get_boxes(feature_y3, self.anchors[0])

        # [batch_size, 19 * 19 * 3 + 38 * 38 * 3 + 76 * 76 * 3, 4]
        boxes = tf.concat([boxes_y1, boxes_y2, boxes_y3], 1)
        # [batch_size, 19 * 19 * 3 + 38 * 38 * 3 + 76 * 76 * 3, 1]
        conf = tf.concat([conf_y1, conf_y2, conf_y3], 1)
        # [batch_size, 19 * 19 * 3 + 38 * 38 * 3 + 76 * 76 * 3, num_classes]
        prob = tf.concat([prob_y1, prob_y2, prob_y3], 1)
        # [batch_size, 19 * 19 * 3 + 38 * 38 * 3 + 76 * 76 * 3, num_classes]
        scores = conf * prob

        boxes = tf.reshape(boxes, [-1, 4])
        scores = tf.reshape(scores, [-1, self.num_classes])

        box_list, score_list = [], []
        for i in range(self.num_classes):
            nms_indices = tf.image.non_max_suppression(boxes=boxes,
                                                       scores=scores[:, i],
                                                       max_output_size=max_boxes,
                                                       iou_threshold=iou_threshold,
                                                       score_threshold=score_threshold,
                                                       name='nms_indices')

            box_list.append(tf.gather(boxes, nms_indices))
            score_list.append(tf.gather(scores, nms_indices))

        boxes = tf.concat(box_list, axis=0, name='pred_boxes')
        scores = tf.reduce_max(tf.concat(score_list, axis=0), axis=1, name='pred_scores')
        labels = tf.argmax(tf.concat(score_list, axis=0), axis=1, name='pred_labels')

        return boxes, scores, labels
