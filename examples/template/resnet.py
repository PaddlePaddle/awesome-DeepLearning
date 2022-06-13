# coding=utf-8
import paddle.v2 as paddle
 
# ***********************定义VGG卷积神经网络模型***************************************
def vgg_bn_drop(datadim):
    # 获取输入数据大小
    img = paddle.layer.data(name="image",
                            type=paddle.data_type.dense_vector(datadim))
 
    def conv_block(ipt, num_filter, groups, dropouts, num_channels=None):
        return paddle.networks.img_conv_group(
            input=ipt,
            num_channels=num_channels,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act=paddle.activation.Relu(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=paddle.pooling.Max())
 
    conv1 = conv_block(img, 64, 2, [0.3, 0], 3)
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])
 
    drop = paddle.layer.dropout(input=conv5, dropout_rate=0.5)
    fc1 = paddle.layer.fc(input=drop, size=512, act=paddle.activation.Linear())
    bn = paddle.layer.batch_norm(input=fc1,
                                 act=paddle.activation.Relu(),
                                 layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(input=bn, size=512, act=paddle.activation.Linear())
 
    # 通过神经网络模型再使用Softmax获得分类器(全连接)
    out = paddle.layer.fc(input=fc2,
                          size=10,
                          act=paddle.activation.Softmax())
    return out