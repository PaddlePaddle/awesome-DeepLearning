# 基于PaddlePaddle框架的YOLOv1复现

论文名称：You only look once unified real-time object detection
论文地址：https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf

# 一、前言
近几年来，目标检测算法取得了很大的突破。比较流行的算法可以分为两类，一类是基于Region Proposal的R-CNN系算法（R-CNN，Fast R-CNN, Faster R-CNN），它们是two-stage的，需要先使用启发式方法（selective search）或者CNN网络（RPN）产生Region Proposal，然后再在Region Proposal上做分类与回归。而另一类是Yolo，SSD这类one-stage算法，其仅仅使用一个CNN网络直接预测不同目标的类别与位置。第一类方法是准确度高一些，但是速度慢，但是第二类算法是速度快，但是准确性要低一些。这可以在图2中看到。本文介绍的是Yolo算法，其全称是You Only Look Once: Unified, Real-Time Object Detection，其实个人觉得这个题目取得非常好，基本上把Yolo算法的特点概括全了：You Only Look Once说的是只需要一次CNN运算，Unified指的是这是一个统一的框架，提供end-to-end的预测，而Real-Time体现是Yolo算法速度快。这里我们谈的是Yolo-v1版本算法，其性能是差于后来的SSD算法的，但是Yolo后来也继续进行改进，产生了Yolo9000算法。

# 二、YOLOv1模型的设计
整体来看，Yolo算法采用一个单独的CNN模型实现end-to-end的目标检测，整个系统如图示：首先将输入图片resize到448x448，然后送入CNN网络，最后处理网络预测结果得到检测的目标。相比R-CNN算法，其是一个统一的框架，其速度更快，而且Yolo的训练过程也是end-to-end的。

![](https://ai-studio-static-online.cdn.bcebos.com/37cace4cc804425bae29ec5ec4372014902800ff5b39494489495b9646213d23)


具体来说，Yolo的CNN网络将输入的图片分割成，其中前4个表征边界框的大小与位置，而最后一个值是置信度。

![](https://ai-studio-static-online.cdn.bcebos.com/722d1fa70c0844939128391fccbcbb96acc1f406c1cd453f9396120766056e3b)


还有分类问题，对于每一个单元格其还要给出预测出。边界框类别置信度表征的是该边界框中目标属于各个类别的可能性大小以及边界框匹配目标的好坏。后面会说，一般会根据类别置信度来过滤网络的预测框。

![](https://ai-studio-static-online.cdn.bcebos.com/be2eea49cbad4009932619cbea6c2764ed515464293643a1988ebad15f3f689c)


```

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_paddle = paddle.to_tensor(self.scale.copy()).astype('float32')
```



# 三、YOLOv1网络结构

Yolo采用卷积网络来提取特征，然后使用全连接层来得到预测值。网络结构参考GooLeNet模型，包含24个卷积层和2个全连接层，如图8所示。对于卷积层，主要使用1x1卷积来做channle reduction，然后紧跟3x3卷积。对于卷积层和全连接层，采用Leaky ReLU激活函数：。但是最后一层却采用线性激活函数。除了上面这个结构，文章还提出了一个轻量级版本Fast Yolo，其仅使用9个卷积层，并且卷积层中使用更少的卷积核。

![](https://ai-studio-static-online.cdn.bcebos.com/a668b84427ae48bbb61cf401d8d4c3fe1b5066e6f5fb4d13b92268c472c028c8)

## 具体实现

```
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        Block (BasicBlock|BottleneckBlock): block module of model.
        depth (int): layers of resnet, default: 50.
        num_classes (int): output dim of last fc layer. If num_classes <=0, last fc layer
                            will not be defined. Default: 1000.
        with_pool (bool): use pool before the last fc layer or not. Default: True.
    Examples:
        .. code-block:: python
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
            resnet50 = ResNet(BottleneckBlock, 50)
            resnet18 = ResNet(BasicBlock, 18)
    """

    def __init__(self, block, depth, num_classes=1000, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = num_classes
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        # if num_classes > 0:
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # if self.with_pool:
        #     x = self.avgpool(x)

        # if self.num_classes > 0:
        #     x = paddle.flatten(x, 1)
        #     x = self.fc(x)

        return x

```



可以看到网络的最后输出为是边界框的预测结果。这样，提取每个部分是非常方便的，这会方面后面的训练及预测时的计算。

![](https://ai-studio-static-online.cdn.bcebos.com/bdd04d80c17648718eeeb5247f58a93c41ce9f4e744f4742ae25652b2334d444)

## 具体实现

```
 def forward(self, x, target=None):
          # backbone
          C_5 = self.backbone(x)

          # head

          # pred
          prediction_result = self.pred(C_5)

          prediction_result=paddle.reshape(prediction_result,shape=[C_5.shape[0], 1 + self.num_classes + 4, -1])
          prediction_result=paddle.fluid.layers.transpose(prediction_result,perm=[0,2,1])


          # prediction_result = prediction_result.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
          # B, HW, C = prediction_result.size()

          # Divide prediction to obj_pred, txtytwth_pred and cls_pred  
          # [B, H*W, 1]
          conf_pred = prediction_result[:, :, :1]
          # [B, H*W, num_cls]
          cls_pred = prediction_result[:, :, 1 : 1 + self.num_classes]
          # [B, H*W, 4]
          txtytwth_pred = prediction_result[:, :, 1 + self.num_classes:]

          # test

          if not self.trainable:
              with paddle.no_grad():
                  # batch size = 1

                  all_conf = paddle.nn.functional.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                  all_bbox = paddle.fluid.layers.clip((self.decode_boxes(txtytwth_pred) / self.scale_paddle)[0], 0., 1.)
                  all_class = (paddle.nn.functional.softmax(cls_pred[0, :, :], 1) * all_conf)

                  # separate box pred and class conf
                  all_conf = all_conf.numpy()
                  all_class = all_class.numpy()
                  all_bbox = all_bbox.numpy()

                  bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

                  return bboxes, scores, cls_inds
          else:
              conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                          pred_txtytwth=txtytwth_pred,
                                                                          label=target)

              return conf_loss, cls_loss, txtytwth_loss, total_loss
```


# 四、训练与损失函数
在训练之前，先在ImageNet上进行了预训练，其预训练的分类模型采用图8中前20个卷积层，然后添加一个average-pool层和全连接层。预训练之后，在预训练得到的20层卷积层之上加上随机初始化的4个卷积层和2个全连接层。由于检测任务一般需要更高清的图片，所以将网络的输入从224x224增加到了448x448。整个网络的流程如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/33430550a83f4b3da031485875f6bc530ec175b895f647cbb49dd74b0173b475)


下面是训练损失函数的分析，Yolo算法将目标检测看成回归问题，所以采用的是均方差损失函数。但是对不同的部分采用了不同的权重值。首先区分定位误差和分类误差。对于定位误差，即边界框坐标预测误差，采用较大的权重。
另外一点时，由于每个单元格预测多个边界框。但是其对应类别只有一个。那么在训练时，如果该单元格内确实存在目标，那么只选择与ground truth的IOU最大的那个边界框来负责预测该目标，而其它边界框认为不存在目标。这样设置的一个结果将会使一个单元格对应的边界框更加专业化，其可以分别适用不同大小，不同高宽比的目标，从而提升模型性能。大家可能会想如果一个单元格内存在多个目标怎么办，其实这时候Yolo算法就只能选择其中一个来训练，这也是Yolo算法的缺点之一。要注意的一点时，对于不存在对应目标的边界框，其误差项就是只有置信度，左标项误差是没法计算的。而只有当一个单元格内确实存在目标时，才计算分类误差项，否则该项也是无法计算的。
综上讨论，最终的损失函数计算如下：

![](https://ai-studio-static-online.cdn.bcebos.com/b6f9632abada456fa85e7ebd9b868cce7dce00914eef41548edc24b260ffa8e7)


其中第一项是边界框中心坐标的误差项，个单元格存在目标。

```
def loss(pred_conf, pred_cls, pred_txtytwth, label):
    obj = 5.0
    noobj = 1.0

    # create loss_f
    conf_loss_function = MSELoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    pred_conf = paddle.nn.functional.sigmoid(pred_conf[:, :, 0])
    pred_cls=paddle.fluid.layers.transpose(pred_cls,perm=[0,2,1])
    # pred_cls = pred_cls.permute(0, 2, 1)
    pred_txty = pred_txtytwth[:, :, :2]
    pred_twth = pred_txtytwth[:, :, 2:]

    gt_obj = label[:, :, 0]
    gt_cls = label[:, :, 1]
    gt_txtytwth = label[:, :, 2:-1]
    gt_box_scale_weight = label[:, :, -1]

    # objectness loss

    pred_conf=pred_conf.astype("float32")
    gt_obj=gt_obj.astype("float32")
    pos_loss, neg_loss = conf_loss_function(pred_conf,gt_obj)

    conf_loss = obj * pos_loss + noobj * neg_loss



    # class loss

    pred_cls=pred_cls.astype('float32')
    gt_cls=gt_cls.astype('int64')

    pred_cls_t=paddle.fluid.layers.transpose(pred_cls,perm=[0,2,1])

    cls_loss_total=paddle.zeros(shape=[0])
    for i in range(32):
        for j in range(169):
            cls_loss_total=paddle.concat(x=[cls_loss_total,cls_loss_function(pred_cls_t[i][j],gt_cls[i][j])],axis=0)
    cls_loss_total=paddle.to_tensor(cls_loss_total,stop_gradient=False)
    cls_loss_total=paddle.reshape(cls_loss_total,shape=[32,169])

    temp_result=cls_loss_total * gt_obj
    temp_result=paddle.to_tensor(temp_result,stop_gradient=False).astype('float32')


    cls_loss = paddle.mean(paddle.sum(temp_result, 1))



    # box loss
    pred_txty=pred_txty.astype('float32')
    gt_txtytwth=gt_txtytwth.astype('float32')


    txty_loss = paddle.mean(paddle.sum(paddle.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight * gt_obj, 1))
    twth_loss = paddle.mean(paddle.sum(paddle.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight * gt_obj, 1))

    txtytwth_loss = txty_loss + twth_loss

    total_loss = conf_loss + cls_loss + txtytwth_loss

    return conf_loss, cls_loss, txtytwth_loss, total_loss
```



# 五、模型复现指标
## 5.1 模型参数
- Batchsize: 32
- 基础学习率: 1e-3
- epoch: 160
- LRstep: 60, 90
- 优化方式: 动量版SGD



## 5.2 复现环境

- Python3.7, opencv-python, PaddlePaddle 2.1.2

## 5.3 实验结果
- VOC2007-test:
![](https://ai-studio-static-online.cdn.bcebos.com/fea016840ab14735afbe809529b85fe0afc55601204f43769b31b7623e1955e6)



# 六、实验流程

登录AI Studio可在线运行：https://aistudio.baidu.com/aistudio/projectdetail/2259467


将平台挂载VOC2007+2012的训练数据集压缩包VOCdevkit.zip移到/work/datasets文件夹下，解压数据集 ```datasets/```下的 ```VOCdevkit.zip``


## 6.1 训练
```Shell
python train.py
```

## 如需设置参数请按照以下格式填写

```
 parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')  
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,help='keep training')
    # parser.add_argument('-r', '--resume', default=None, type=str,
    #                     help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='use gpu.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='Gamma update for SGD')
```

## 6.2 测试
```Shell
python test.py
```

## 如需设置参数请按照以下格式填写
```
parser = argparse.ArgumentParser(description='YOLO Detection')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default='./checkpoints/yolo-model-best.pdparams',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='Confidence threshold')
parser.add_argument('--nms_thresh', default=0.50, type=float,
                    help='NMS threshold')
parser.add_argument('--visual_threshold', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--gpu', action='store_true', default=True,
                    help='use cuda.')
```


## 6.3 评估
```Shell
python eval.py
```

## 如需设置参数请按照以下格式填写

```
parser = argparse.ArgumentParser(description='YOLO Detector Evaluation')
parser.add_argument('-v', '--version', default='yolo',
                    help='yolo.')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--trained_model', type=str,
                    default='./checkpoints/yolo-model-best.pdparams',
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--gpu', action='store_true', default=True,
                    help='Use gpu')
```
