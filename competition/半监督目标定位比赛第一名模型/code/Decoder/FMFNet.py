import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SSM(nn.Layer): # 语义补充模块
    def __init__(self):
        super(SSM, self).__init__()
        self.cv1 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2D(64)
        self.cv2 = nn.Conv2D(64, 64, 3, 1, 3, dilation=3)
        self.bn2 = nn.BatchNorm2D(64)

    def forward(self, x):
        d1 = self.bn1(self.cv1(x))
        d2 = self.bn2(self.cv2(x))
        out = F.relu(d1+d2+x)
        return out


class Feature_mutual_feedback_module(nn.Layer): # 特征互反馈模块 FMF
    def __init__(self):
        super(Feature_mutual_feedback_module, self).__init__()
        self.cv1 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2D(64)
        self.cv2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(64)

        self.cv3 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2D(64)
        self.cv4 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2D(64)

    def forward(self, l, h):
        h_l = F.interpolate(h, size=l.shape[2:], mode='bilinear', align_corners=True)
        l_h = F.interpolate(l, size=h.shape[2:], mode='bilinear', align_corners=True)

        h_l = F.relu(self.bn1(self.cv1(h_l)))
        l_h = F.relu(self.bn2(self.cv2(l_h)))

        l_h_l = self.bn3(self.cv3(l * h_l))
        h_l_h = self.bn4(self.cv4(h * l_h))

        l = F.relu(l_h_l + l)
        h = F.relu(h_l_h + h)
        return l, h


class Progressive_fusion_module(nn.Layer): # Progressive_fusion_module 渐进融合模块 PFM
    def __init__(self):
        super(Progressive_fusion_module, self).__init__()
        self.cv1 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2D(64)

        self.cv2 = nn.Conv2D(128, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2D(64)

        self.cv3 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2D(64)

        self.cv4 = nn.Conv2D(128, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2D(64)

        self.cv5 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm2D(64)

        self.cv6 = nn.Conv2D(128, 64, 3, 1, 1)
        self.bn6 = nn.BatchNorm2D(64)

        self.cv7 = nn.Conv2D(64, 64, 3, 1, 1)
        self.bn7 = nn.BatchNorm2D(64)

        self.cv8 = nn.Conv2D(128, 64, 3, 1, 1)
        self.bn8 = nn.BatchNorm2D(64)

    def forward(self, out1, out2, out3, out4, out5):
        out5 = F.interpolate(out5, size=out4.shape[2:], mode='bilinear', align_corners=True)
        out5 = F.relu(self.bn1(self.cv1(out5)))

        out4 = paddle.concat([out4, out5], axis=1)
        out4 = F.relu(self.bn2(self.cv2(out4)))
        out4 = F.interpolate(out4, size=out3.shape[2:], mode='bilinear', align_corners=True)
        out4 = F.relu(self.bn3(self.cv3(out4)))

        out3 = paddle.concat([out3, out4], axis=1)
        out3 = F.relu(self.bn4(self.cv4(out3)))
        out3 = F.interpolate(out3, size=out2.shape[2:], mode='bilinear', align_corners=True)
        out3 = F.relu(self.bn5(self.cv5(out3)))

        out2 = paddle.concat([out2, out3], axis=1)
        out2 = F.relu(self.bn6(self.cv6(out2)))
        out2 = F.interpolate(out1, size=out2.shape[2:], mode='bilinear', align_corners=True)
        out2 = F.relu(self.bn7(self.cv7(out2)))

        out1 = paddle.concat([out1, out2], axis=1)
        out1 = F.relu(self.bn8(self.cv8(out1)))
        return out1


class FMFMs(nn.Layer):
    def __init__(self):
        super(FMFMs, self).__init__()
        self.mm45, self.mm34, self.mm23, self.mm12 = Feature_mutual_feedback_module(), \
                                                     Feature_mutual_feedback_module(), \
                                                     Feature_mutual_feedback_module(), \
                                                     Feature_mutual_feedback_module()

    def forward(self, o1, o2, o3, o4, o5):
        out4, out5 = self.mm45(o4, o5)  # new45 new54
        out3, out4 = self.mm34(o3, out4)  # new34 newnew345
        out2, out3 = self.mm23(o2, out3)  # new2 newnew234
        out1, out2 = self.mm12(o1, out2)
        return out1, out2, out3, out4, out5


class SEModule(nn.Layer):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return inputs * x


class FMFModel(nn.Layer):
    def __init__(self, backbone):
        super(FMFModel, self).__init__()
        self.backbone = backbone
        self.se1, self.se2,self.se3,self.se4,self.se5 = SEModule(64), SEModule(64), SEModule(64), SEModule(64), SEModule(64)
        self.squeeze5 = nn.Sequential(nn.Conv2D(2048, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.squeeze4 = nn.Sequential(nn.Conv2D(1024, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.squeeze3 = nn.Sequential(nn.Conv2D(512, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.squeeze2 = nn.Sequential(nn.Conv2D(256, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.squeeze1 = nn.Sequential(nn.Conv2D(64, 64, 3, 1, 1), nn.BatchNorm2D(64), nn.ReLU())
        self.fa1, self.fa2, self.fa3, self.fa4, self.fa5 = SSM(), \
                                                           SSM(), \
                                                           SSM(), \
                                                           SSM(), \
                                                           SSM()
        self.FMF1, self.FMF2 = FMFMs(), FMFMs()
        self.FMF3 = FMFMs()
        self.mso = Progressive_fusion_module()
        self.linear = nn.Conv2D(64, 1, 3, 1, 1)
        for p in self.backbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.backbone(x)
        out1, out2, out3, out4, out5 = self.squeeze1(out1), self.squeeze2(out2), \
                                            self.squeeze3(out3), self.squeeze4(out4), \
                                            self.squeeze5(out5)
        out1, out2, out3, out4, out5 = self.se1(out1), self.se2(out2), \
                                            self.se3(out3), self.se4(out4), \
                                            self.se5(out5)
        out1, out2, out3, out4, out5 = self.fa1(out1), self.fa2(out2), \
                                            self.fa3(out3), self.fa4(out4), \
                                            self.fa5(out5)
        out1, out2, out3, out4, out5 = self.FMF1(out1, out2, out3, out4, out5)
        out1, out2, out3, out4, out5 = self.FMF2(out1, out2, out3, out4, out5)
        out1, out2, out3, out4, out5 = self.FMF3(out1, out2, out3, out4, out5)
        out = self.mso(out1, out2, out3, out4, out5)
        out = F.interpolate(self.linear(out), size=x.shape[2:], mode='bilinear', align_corners=True)
        return [out]
