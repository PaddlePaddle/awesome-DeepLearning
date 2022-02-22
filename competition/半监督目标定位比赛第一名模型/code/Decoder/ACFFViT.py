import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MFR(nn.Layer):
    def __init__(self, in_channel_left, in_channel_right):
        super(MFR, self).__init__()
        self.conv0 = nn.Conv2D(in_channel_left, 192, 3, 1, 1)
        self.bn0 = nn.BatchNorm2D(192)
        self.conv1 = nn.Conv2D(in_channel_right, 192, 1)
        self.bn1 = nn.BatchNorm2D(192)

        self.conv2 = nn.Conv2D(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(192)

        self.conv13 = nn.Conv2D(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn13 = nn.BatchNorm2D(192)
        self.conv31 = nn.Conv2D(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn31 = nn.BatchNorm2D(192)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)))
        down = F.relu(self.bn1(self.conv1(down)))
        left = F.relu(self.bn2(self.conv2(left)))

        down = F.relu(self.bn31(self.conv31(down)))
        down = self.bn13(self.conv13(down))
        return F.relu(left + down)


class ChannelAttention(nn.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        self.fc1 = nn.Conv2D(in_planes, in_planes // 16, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_planes // 16, in_planes, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CFF(nn.Layer):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(CFF, self).__init__()
        self.conv0 = nn.Conv2D(in_channel_left, 192, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2D(192)

        self.conv1 = nn.Conv2D(in_channel_down, 192, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(192)
        self.conv2 = nn.Conv2D(in_channel_right, 192, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(192)

        self.conv3 = nn.Conv2D(192 * 3, 192, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(192)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)))  # 256 channels

        down = F.relu(self.bn1(self.conv1(down)))  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)))  # 256 channels

        down = F.interpolate(down, size=left.shape[2:], mode='bilinear')
        right = F.interpolate(right, size=left.shape[2:], mode='bilinear')

        x = left * down  # l*h
        y = left * right  # l*c
        z = right * down  # h*c
        out = paddle.concat([x, y, z], 1)
        return F.relu(self.bn3(self.conv3(out)))


class SR(nn.Layer):
    def __init__(self, in_channel):
        super(SR, self).__init__()
        self.conv1 = nn.Conv2D(in_channel, 192, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(192)
        self.conv2 = nn.Conv2D(192, 192 * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.conv2(out1)
        w, b = out2[:, :192, :, :], out2[:, 192:, :, :]
        return F.relu(w * out1 + b)


class ACFFViTModel(nn.Layer):
    def __init__(self, backbone):
        super(ACFFViTModel, self).__init__()
        self.backbone = backbone

        self.ca = ChannelAttention(1536)
        self.cam = MFR(1536, 1536)
        self.fam4 = CFF(768, 192, 192)
        self.fam3 = CFF(384, 192, 192)
        self.fam2 = CFF(192, 192, 192)

        self.srm2 = SR(192)
        self.srm3 = SR(192)
        self.srm4 = SR(192)

        self.linear2 = nn.Conv2D(192, 1, 3, 1, 1)
        for p in self.backbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0

    def forward(self, x):
        x2, x3, x4, x5 = self.backbone(x)
        x5 = x5 * self.ca(x5)
        x5 = self.cam(x5, x5)
        x4 = self.srm4(self.fam4(x4, x5, x5))
        x3 = self.srm3(self.fam3(x3, x4, x5))
        x2 = self.srm2(self.fam2(x2, x3, x5))

        x2 = F.interpolate(self.linear2(x2), mode='bilinear', size=x.shape[2:], align_corners=True)
        return [x2]
