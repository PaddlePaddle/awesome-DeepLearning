# 导入库
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.nn import functional as F
from math import floor, ceil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# In[1] 设置超参数
num_epochs = 60
batch_size = 100
learning_rate = 0.001

# 获取数据：训练数据和测试数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)


# 定义卷积核
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


# 义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 下采样
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 搭建残差神经网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # 构建残差块,恒等映射
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        # 不构建残差块,进行了下采样
        # layers中记录的是数字,表示对应位置的残差块数目
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        # 不构建残差块,进行了下采样
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(3136, 128)
        self.normfc12 = nn.LayerNorm((128), eps=1e-5)
        self.fc2 = nn.Linear(128, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        # blocks是残差块的数目
        # 由于输出尺寸会改变,用make_layers去生成一大块对应尺寸完整网络结构
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        # layer1是三块in_channels等于16的网络结构，包括三个恒等映射
        out = self.layer1(out)
        # layer2包括了16->32下采样,然后是32的三个恒等映射
        out = self.layer2(out)
        # layer3包括了32->64的下采样,然后是64的三个恒等映射
        out = self.layer3(out)
        # 全连接压缩
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.normfc12(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 定义模型和损失函数
# [2,2,2]:不同in_channels下的恒等映射数目
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 设置一个通过优化器更新学习率的函数
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 定义测试函数
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# 训练模型更新学习率
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    in_epoch = time.time()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化e
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    test(model, test_loader)
    out_epoch = time.time()
    print(f"use {(out_epoch - in_epoch) // 60}min{(out_epoch - in_epoch) % 60}s")
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
# 测试模型并保存
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'resnet.ckpt')
