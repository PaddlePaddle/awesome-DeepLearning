import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 超参数设置
Batch_size = 50
Lr = 0.1
Epoch = 1
# 数据读取
train_dataset = torchvision.datasets.MNIST(
    root="./data/mnist",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    root="./data/mnist",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=Batch_size
)
test_x = torch.unsqueeze(test_dataset.data, dim=1).type(torch.Tensor)
test_y = test_dataset.targets
print(test_y.shape, test_x.shape)

# 构造resent网络
# 网络共八层，分别是1个输入层，1个输出层，三个Block，每个Block中有1个Basicblock，每个Basicblock中有2层layer。
# 考虑到MNIST的数据集的大小为28 x 28，没经过第2个和都3个Block，大小都减半，也就是说，进入输出层时的大小为7 x 7，所以在最后用到的Average_pool的大小为7。
# 因为MNIST的channel数为1，所以in_channel = 1.
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out

# 最终网络结构
ResNet18 = ResNet(Basicblock, [1, 1, 1, 1], 10)
print(ResNet18)

opt = torch.optim.SGD(ResNet18.parameters(), lr=Lr)
loss_fun = nn.CrossEntropyLoss()
a = []
ac_list = []
# 训练
for epoch in range(Epoch):
    for i, (x, y) in enumerate(train_loader):
        output = ResNet18(x)
        loss = loss_fun(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0:
            a.append(i)
            test_output = torch.max(ResNet18(test_x), dim=1)[1]
            loss = loss_fun(ResNet18(test_x), test_y).item()
            accuracy = torch.sum(torch.eq(test_y, test_output)).item() / test_y.numpy().size
            ac_list.append(accuracy)
            print('Epoch:%d' % Epoch, '|loss%.4f' % loss, '|accuracy%.4f' % accuracy)

print('real value', test_y[: 10].numpy())
print('train value', torch.max(ResNet18(test_x)[: 10], dim=1)[1].numpy())

plt.plot(a, ac_list, color='b')
plt.show()