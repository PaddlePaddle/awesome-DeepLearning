import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

batch_size = 64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_data = datasets.MNIST('MNIST', train=True, transform=transform)
test_data = datasets.MNIST('MNIST', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


class InceptionA(torch.nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, 1)

        self.branch3x3_2 = torch.nn.Conv2d(16, 24, 3, padding=1)

        self.branch3x3_3 = torch.nn.Conv2d(24, 24, 3, padding=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, 1)

        self.branch5x5_2 = torch.nn.Conv2d(16, 24, 5, padding=2)

        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, 1)

        self.branch_pool = torch.nn.Conv2d(in_channels, 24, 1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)

        branch3x3 = self.branch3x3_2(branch3x3)

        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)

        branch5x5 = self.branch5x5_2(branch5x5)

        branch1x1 = self.branch1x1(x)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]

        return torch.cat(outputs, dim=1)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 10, 5)

        self.conv2 = torch.nn.Conv2d(88, 20, 5)

        self.incep1 = InceptionA(in_channels=10)

        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(2)

        self.fc = torch.nn.Linear(2200, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))

        x = self.incep1(x)

        x = F.relu(self.mp(self.conv2(x)))

        x = self.incep2(x)

        x = x.view(-1, 2200)

        x = self.fc(x)

        return x


device = torch.device('cuda:0')

model = Net()
model.to(device)

lr = 0.01

momentum = 0.5

log_interval = 10

epochs = 10

test_batch_size = 1000


def train(epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data = torch.tensor(data).type(torch.FloatTensor).cuda()

        target = torch.tensor(target).type(torch.LongTensor).cuda()

        torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum).zero_grad()

        output = model(data)

        loss = nn.CrossEntropyLoss()(output, target)

        loss.backward()

        torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum).step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()

    test_loss = 0

    correct = 0

    for data, target in test_loader:
        data = torch.tensor(data).type(torch.FloatTensor).cuda()

        target = torch.tensor(target).type(torch.LongTensor).cuda()

        output = model(data)

        loss = nn.CrossEntropyLoss()(output, target)

        test_loss += loss.item() * target.size(0)

        pred = torch.max(output, 1)[1]

        correct += (pred == target.cuda()).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {} / {}({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    for epoch in range(1):
        train(epoch)

    test()

















