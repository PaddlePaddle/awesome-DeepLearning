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


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )

        self.fc1 = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


device = torch.device('cuda:0')

model = LeNet()
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




