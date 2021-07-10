import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, out_channels)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 16 * 5 * 5)
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        out = F.log_softmax(out)

        return out
