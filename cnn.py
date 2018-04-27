import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 20, 300)
        self.sg1 = nn.Sigmoid()
        self.fc2 = nn.Linear(300, 10)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, 8 * 8 * 20)
        x = self.sg1(self.fc1(x))
        x = self.lsm(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
