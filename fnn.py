import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):

    def __init__(self):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(40 * 40, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.Sigmoid()(self.fc1(x))
        x = nn.LogSoftmax(dim=1)(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

