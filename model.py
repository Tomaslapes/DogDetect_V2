import torch.nn as nn
from torch.nn import functional as F
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=10)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=40,kernel_size=10,stride=2)

        self.fc1 = nn.Linear(60840,1024)#40*40*40
        self.fc2 = nn.Linear(1024,128)
        self.out = nn.Linear(128,1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.reshape(-1,60840)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        return x
