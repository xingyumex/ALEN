import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


class Align(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Align, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        return self.conv(x)

class CoreNet(nn.Module):
    def __init__(self):
        super(CoreNet, self).__init__()

        self.initial = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.final = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.adjust1 = Align(32, 64)
        self.adjust2 = Align(64, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_initial = self.initial(x)
        x1 = self.relu(self.conv1(x_initial))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2 + self.adjust1(x1)))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4 + self.adjust2(x3)))
        x6 = self.relu(self.conv3(x5))
        x7 = self.relu(self.conv6(x6 + x5))  
        x8 = self.relu(self.conv1(x7))
        x_final = self.final(x8 + x7)

        return x_final

class McNet(nn.Module):
    def __init__(self):
        super(McNet, self).__init__()
        self.channel1 = CoreNet()
        self.channel2 = CoreNet()
        self.channel3 = CoreNet()
        self.final_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        r = self.channel1(x[:, 0:1, :, :])
        g = self.channel2(x[:, 1:2, :, :])
        b = self.channel3(x[:, 2:3, :, :])
        combined = torch.cat([r, g, b], dim=1)
        x = self.sigmoid(self.final_conv(combined))
        return x
    