import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import torch.optim as opt


class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1, padding_mode='zeros')
        self.BatchNorm1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, padding_mode='zeros')
        self.BatchNorm2 = nn.BatchNorm2d(channels_out)
        self.conv1x1 = nn.Conv2d(self.channels_in, self.channels_out, kernel_size=1, stride=stride)
        self.BatchNorm3 = nn.BatchNorm2d(channels_out)
        # x-> Conv1 -> BatchNorm1 -> ReLU -> Conv2 -> BatchNorm2 -> ReLU -> fx + BatchNorm3(x)

    def forward(self, x):
        fx = self.conv1(x)
        fx = F.relu(self.BatchNorm1(fx))
        fx = self.conv2(fx)
        fx = F.relu(self.BatchNorm2(fx))
        x = self.conv1x1(x)
        x = self.BatchNorm3(x)
        return fx + x


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv2D: channel_in | channel_out | filter_size | stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.BatchNorm = nn.BatchNorm2d(64)  # 64->no of channels
        self.MaxPool = nn.MaxPool2d(3, stride=2)  # (300x300->149x149)
        self.ResBlock1 = ResBlock(64, 64, 1)
        self.ResBlock2 = ResBlock(64, 128, 2)
        self.ResBlock3 = ResBlock(128, 256, 2)
        self.ResBlock4 = ResBlock(256, 512, 2)  # Output -> (batch_size, C:512, H:149, W:149)
        self.GlobAvgPool = None  # Global Average Pooling -> avg each of the 512 feature maps to a single value
        self.FCN = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.BatchNorm(x))
        x = self.MaxPool(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        if self.GlobAvgPool is None:  # MARK: Check if adaptive pooling gives better results
            self.GlobAvgPool = nn.AvgPool2d(x.shape[2])  # filter size equal to feature size
        x = self.GlobAvgPool(x)
        x = torch.flatten(x, start_dim=1)  # dim=0 is the batch dim, so we flatten other dims
        x = self.FCN(x)
        x = torch.sigmoid(x)
        return x

