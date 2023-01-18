import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNet(nn.Module):

    # 深度可分离卷积由分组卷积和点卷积构成（基本单元）
    def conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            # 分组卷积
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel,
                      bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            # 点卷积
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def __init__(self):
        super(MobileNet, self).__init__()
        # 标准卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 深度可分离卷积
        self.convdw2 = self.conv_dw(32, 32, 1)
        self.convdw3 = self.conv_dw(32, 64, 2)

        self.convdw4 = self.conv_dw(64, 64, 1)
        self.convdw5 = self.conv_dw(64, 128, 2)

        self.convdw6 = self.conv_dw(128, 128, 1)
        self.convdw7 = self.conv_dw(128, 256, 2)

        self.convdw8 = self.conv_dw(256, 256, 1)
        self.convdw9 = self.conv_dw(256, 512, 2)

        # 全连接层
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.convdw2(out)
        out = self.convdw3(out)
        out = self.convdw4(out)
        out = self.convdw5(out)
        out = self.convdw7(out)
        out = self.convdw8(out)
        out = self.convdw9(out)

        out =  F.avg_pool2d(out, 2)
        out = out.view(-1, 512)
        out = self.fc(out)

        return out


