import torch
import torch.nn as nn
import torch.nn.functional as F

# 基本跳连单元
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        # 主干分支
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )

        # 跳连分支
        self.shortcut = nn.Sequential()
        # 如果输入通道和输出通道不相等或者步长不是1，那么跳连分支必须再进行卷积
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel),
            )
    def forward(self, x):
        # 主干
        out1 = self.layer(x)
        # 分支
        out2 = self.shortcut(x)
        # 最终结果 = 主干+分支
        out = out1 + out2
        out = F.relu(out)

class ResNet(nn.Module):
    def make_layer(self, block, out_channel, stride, num_block):
        layer_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layer_list.append(block(self.in_channel, out_channel, in_stride))
            self.in_channel = out_channel

        return nn.Sequential(*layer_list)

    def __init__(self):
        super(ResNet, self).__init__()
        # 第一层采用普通卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU
        )

        # 四层ResNet
        self.in_channel = 32
        self.layer1 = self.make_layer(ResBlock, 64, 2, 2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, 2)

        self.fc = nn.Linear(512, 10)


    def farward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out