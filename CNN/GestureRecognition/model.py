import torch
import torch.nn as nn
import torch.nn.functional as F

from Parameters import  parameters

# 定义mish激活函数
def mish(x):
    return x * torch.tanh(F.softplus(x))

# 封装mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


# 深度可分离卷积
class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSConv2d, self).__init__()
        # 保证kernel_size必须是奇数
        assert kernel_size % 2 == 1, "kernel_size必须为奇数"
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size//2, kernel_size//2),
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, input_x):
        out = self.depth_conv(input_x)
        out  = self.pointwise_conv(out)

        return out

# 编写MTB模块（残差网络）
class MTB(nn.Module):
    def __init__(self, in_channels):
        super(MTB, self).__init__()

        self.left_flow = nn.Sequential(
            # 点卷积
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            Mish(),
            # 深度可分离卷积
            DSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            Mish(),
            # 7×7卷积
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=(7//2, 7//2)),
        )

        self.right_flow = nn.Sequential(
            # 7×7卷积
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=(7 // 2, 7 // 2)),
            nn.BatchNorm2d(in_channels),
            Mish(),
            # 深度可分离卷积
            DSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            Mish(),
            # 点卷积
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
        )

    def forward(self, input_ft):
        left = self.left_flow(input_ft)
        right = self.right_flow(input_ft)
        out = left + right + input_ft

        out = mish(out)
        return out


# [N, 3, 112, 112] -> [N, 256, 7, 7]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=parameters.data_channels, out_channels=64, kernel_size=3, padding=(3//2, 3//2)),
            nn.BatchNorm2d(64),
            Mish(),
            # MTB模块
            MTB(in_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(3 // 2, 3 // 2)),
            nn.BatchNorm2d(128),
            Mish(),
            # MTB模块
            MTB(in_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=(3 // 2, 3 // 2)),
            nn.BatchNorm2d(256),
            Mish(),
            # MTB模块
            MTB(in_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            MTB(in_channels=256),
            MTB(in_channels=256),
            MTB(in_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=256*7*7, out_features=2048),
            Mish(),
            nn.Dropout(parameters.fc_dropout_prob),

            nn.Linear(in_features=2048, out_features=1024),
            Mish(),
            nn.Dropout(parameters.fc_dropout_prob),

            nn.Linear(in_features=1024, out_features=parameters.classes_num)
        )

    def forward(self, input_x):
        out = self.conv(input_x)  # [N, 256, 7, 7]
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)

        return out


# 验证
if __name__ == '__main__':
    net = Net()
    x = torch.randn(size=(5, 3, 112, 112))
    y_pred = net(x)
    print(y_pred.size())
