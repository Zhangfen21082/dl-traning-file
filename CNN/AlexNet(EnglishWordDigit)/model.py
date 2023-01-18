import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
        out_dim：最终分类的数目
        init_weights：设置是否初始化权重，默认为False
    """
    def __init__(self, num_classes, init_weights=False):
        super(AlexNet, self).__init__()

        # 标准AlexNet
        self.conv = nn.Sequential(
            # [224, 224, 3] -> [55, 55, 96]
            # [55, 55, 96] -> [27, 27, 96]
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),  # 可以载入更大的模型
            nn.MaxPool2d(kernel_size=3, stride=2),

            # [27, 27, 96] -> [27, 27, 256]
            # [27, 27, 256] -> [13, 13, 256]
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # [13, 13, 256] -> [13, 13, 384]
            # [13, 13, 384] -> [13, 13, 384]
            # [13, 13, 384] -> [13, 13, 256]
            # [13, 13, 256] -> [6, 6, 256]
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            # 到这里需要使用dropout减少过拟合
            nn.Dropout(p=0.5),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

        # 如果设置了初始化权重，那么就调用对应方法
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        # 进入全连接层前展平
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    # 权重初始化（KaiMing）
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)