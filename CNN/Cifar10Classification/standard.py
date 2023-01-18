import torch.nn as nn
from torchvision import models


# class resNet18(nn.Module):
#     def __init__(self):
#         super(resNet18, self).__init__()
#         """
#         pretrained为True会返回在ImageNet上预训练过的模型，如果为False表示只需要模型
#         pregress为True在下载模型时会通过标准错误流输出进度条
#         """
#         self.model = models.resnet18(False)
#         self.model.ad


resnet = models.resnet18()
print(resnet)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
print("-------------------------------------------")
print(resnet)

