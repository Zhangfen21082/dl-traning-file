import torch
from config import parametes
from preprocess import train_sampler

#  测试用
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层
        self.conv = torch.nn.Sequential(
            #  卷积运算
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            #  Batch Normalization
            torch.nn.BatchNorm2d(32),
            #  激活函数
            torch.nn.ReLU(),
            # 池化
            torch.nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc = torch.nn.Linear(14 * 14 * 32, parametes.out_dim)

    def forward(self, input_x):
        # 通过卷积层
        out = self.conv(input_x)
        #print(out.shape)
        out = out.view(out.size()[0], -1)  # 进入全连接层之前将数据拉直
        #print(out.shape)
        out = self.fc(out)

        return out


if __name__ == '__main__':
    cnn = CNN()
    cnn = cnn.to(parametes.device)

    train_set_raw = dataset.MNIST(
        root='./data/',
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )

    train_loader = DataLoader(train_set_raw, sampler=train_sampler, batch_size=64, drop_last=True)
    for i, (X, y) in enumerate(train_loader):
        X = X.to(parametes.device)
        y = y.to(parametes.device)

        pred = cnn(X)
        a, b = pred.max(1)
        print(a)
        print(b)
        print(pred.size())
        break




