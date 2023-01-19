import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

logger = SummaryWriter('./log')

##########################################################参数################################
# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print("GPU上运行")
else:
    print("CPU上运行")
# 图片格式
img_size = [1, 28, 28]

# batchsize
batchsize = 64

# latent_dim
latent_dim = 100

# 数据集及变化
data_transforms = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)
dataset = torchvision.datasets.MNIST(root='~/autodl-tmp/dataset', train=True, download=False, transform=data_transforms)

print(len(dataset))
# 生成器模型
"""
根据输入生成图像
"""


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, np.prod(img_size, dtype=np.int32)),

            nn.Tanh()
        )

    def forward(self, x):
        # [batchsize, latent_dim]
        output = self.model(x)
        image = output.reshape(x.shape[0], *img_size)
        return image


# 判别器模型
"""
判别图像真假
"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(img_size, dtype=np.int32), 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 1),
            nn.ReLU(inplace=True),

            nn.Sigmoid(),
        )

    def forward(self, x):
        # [batch_size, 1, 28, 28]
        x = x.reshape(x.shape[0], -1)
        output = self.model(x)

        return output


# 优化器和损失函数
generator = Generator()
generator = generator.to(device)
discriminator = Discriminator()
discriminator = discriminator.to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
loss_func = nn.BCELoss()


##########################################################训练################################

def train():
    step = 0
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=8)
    for epoch in range(1, 100):
        print("-----------当前epoch：{}-----------".format(epoch))
        for i, batch in enumerate(dataloader):
            print("-----------当前batch：{}/{}-----------".format(i, (len(dataloader))))
            # 拿到真实图片
            X, _ = batch
            X = X.to(device)
            # 采用标准正态分布得到的batchsize × latent_dim的向量
            z = torch.randn(batchsize, latent_dim)
            z = z.to(device)
            # 送入生成器生成假图片
            pred_X = generator(z)

            g_optimizer.zero_grad()
            """
            生成器损失：
            让生成的图像与通过辨别器与torch.ones(batchsize, 1)越接近越好

            """
            g_loss = loss_func(discriminator(pred_X), torch.ones(batchsize, 1).to(device))
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            """
            辨别器损失：
            一方面让真实图片通过辨别器与torch.ones(batchsize, 1)越接近越好
            另一方面让生成图片通过辨别器与torch.zeros(batchsize, 0)越接近越好
            """

            d_loss = 0.5 * (loss_func(discriminator(X), torch.ones(batchsize, 1).to(device)) + loss_func(
                discriminator(pred_X.detach()), torch.zeros(batchsize, 1).to(device)))

            d_loss.backward()
            d_optimizer.step()

            print("生成器损失{}".format(g_loss), "辨别器损失{}".format(d_loss))

            logger.add_scalar('g_loss', g_loss, step)
            logger.add_scalar('d_loss', d_loss, step)

            step = step + 1
            if step % 1000 == 0:
                save_image(pred_X.data[:25], "./image_save/image_{}.png".format(step), nrow=5)


if __name__ == '__main__':
    train()