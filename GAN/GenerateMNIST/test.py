import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
batchsize = 16

# latent_dim
latent_dim = 64

# 数据集及变化
data_transforms = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)
dataset = torchvision.datasets.MNIST(root='./data/', train=True, download=False, transform=data_transforms)

# 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, torch.prob(img_size, dtype=torch.int32)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [batchsize, latent_dim]
        output = self.model(x)
        image = output.reshape(x.shape[0], *img_size)
        return image

# 判别器模型
class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(torch.prob(img_size, dtype=torch.int32), 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 1),
            nn.ReLU(inplace=True),

            nn.Sigmoid(),
        )

    def forward(self, x):
        # [batch_size, 1, 28, 28]
        x = x.reshape(x.shape[0], -1)
        output = self.model(x)

        return output

generator = Generator()
discriminator = Discriminator()
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
loss_func = nn.BCELoss()

##########################################################训练################################
dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, drop_last=True)
for epoch in range(100):
    for i, batch in enumerate(dataloader):
        # 拿到真实图片
        X, _ = batch
        X = X.to(device)
        # 生成噪声
        z = torch.randn(batchsize, latent_dim)
        # 送入生成器生成假图片
        pred_X = generator(z)

        # 送入辨别器生成器损失
        g_loss = loss_func(discriminator(pred_X), torch.ones(batchsize, 1))
        g_optimizer.zero_grad()
        g_loss.backward()
        g_loss.step()


        d_loss = 0.5 * (loss_func(discriminator[X], torch.ones(batchsize, 1)) + loss_func(discriminator(pred_X.detach()), torch.zeros(64, 1)))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_loss.step()

        if i % 1000 == 0:
            for index, image in enumerate(pred_X):
                torchvision.utils.save_image(image, "./image_save/image_{}".format(index))



