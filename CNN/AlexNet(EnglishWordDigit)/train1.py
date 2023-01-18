import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import AlexNet
from tensorboardX import SummaryWriter
import os

# 日志
logger = SummaryWriter('./log/')

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# 数据转换
# torchvision.transforms是pytorch中的图像预处理包。一般用Compose把多个步骤整合到一起
data_transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # 纵横比裁切
            transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定图像
            transforms.ToTensor(),  # 张量转换
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 用均值和标准差归一化图像
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    )
}

# batch_size 大小
batch_size = 64

# 图片路径
image_train_path = r'./dataSet/raw/train'
image_val_path = r'./dataSet/raw/val'

# 加载训练数据和测试数据
train_dataset = datasets.ImageFolder(root=image_train_path, transform=data_transform["train"])
val_dataset = datasets.ImageFolder(root=image_val_path, transform=data_transform["val"])

# loader加载
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True,drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, num_workers=8, pin_memory=True, shuffle=True, drop_last=False)


# 加载网络（分为41类，并设置权重初始化）
net = AlexNet(out_dim=41, init_weights=True)
net.to(device)

# 交叉熵损失函数
loss_function = nn.CrossEntropyLoss()

# 优化器（Adam）
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


def evaluate(net, val_loader, loss_func):
    # 进入eval模式
    correct_num = 0
    net.eval()
    sum_loss = 0
    with torch.no_grad():
        for(X, y) in val_loader:
            X = X.to(device)
            y = y.to(device)
            pred = net(X)
            correct_num += (torch.max(pred, dim=1)[1] == y).sum().item()
            loss = loss_func(pred, y)
            sum_loss += loss.item()
    # 返回train
    net.train()
    print(correct_num / 310)
    return sum_loss / len(val_loader)


# 模型保存
def save_checkpoint(net, epoch, optimizer, checkpoint_path):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(save_dict, checkpoint_path)




def train():
    # 步数
    step = 1
    eval_loss = 0
    net.train()  # 进入train模式
    for epoch in range(0, 30):
        print("-----------当前epoch：{}-----------".format(epoch))
        for i, (X, y) in enumerate(train_loader):
            print("-----------当前batch：{}/{}-----------".format(i,  (3100 // 64)))
            X = X.to(device)
            y = y.to(device)

            pred = net(X)
            loss = loss_function(pred, y)
            # print(pred)
            # print(torch.max(pred, dim=1)[1])
            # print(y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_scalar('loss/train', loss, step)

            # 验证集评估
            if not step % 10:
                eval_loss = evaluate(net, val_loader, loss_function)
                logger.add_scalar('loss/val', eval_loss, step)

            # 模型保存
            if not step % 100:
                model_path = "epoch-{}_step-{}.pth".format(epoch, step)
                save_checkpoint(net, epoch, optimizer, os.path.join('model_save', model_path))


            logger.flush()
            print("当前step：{}；当前train_loss：{:.5f}；当前val_loss：{:.5f}".format(step, loss.item(), eval_loss))
            step += 1

    logger.close()

if __name__ == '__main__':

    train()

    # print(val_dataset.classes)
    # """
    #     数字0-9 ：0 ~ 9
    #     a-z ：10 ~ 35
    #     A-Z：36 ~ 61
    # """
    # print(val_dataset.class_to_idx)
    # # 名字
    # print(val_dataset.imgs)

    # for i, (X, y) in enumerate(train_loader):
    #     X = X.numpy()
    #     y = y.numpy()
    #
    #     for index in range(np.shape(X)[0]):
    #         image_data = X[index]
    #         image_label = y[index]
    #         print(image_label)
    #         # RGB -> BGR
    #         image_data = image_data.transpose(1, 2, 0)
    #         cv2.imshow("image_data", image_data)
    #         cv2.waitKey(0)
    #         time.sleep(100)
