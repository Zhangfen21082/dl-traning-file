import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter

from VGG13 import VGG13
from dataloader import train_dataset, test_dataset


######################## 参数指定 ###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
batch_size = 64  # batch_size大小
epochs = 20 # 总训练轮数

# 数据集加载
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 网络结构指定
net = VGG13()
net = net.to(device)

# 损失函数（交叉熵）
loss_function = nn.CrossEntropyLoss()

# 优化器（Adam）
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# 日志
logger = SummaryWriter('./log')

# 训练开始

step = 1
for epoch in range(1, epochs+1):
    print("第{}轮epoch开始".format(epoch))
    # 训练
    net.train()
    train_loss = 0.0
    for batch, (train_images, train_labels) in enumerate(train_loader):
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)
        outputs = net(train_images)
        loss1 = loss_function(outputs, train_labels)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        train_loss += loss1.item()
        logger.add_scalar('loss/train', loss1, step)


        # 训练集batch进度
        rate = (batch + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss1), end="")
    print()


    # 验证
    net.eval()
    sum_loss = 0.0
    correct_num = 0
    with torch.no_grad():
        for batch, (val_images, val_labels) in enumerate(test_loader):
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            outputs = net(val_images)
            loss2 = loss_function(outputs, val_labels)
            sum_loss += loss2.item()

            predict_label = torch.max(outputs, dim=1)[1]
            correct_num += (predict_label == val_labels).sum().item()

            # 测试集batch进度
            rate = (batch + 1) / len(test_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rval loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss2), end="")
        print()

        test_loss = sum_loss / len(test_loader)
        test_accurate = correct_num / len(test_dataset)
        logger.add_scalar('loss/test', test_loss, epoch)
        logger.add_scalar('accurate/test', test_accurate, epoch)

        torch.save(net.state_dict(), './model_save_epoch{}'.format(epoch))


        print("第{}轮epoch结束;测试集平均损失{};验证集平均损失{};正确率{}".format(epoch, train_loss / len(train_loader), test_loss, test_accurate))

print("训练完成")

