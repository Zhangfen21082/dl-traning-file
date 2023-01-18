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
epochs = 200 # 总训练轮数

# 训练和验证数据加载
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

step = 1
# 训练
for epoch in range(1, epochs):
    print("-----------当前epoch：{}-----------".format(epoch))
    net.train()
    for i, (X, y) in enumerate(train_loader):
        print("\r-----------当前trainbatch：{}/{}-----------".format(i, (len(train_loader))), end="")
        X = X.to(device)
        y = y.to(device)

        outputs = net(X)
        loss = loss_function(outputs, y)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, dim=1)
        correct = pred.eq(y.data).cpu().sum()
        accurate = correct * 100.0 / batch_size
        # print("train_loss: ", loss.item(), "accurate: ", accurate.item())
        logger.add_scalar('loss/train', loss, step)
        logger.add_scalar('accurate/train', accurate, step)

    net.eval()
    sum_loss = 0.
    sum_correct = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            print("\r-----------当前testbatch：{}/{}-----------".format(i, (len(test_loader))), end="")
            X = X.to(device)
            y = y.to(device)
            outputs = net(X)
            _, pred = torch.max(outputs.data, dim=1)
            correct = pred.eq(y.data).cpu().sum()
            loss = loss_function(outputs, y)
            sum_loss += loss.item()
            sum_correct += correct.item()

        test_loss = sum_loss * 1.0 / len(test_loader)
        test_accurate = sum_correct * 100.0 / len(test_loader) / batch_size
        logger.add_scalar('loss/test', test_loss, epoch)
        logger.add_scalar('accurate/test', test_accurate, epoch)
        print("test_loss: ", test_loss.item(), "test_accurate: ", test_accurate.item())
    step += 1
    # 学习率衰减
    scheduler.step()

logger.close()

