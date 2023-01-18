import os
import torch
import random
import numpy as np
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.utils.data import DataLoader


from Parameters import parameters
from dataloader import MyDataset
from model import Net


# 日志记录
logger = SummaryWriter('./log')

# 随机种子
torch.manual_seed(parameters.seed)  # CPU随机种子
if parameters.device == 'cuda':
    torch.cuda.manual_seed(parameters.seed)  # GPU随机种子（若有）
random.seed(parameters.seed)  # random随机种子
np.random.seed(parameters.seed)  # numpy随机种子

# 使用验证集对模型进行评估
def evaluate(model, val_loader, loss_func):
    #  进入eval模式
    model.eval()
    sum_loss = 0.
    #  with torch.no_grad()含义：https://blog.csdn.net/qq_42251157/article/details/124101436
    with torch.no_grad():
        for batch in val_loader:
            X, y = batch
            X = X.to(parameters.device)
            y = y.to(parameters.device)
            pred = model(X)
            loss = loss_func(pred, y)
            sum_loss += loss.item()
    #  特别注意返回train模式
    model.train()
    return sum_loss / len(val_loader)

# 保存模型
def save_checkpoint(model, epoch, optimizer, checkpoint_path):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(save_dict, checkpoint_path)



# 训练函数
def train():
    # 有关argparse.ArgumentParser用法：https://blog.csdn.net/u011913417/article/details/109047850
    # 其作用是解析命令行参数，目的是在终端窗口(ubuntu是终端窗口，windows是命令行窗口)输入训练的参数和选项
    parser = ArgumentParser(description='Model Training')
    parser.add_argument(
        '--c',
        # 当模型再次训练时选择从头开始还是从上次停止的地方开始
        default=None,  # 当参数未在命令行中出现时使用的值
        type=str,  # 参数类型
        help='from head or last checkpoint?'  # 参数说明
    )
    args = parser.parse_args()

    #  模型实例
    model = Net()
    model = model.to(parameters.device)

    # 损失函数（这里比较简单所以直接定义，否则需要新建文件loss.py存放）
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), parameters.init_lr)

    # 训练数据加载
    trainset = MyDataset(parameters.metadata_train_path)

    train_loader = DataLoader(trainset, batch_size=parameters.batch_size, shuffle=True, drop_last=True)
    # 验证数据加载（在evaluation函数中进行评估）
    valset = MyDataset(parameters.metadata_eval_path)
    val_loader = DataLoader(valset, batch_size=parameters.batch_size, shuffle=True, drop_last=False)

    # 起始训练轮数， 步数
    start_epoch, step = 0, 0

    # 判断参数，是否需要从检查点开始训练
    # 主要针对大型数据，可能会训练几个小时或几天，所以容易出现问题
    if args.c:
        checkpoint = torch.load(args.c)  # 加载模型
        #  加载参数（权重系数、偏置值、梯度等等）
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("参数加载成功")

    else:
        print("从头开始训练")

    # 关于model.train()说明：https://blog.csdn.net/weixin_44211968/article/details/123774649
    model.train()  # 启用 batch normalization 和 dropout

    # 训练过程
    for epoch in range(start_epoch, parameters.epochs):
        print("-----------当前epoch：{}-----------".format(epoch))
        for i, batch in enumerate(train_loader):
            print("-----------当前batch：{}/{}-----------".format(i, len(train_loader)))
            X, y = batch
            X = X.to(parameters.device)
            y = y.to(parameters.device)
            pred = model(X)
            loss = loss_func(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar('loss/train', loss, step)

            # 每10步进行验证集评估并保存
            if not step % parameters.verbose_step:
                eval_loss = evaluate(model, val_loader, loss_func)
                logger.add_scalar('loss/val', eval_loss, step)
            if not step % parameters.save_step:
                model_path = "epoch-{}_step-{}.pth".format(epoch, step)
                save_checkpoint(model, epoch, optimizer, os.path.join('movel_save', model_path))

            step += 1
            logger.flush()
            print("当前step：{}；当前train_loss：{:.5f}；当前val_loss：{:.5f}".format(step, loss.item(), eval_loss))
    logger.close()

if __name__ == '__main__':
    train()

