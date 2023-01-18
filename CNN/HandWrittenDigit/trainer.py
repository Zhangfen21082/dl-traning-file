import os
import torch
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from config import parametes
from model import CNN
import preprocess

# 日志
logger = SummaryWriter('./log')

# 验证集评估
def evaluate(cnn, val_loader, loss_func):
    # 进入eval模式
    cnn.eval()
    sum_loss = 0.
    with torch.no_grad():
        for (X, y) in val_loader:
            X = X.to(parametes.device)
            y = y.to(parametes.device)
            pred = cnn(X)
            loss = loss_func(pred, y)
            sum_loss += loss.item()
    # 返回train模式
    cnn.train()
    return sum_loss / len(val_loader)


# 保存模型
def save_checkpoint(cnn, epoch, optimizer, checkpoint_path):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': cnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(save_dict, checkpoint_path)

# 训练函数
def train():
    # 命令行参数
    parser = ArgumentParser(description='Model Training')
    parser.add_argument(
        '--c',
        default=None,
        type=str,
        help='from head or given model?'
    )
    args = parser.parse_args()

    # 模型实例
    cnn = CNN()
    cnn = cnn.to(parametes.device)

    # 损失函数：交叉熵
    loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 优化器
    optimizer = torch.optim.Adam(cnn.parameters(), lr=parametes.init_lr)

    # 训练集、验证集加载
    train_loader = DataLoader(preprocess.train_set, sampler=preprocess.train_sampler, batch_size=parametes.batch_size, drop_last=True)
    val_loader = DataLoader(preprocess.train_set, sampler=preprocess.val_sampler, batch_size=parametes.batch_size, drop_last=False)

    # 起始训练轮数，步数
    start_epoch, step = 0, 0

    # 参数判断
    if args.c:
        checkpoint = torch.load(args.c)
        # 参数加载
        start_epoch = checkpoint['epoch']
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("参数加载成功")
    else:
        print("从头开始训练")

    # 进入train()模式
    cnn.train()

    for epoch in range(start_epoch, parametes.epochs):
        print("-----------当前epoch：{}-----------".format(epoch))
        for i, (X, y) in enumerate(train_loader):
            print("-----------当前batch：{}/{}-----------".format(i, (48000 // parametes.batch_size)))
            X = X.to(parametes.device)
            y = y.to(parametes.device)

            pred = cnn(X)
            loss = loss_func(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.add_scalar('loss/train', loss, step)

            # 验证集评估
            if not step % parametes.verbose_step:
                eval_loss = evaluate(cnn, val_loader, loss_func)
                logger.add_scalar('loss/val', eval_loss, step)
            # 模型保存
            if not step % parametes.save_step:
                model_path = "epoch-{}_step-{}.pth".format(epoch, step)
                save_checkpoint(cnn, epoch, optimizer, os.path.join('model_save', model_path))
            step += 1
            logger.flush()
            print("当前step：{}；当前train_loss：{:.5f}；当前val_loss：{:.5f}".format(step, loss.item(), eval_loss))
    logger.close()
if __name__ == '__main__':
    train()
