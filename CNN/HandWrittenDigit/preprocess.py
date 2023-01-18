from torch.utils.data import sampler

from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms

#  按照80%的训练数据和20%的验证数据拆分原始训练数据，得到sampler
split_num = int(60000 * 0.8)
index_list = list(range(60000))
train_index, val_index = index_list[:split_num], index_list[split_num:]

train_sampler = sampler.SubsetRandomSampler(train_index)
val_sampler = sampler.SubsetRandomSampler(val_index)


#  原始数据训练数据
train_set = dataset.MNIST(
    root='./data/',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)
#  测试集
test_set = dataset.MNIST(
    root='./data/',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)


if __name__ == '__main__':


    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=1, drop_last=True)
    val_loader = DataLoader(train_set, sampler=val_sampler, batch_size=1, drop_last=False)
    print(len(train_loader))
    print(len(val_loader))
    # for batch in train_loader:
    #     X, y = batch
    #     print(X)
    #     print(y)
