import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import glob


# 类别名字
label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]
# 类比名字映射索引
label_dict = {}
for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

class MyDataset(Dataset):
    """
        im_list：是一个列表，每一个元素是图片路径
        transform：对图片进行增强
        loader：使用PIL对图片进行加载
    """
    def __init__(self, im_list, transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        # imgs为二维列表，每一个子列表中第一个元素存储im_list，第二个通过label_dict映射为索引
        imgs = []

        for im_item in im_list:
            # 路径'./data/test/airplane/aeroplane_s_000002.png'中倒数第二个是标签名
            im_label_name = im_item.split("\\")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        im__path, im_label = self.imgs[index]

        # 会调用PIL加载图片数据
        im_data = self.loader(im__path)
        # 如果给了transoform那么就对图片进行增强
        if self.transform is not None:
            im_data = self.transform(im_data)

        return im_data, im_label

    def __len__(self):
        return len(self.imgs)

im_train_list = glob.glob(r'./data/train/*/*.png')
im_test_list = glob.glob(r'./data/test/*/*.png')
train_dataset = MyDataset(im_train_list, transform=train_transforms)
test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())

if __name__ == '__main__':
    im_train_list = glob.glob(r'./data/train/*/*.png')
    im_test_list = glob.glob(r'./data/test/*/*.png')

    train_dataset = MyDataset(im_train_list, transform=train_transforms)
    test_dataset = MyDataset(im_test_list, transform=transforms.ToTensor())
    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=False, num_workers=0)

    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomGrayscale(0.1),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root='./data/test', transform=transforms.ToTensor)
    print(train_dataset.classes[: 5])
    print("-"*30)
    print(train_dataset.class_to_idx)
    print("-"*30)
    print(train_dataset.imgs[: 5])
    
    """