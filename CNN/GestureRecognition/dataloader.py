import torch
from torch.utils.data import DataLoader
from Parameters import parameters
from preporcess import load_meta, load_image
from torchvision import transforms

transform_train = transforms.Compose(
    [
        transforms.Resize((112, 112)),  # 保证输入图像大小为112×112
        transforms.RandomRotation(degrees=45),  # 减小倾斜图片影像
        transforms.GaussianBlur(kernel_size=(3, 3)),  # 抑制模糊图片影响
        transforms.RandomHorizontalFlip(),  # 左右手
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Resize((112, 112)),  # 保证输入图像大小为112×112
        # transforms.RandomRotation(degrees=45),  # 减小倾斜图片影像
        # transforms.GaussianBlur(kernel_size=(3, 3)),  # 抑制模糊图片影响
        # transforms.RandomHorizontalFlip(),  # 左右手
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
    ]
)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        self.dataset = load_meta(metadata_path)  # [(0, image_path), (), ...]
        self.metadata_path = metadata_path

    def __getitem__(self, idx):
        item = self.dataset[idx]
        cls_id, path = int(item[0]), item[1]
        img = load_image(path)

        if self.metadata_path == parameters.metadata_train_path or self.metadata_path == parameters.metadata_eval_path:
            return transform_train(img), cls_id

        # 对于测试集不需要数据增强
        return transform_test(img), cls_id
    def __len__(self):
        return len(self.dataset)