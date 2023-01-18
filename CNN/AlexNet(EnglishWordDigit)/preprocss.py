import os
import random
import shutil
import json


# 数据集文件夹
data_path = r'E:\Postgraduate\DeepLearning\traing program\deep learning\CNN\coder\AlexNet(EnglishWordDigit)\dataSet\raw'
# 数据集文件夹中的图片文件夹
img_folder_path = r'E:\Postgraduate\DeepLearning\traing program\deep learning\CNN\coder\AlexNet(EnglishWordDigit)\dataSet\raw\Hnd\Img'

# ASCII
# 'a':97
# 'A':65
print(chr(11+54))
print(chr(37+60))

# 创建文件夹
def make_dir():
    for i in range(1, 63):
        # 创建大写字母A-Z文件夹
        if i >= 11 and i < 37:
            os.mkdir(os.path.join(data_path, 'train', 'upper_' + chr(i+54)))
            os.mkdir(os.path.join(data_path, 'val', 'upper_' + chr(i+54)))

        # 创建小写字母a-z文件夹
        if i >= 37:
            os.mkdir(os.path.join(data_path, 'train', 'lower_' + chr(i + 60)))
            os.mkdir(os.path.join(data_path, 'val', 'lower_' + chr(i + 60)))

# 放置
def copy():
    """
    Sample001 - Sample010 : 数字0-9
    Sample011 - Sample036：大写字母A-Z
    Sample037 - Sample062：小写字母a-z
    """
    # 所有的Sample文件夹的列表
    Img_folder = os.listdir(img_folder_path)

    # 训练集图片数目
    img_to_train_num = 54
    # print(Img_folder)

    for i in range(1, 63):
        # 处理 Sample011 - Sample036：大写字母A-Z
        if i >= 11 and i < 37:
            # 每个Sample下图片名的列表
            Sample_path = os.listdir(os.path.join(img_folder_path, Img_folder[i - 1]))
            # 随机抽取50张训练图片
            sample_train = random.sample(Sample_path, img_to_train_num)
            # 剩余5张做验证图片
            sample_val = list(set(Sample_path).difference(sample_train))  # 所有图片减去训练图片就是验证图片
            # 开始复制
            for img_name in sample_train:
                src = os.path.join(img_folder_path, Img_folder[i-1], img_name)
                tar = os.path.join(data_path, 'train', r'upper_' + chr(i + 54), img_name)
                shutil.copy(src, tar)
            for img_name in sample_val:
                src = os.path.join(img_folder_path, Img_folder[i-1], img_name)
                tar = os.path.join(data_path, 'val', r'upper_' + chr(i + 54), img_name)
                shutil.copy(src, tar)

        # 处理 Sample037 - Sample062：小写字母a-z
        if i >= 37:
            # 每个Sample下图片名的列表
            Sample_path = os.listdir(os.path.join(img_folder_path, Img_folder[i - 1]))
            # 随机抽取50张训练图片
            sample_train = random.sample(Sample_path, img_to_train_num)
            # 剩余5张做验证图片
            sample_val = list(set(Sample_path).difference(sample_train))  # 所有图片减去训练图片就是验证图片
            # 开始复制
            for img_name in sample_train:
                src = os.path.join(img_folder_path, Img_folder[i-1], img_name)
                tar = os.path.join(data_path, 'train', r'lower_' + chr(i+ 60), img_name)
                shutil.copy(src, tar)
            for img_name in sample_val:
                src = os.path.join(img_folder_path, Img_folder[i-1], img_name)
                tar = os.path.join(data_path, 'val', r'lower_' + chr(i + 60), img_name)
                shutil.copy(src, tar)


def jsons():
    letter_dict = {}
    for idx, name in enumerate(os.listdir('./dataSet/raw/train')):
        # if 'lower' in name:
        #     letter_dict[idx]=chr(ord(str(name).split("_")[-1])+32)
        # else:
        #     letter_dict[idx]=str(name).split("_")[-1]
        if idx >= 10 and idx < 36:
            letter_dict[idx] = chr(idx + 55)
        if idx >= 36:
            letter_dict[idx] = chr(idx + 61)

    with open("index.json", "w")as f:
        json.dump(letter_dict, f)

if __name__ == '__main__':
   # make_dir()
   #copy()
   jsons()