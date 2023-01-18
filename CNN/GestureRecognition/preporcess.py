import os
import json
from Parameters import parameters
import random
from PIL import Image

random.seed(parameters.seed)


# 获取某个文件夹下面所有后缀名为suffix的文件，并返回其path的list
def recursive_fetching(root, suffix):
    all_file_path = []

    # get_all_files函数会被递归调用
    def get_all_files(path):
        all_file_list = os.listdir(path)
        # 遍历path文件夹下的所有文件和目录
        for file in all_file_list:
            filepath = os.path.join(path, file)
            # 如果是目录则再次递归
            if os.path.isdir(filepath):
                get_all_files(filepath)
            # 如果是文件则保存其文件路径和文件名到all_file_path中
            elif os.path.isfile(filepath):
                all_file_path.append(filepath)

    # 把根目录传入
    get_all_files(root)
    # 筛选所有后缀名为suffix的文件
    file_paths = [it for it in all_file_path if os.path.split(it)[-1].split('.')[-1].lower() in suffix]

    return file_paths

# 加载meta文件
def load_meta(meta_path):
    with open(meta_path, 'r') as fr:
        return [line.strip().split('|') for line in fr.readlines()]

# 加载图片
def load_image(image_path):
    return Image.open(image_path)

# 构建类别到id的映射
cls_mapper = {
    "clsToid": {"A": 0, "B": 1, "C": 2, "Five": 3, "Point": 4, "V": 5},
    "idTocls": {0: "A", 1: "B", 2: "C", 3: "Five", 4: "Point", 5: "V"}
}

if not os.path.exists(parameters.cls_mapper_path):
    json.dump(cls_mapper, open(parameters.cls_mapper_path, 'w'))

train_items = recursive_fetching(parameters.train_data_root, 'ppm')  # 获取Marcel-Train文件夹下数据路径
test_items = recursive_fetching(parameters.test_data_root, 'ppm')  # 获取Marcel-Test文件夹下数据路径
dataset = train_items + test_items  # 合并
random.shuffle(dataset)  # 打乱数据集
dataset_num = len(dataset)
print("数据集总数目：", dataset_num)



"""
    最终dataset_dict大概长这样子
    dataset_dict = {
    0: ["./data/Marcel-Test/A/complex/A-complex32.ppm", "./data/Marcel-Test/A/complex/A-complex31.ppm", ...]
    1: ["./data/Marcel-Train/B/B-train119.ppm", "./data/Marcel-Test/B/uniform/B-uniform04.ppm", ...]
    ...
    5: [...]
}
"""
dataset_dict = {}
for it in dataset:
    # 例如"./data/Marcel-Train/B/B-train119.ppm"，cls_name就是B, cls_id就是1
    cls_name = os.path.split(it)[-1].split('-')[0]
    cls_id = cls_mapper["clsToid"][cls_name]
    # 例如，把所有属于B的训练数据和图片数据放到一个列表中，该列表的k值为1
    if cls_id not in dataset_dict:
        dataset_dict[cls_id] = [it]
    else:
        dataset_dict[cls_id].append(it)

# 每个列表按照比例分配到train、eval、test中
train_ratio, eval_ratio, test_ratio = 0.8, 0.1, 0.1
train_set, eval_set, test_set = [], [], []

for idx, set_list in dataset_dict.items():
    length = len(set_list)
    train_num, eval_num = int(length*train_ratio), int(length*eval_ratio)
    test_num = length - train_num - eval_num
    random.shuffle(set_list)
    train_set.extend(set_list[:train_num])
    eval_set.extend(set_list[train_num:train_num+eval_num])
    test_set.extend(set_list[train_num+eval_num:])

random.shuffle(train_set)
random.shuffle(eval_set)
random.shuffle(test_set)
# print(train_set)
# print(eval_set)
# print(test_set)
# print(len(train_set) + len(eval_set) + len(test_set))

# 写入metafile

with open(parameters.metadata_train_path, 'w') as fw:
    for path in train_set:
        cls_name = os.path.split(path)[-1].split('-')[0]
        cls_id = cls_mapper["clsToid"][cls_name]
        fw.write("%d|%s\n" % (cls_id, path))


with open(parameters.metadata_eval_path, 'w') as fw:
    for path in eval_set:
        cls_name = os.path.split(path)[-1].split('-')[0]
        cls_id = cls_mapper["clsToid"][cls_name]
        fw.write("%d|%s\n" % (cls_id, path))

with open(parameters.metadata_test_path, 'w') as fw:
    for path in test_set:
        cls_name = os.path.split(path)[-1].split('-')[0]
        cls_id = cls_mapper["clsToid"][cls_name]
        fw.write("%d|%s\n" % (cls_id, path))



# 测试，看一下所有图片的颜色模式和对应大小
mode_set, size_set = [], []
for _, path in load_meta(parameters.metadata_train_path):
    img = load_image(path)
    mode_set.append(img.mode)
    size_set.append(img.size)


print(set(mode_set), set(size_set))
