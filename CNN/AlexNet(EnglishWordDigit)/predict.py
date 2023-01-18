import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import json


def get_predict():
    data_transform = transforms.Compose( # 数据转换模型
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open("test.png") # 加载图片，自定义的图片名称
    img = data_transform(img) # 图片转换为矩阵
    # 对数据维度进行扩充
    img = torch.unsqueeze(img, dim=0)
    # 创建模型
    model = AlexNet(num_classes=41)
    # 加载模型权重
    model_weight_path = "./AlexNet.pth" #与train.py里的文件名对应
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'), strict=False)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img)) # 图片压缩
        predict = torch.softmax(output, dim=0) # 求softmax值
        predict_cla = torch.argmax(predict).numpy() # 预测分类结果
        with open("index.json","r")as f:
            data = json.load(f)
        print("预测结果为",data[str(predict_cla)])
        return data[str(predict_cla)]

