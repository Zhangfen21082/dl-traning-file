import torch
from torch.utils.data import DataLoader
from dataloader import MyDataset
from model import Net
from Parameters import parameters
import numpy as np
import cv2


#  网络实例
model = Net()
#  加载模型：观察tensorboard可知，迭代600次时模型收敛
checkpoint = torch.load('./model_save/epoch86_step6000.pth', map_location=parameters.device)
#  加载模型参数
model.load_state_dict(checkpoint['model_state_dict'])

#  加载测试数据
testset = MyDataset(parameters.metadata_test_path)
test_loader = DataLoader(testset, batch_size=parameters.batch_size, shuffle=True, drop_last=False)

#  预测时，进入eval模式
model.eval()

#  预测正确的个数
correct_num = 0

# json文件
cls_mapper = {
    "clsToid": {"A": 0, "B": 1, "C": 2, "Five": 3, "Point": 4, "V": 5},
    "idTocls": {0: "A", 1: "B", 2: "C", 3: "Five", 4: "Point", 5: "V"}
}

with torch.no_grad():
    for batch in test_loader:
        X, y = batch
        X = X.to(parameters.device)
        y = y.to(parameters.device)
        pred = model(X)
        correct_num += (torch.argmax(pred, 1) == y).sum()

        X = X.numpy()
        y = y.numpy()
        pred = pred.numpy()

        for index in range(np.shape(X)[0]):

            image_data = X[index]
            image_label = y[index]
            image_pred = np.argmax(pred[index])

            # RGB -> BGR
            image_data = image_data.transpose(1, 2, 0)
            print("此图预测值为{}，真实值为{}".format(cls_mapper["idTocls"][image_pred], cls_mapper["idTocls"][image_label]))
            cv2.imshow("image_data", image_data)
            cv2.waitKey(0)

print("测试数据{}个，正确预测{}个，预测准确率：{}%".format(len(testset), correct_num, (correct_num / len(testset)) * 100))

