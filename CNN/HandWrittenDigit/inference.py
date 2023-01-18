import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CNN
from config import parametes
import preprocess
import cv2



# 实例
cnn = CNN()
# 加载模型
checkpoint = torch.load('./model_save/epoch-9_step-7400.pth', map_location=parametes.device)
# 加载模型参数
cnn.load_state_dict(checkpoint['model_state_dict'])
# 记载测试数据
test_loader = DataLoader(preprocess.test_set, batch_size=1, shuffle=True,
                          drop_last=False)
# 进入eval模式
cnn.eval()

# 预测正确的个数
correct_num = 0

with torch.no_grad():
    for i, (X, y) in enumerate(test_loader):
        X = X.to(parametes.device)
        y = y.to(parametes.device)

        outputs = cnn(X)
        #  pred为这64个图片的预测数字
        _, pred = outputs.max(1)
        correct_num += (pred == y).sum()

        # print(outputs)
        # a, b = outputs.max(1)
        # print(outputs.shape)
        #
        # print(a)
        # print(b)
        #break
        # 打开图片查看
        # tensor转numpy
        X = X.numpy()
        y = y.numpy()
        pred = pred.numpy()

        for index in range(np.shape(X)[0]):
            image_data = X[index]
            image_label = y[index]
            image_pred = pred[index]
            # RGB -> BGR
            image_data = image_data.transpose(1, 2, 0)
            print("此图预测值为{}，真实值为{}".format(image_pred, image_label))
            cv2.imshow("image_data", image_data)
            cv2.waitKey(0)


print("测试数据{}个，正确预测{}个，预测准确率：{}%".format(len(preprocess.test_set), correct_num, (correct_num / len(preprocess.test_set)) * 100))




