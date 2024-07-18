import torch
from torch import nn
import numpy as np
from Data_cnn_TWK0 import ShuJu
from model_ICM_MLP import MLP_Mixer

n = 80
high_size, wide0, wide1, wide2, hidden_dim, num_classes = 1, 31, 6, 12, 48, 1  # wide:特征数, hidden_dim:隐藏层尺寸, n是网络层数

# 导入数据集
train_loader, test_loader, train_set, test_set = ShuJu(1)

# 加载已训练好的模型
net = MLP_Mixer(high_size, wide0, wide1, wide2, hidden_dim, num_classes, n)
net.load_state_dict(torch.load('E:/Deel/DM/WeriKuang-Mixer_MLP/WK-Mixer_MLP1/MLP-Mixer_relative/pth/ICM_MLP.pth'))
net.eval()

output = []
output1 = []
# 输入数据预处理
for data in train_loader:
    imgs, targets = data
    # 进行推理
    with torch.no_grad():
        out = net(imgs)
    # 输出列表赋值
    output.extend(out.tolist())
    output1.extend(targets.tolist())

# 为方便复制结果，转化维度
output = np.array(output)
output = np.reshape(output, (1, -1))
output = np.round(output, decimals=2)
output = output.tolist()

output1 = np.array(output1)
output1 = np.reshape(output1, (1, -1))
output1 = np.round(output1, decimals=2)
output1 = output1.tolist()



print("真实值：", output1, "\n预测值：", output)
