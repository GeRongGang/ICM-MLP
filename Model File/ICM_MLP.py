import torch
import torch.nn as nn
'''
交叉矩阵+自注意力机制+残差块+Dropout0.5
在训练的第二和第三训练阶段，只需要改变所传入的参数值便可
模型参数介绍：数据高度、数据宽度、隐藏层尺寸、输出尺寸、隐藏层层数。示例：net = MLP_Mixer(7, 7, 16, 2, 6)

注意：
1、仅第一个交叉矩阵块和最后FC层不包含自注意力机制
2、无
'''

#两个dropout的丢失率
dropout_1 = 0.2
dropout_2 = 0.6
dropout_3 = 0.2
dropout_4 = 0.6

#自注意力机制
class Self_Attention(nn.Module):
    def __init__(self, dim): # (线性层形状，dim是数据最后一维维度，dk是kq形状，dv是v的形状）
        super(Self_Attention, self).__init__()
        self.scale = dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        atten = (q @ k.transpose(0, 1)) * self.scale
        atten = atten.softmax(dim=-1)

        x = atten * v

        return x

# 定义Patch Mixer部分
'''函数输入部分，主要进行尺寸的变换与矩阵的内积
该类输出时，数据的尺寸已经变成指定隐藏层尺寸的方阵了'''
class PatchMixer1(nn.Module):
    def __init__(self, high_size, wide0, hidden_dim, wide1, wide2):
        super(PatchMixer1, self).__init__()
        self.high_size = high_size
        self.wide0 = wide0
        self.wide1 = wide1
        self.wide2 = wide2
        self.hidden_dim = hidden_dim

        self.mlp1 = nn.Sequential(nn.Linear(self.wide0, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))#将x1的特征数拓展为隐藏层大小
        self.mlp2 = nn.Sequential(nn.Linear(self.wide1, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))#将x2的特征数拓展为隐藏层大小
        self.mlp3 = nn.Sequential(nn.Linear(self.wide2, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))#将x3的特征数拓展为隐藏层大小
        self.mlp4 = nn.Sequential(nn.Linear(self.high_size, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))#将x3的特征数拓展为隐藏层大小

    def forward(self, x):
        x = x.to(next(self.mlp1.parameters()).dtype)
        x1 = x[:, :self.wide0]
        x2 = x[:, self.wide0:self.wide0+self.wide1]
        x3 = x[:, self.wide0+self.wide1:]
        x1 = self.mlp1(x1)
        x2 = self.mlp2(x2)
        x3 = self.mlp3(x3)
        x1 = x1.transpose(0, 1)
        x3 = x3.transpose(0, 1)
        # print(x1.shape, x2.shape, x3.shape)
        x = x1 @ x2
        x = x @ x3
        x = self.mlp4(x)
        return x

class PatchMixer(nn.Module):
    def __init__(self, high_size, wide, hidden_dim):
        super(PatchMixer, self).__init__()
        self.high_size = high_size
        self.wide = wide
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.wide, hidden_dim),#将数据拓展为隐藏层大小
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.transpose(0, 1)
        return x

# 定义Channel Mixer部分
class ChannelMixer(nn.Module):
    def __init__(self, high_size, hidden_dim):
        super(ChannelMixer, self).__init__()
        self.high_size = high_size
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.high_size, hidden_dim),#将数据拓展为隐藏层大小的方阵
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.transpose(0, 1)
        return x

#除去第一层外的残差网络
class minxer_1_Res(nn.Module):
    def __init__(self, hidden_dim):
        super(minxer_1_Res, self).__init__()
        self.PatchMixer = PatchMixer(hidden_dim, hidden_dim, hidden_dim)
        self.dropout_1 = nn.Dropout(dropout_1)
        self.Self_Attention_0 = Self_Attention(dim=hidden_dim)
        self.dropout_2 = nn.Dropout(dropout_2)

        self.ChannelMixer = ChannelMixer(hidden_dim, hidden_dim)
        self.dropout_3 = nn.Dropout(dropout_3)
        self.Self_Attention_1 = Self_Attention(dim=hidden_dim)
        self.dropout_4 = nn.Dropout(dropout_4)#参数是丢失率


    def forward(self, x):
        X = x
        x = self.PatchMixer(x)
        x = self.dropout_1(x)
        x = self.Self_Attention_0(x)
        x = self.dropout_2(x)
        x = self.ChannelMixer(x)
        x = self.dropout_3(x)
        x = self.Self_Attention_1(x)
        x = self.dropout_4(x)
        x = X + x

        return x

# 定义MLP-Mixer模型
class MLP_Mixer(nn.Module):
    def __init__(self, high_size, wide0, wide1, wide2, hidden_dim, num_classes, n):
        super(MLP_Mixer, self).__init__()
        self.high_size = high_size  # 由批量大小传入
        self.wide0 = wide0
        self.wide1 = wide1
        self.wide2 = wide2
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.n = n
        self.mixer_1 = minxer_1_Res(hidden_dim)

        #网络首段结构(将输入整合为方阵阶段)
        self.mixer_0 = nn.Sequential(PatchMixer1(high_size, wide0, hidden_dim, wide1, wide2),
                                     ChannelMixer(hidden_dim, hidden_dim))

        self.F = nn.Linear(hidden_dim*hidden_dim, self.num_classes)

        self.model = nn.Sequential()
        for i in range(self.n):                #网络设置为n层
            name = "minxer{}".format(i + 1)
            if i == 0:
                self.model.add_module(name, self.mixer_0)
            else:
                self.model.add_module(name, self.mixer_1)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(self.high_size, -1)  # 将3维数据改为2维(批量大小，特征数量)
        x = self.model(x)
        x = torch.flatten(x, start_dim=0)
        x = x.reshape(1, -1)
        x = self.F(x)
        # print("输出：", x.shape)
        return x

if __name__ == '__main__':

    x = torch.rand(1, 1, 1, 49)
    print(MLP_Mixer(1, 31, 6, 12, 48, 1, 5).model, MLP_Mixer(1, 31, 6, 12, 48, 1, 5).F)
    net = MLP_Mixer(1, 31, 6, 12, 48, 1, 5)
    output = net(x)
    print(output)
