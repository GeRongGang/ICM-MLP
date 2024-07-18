'''
将表格数据转换为图片数据
功能模块：数据集拆分并保存、表格数据转换为图片
注：将完整数据及地址指向train_csv_file变量，便可将输出中的train_loader, train_dataset,作为完整数据输出
对象：铁尾矿不完整数据集、3倍数据、
'''
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch
import chardet
from sklearn.model_selection import train_test_split
import numpy as np

def ShuJu(batch_size = 1):
    '''batch_size:dataloder的批量大小'''

    train_csv_file = r'E:\Deel\DM\data\HunNingTu\TieWeiKuang_1_D\train.csv'  # 定义训练集和验证集的文件路径
    val_csv_file = r'E:\Deel\DM\data\HunNingTu\TieWeiKuang_1_D\test.csv'
    csv_file_path = r'E:\Deel\DM\data\HunNingTu\TieWeiKuang_1_D\data.csv'  # 完整数据所在文件所在文件夹
    statistics_file = r'E:\Deel\DM\data\HunNingTu\TieWeiKuang_1_D\mean_stds.csv'  # 均值和标准差的保存路径

    # 检测文件的编码方式
    with open(train_csv_file, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    # print(encoding)

    # 使用检测到的编码方式读取文件
    train_data_0 = pd.read_csv(train_csv_file, encoding=encoding)
    val_data_0 = pd.read_csv(val_csv_file, encoding=encoding)
    # print(train_data_0)
    print(train_data_0.shape, val_data_0.shape)
    # print(train_data_0.iloc[0:4, [0, 18, 19, 20, -3, -2, -1]])
    # print(train_data_0.dtypes)
    # print(train_data_0.columns)

    #将训练集和验证集中的数值数据转换为float32
    currency = ['SNQD', 'CGL1Min', 'CGL1Max', 'CGL2Min', 'CGL2Max', 'SXD1',
                'SXD2', 'PShi', 'PSN', 'PS1', 'PS2', 'PCGL1', 'PCGL2', 'PKWCL1',
                'PKWCL2', 'PKWCL3', 'PJJJ', 'PWJJ2', 'CS', 'KWCL3', 'WJJ2']  # 需要更改为float的列
    for i in currency:
        train_data_0[i] = train_data_0[i].astype(np.float32)
    # print(train_data_0.dtypes)

    currency = ['SNQD', 'CGL1Min', 'CGL1Max', 'CGL2Min', 'CGL2Max', 'SXD1',
                'SXD2', 'PShi', 'PSN', 'PS1', 'PS2', 'PCGL1', 'PCGL2', 'PKWCL1',
                'PKWCL2', 'PKWCL3', 'PJJJ', 'PWJJ2', 'CS', 'KWCL3', 'WJJ2']  # 需要更改为float的列
    for i in currency:
        val_data_0[i] = val_data_0[i].astype(np.float32)
    # print(val_data_0.dtypes)

    # 去掉第一列——ID列
    train_data = train_data_0.iloc[:, 1:-1]
    train_labels = train_data_0.iloc[:, -1]
    val_data = val_data_0.iloc[:, 1:-1]
    val_labels = val_data_0.iloc[:, -1]
    # print(train_data)
    # print(val_data, val_labels)
    # print(train_data.describe())#查看均值等各项值

    # 对train，按列进行归一化(保存均值和方差，应用到val和test中)
    numeric_train_data = train_data.dtypes[train_data.dtypes != 'object'].index
    # numeric_train_labels = train_labels.dtypes[train_labels.dtypes != 'object'].index
    # 计算均值和标准差
    means_train = train_data[numeric_train_data].mean()
    means_labels = train_labels.mean()
    stds_train = train_data[numeric_train_data].std()
    stds_labels = train_labels.std()
    # 保存均值和标准差到文件
    statistics = pd.DataFrame({'mean_train': means_train, 'std_train': stds_train, 'mean_labels': means_labels, 'std_labels': stds_labels})
    statistics.to_csv(statistics_file, index=True)
    # 训练集归一化
    train_data[numeric_train_data] = (train_data[numeric_train_data] - means_train) / stds_train
    # train_labels = (train_labels - means_labels) / stds_labels
    train_data[numeric_train_data] = train_data[numeric_train_data].fillna(0)  # fillna(0)将缺失值设置为0
    # 验证集归一化
    val_data[numeric_train_data] = (val_data[numeric_train_data] - means_train) / stds_train
    # val_labels = (val_labels - means_labels) / stds_labels
    val_data[numeric_train_data] = val_data[numeric_train_data].fillna(0)  # fillna(0)将缺失值设置为0
    # print(val_data)'''
    #转为numpy数据类型
    train_data_np = train_data.to_numpy()
    train_labels_np = train_labels.to_numpy()
    val_data_np = val_data.to_numpy()
    val_labels_np = val_labels.to_numpy()
    print(train_data_np.shape, val_data_np.shape)
    #将数据转换为3维尺寸（类似灰度图像）
    train_data_np = train_data_np.reshape(-1, 1, 1, train_data_np.shape[1])
    train_labels_np = train_labels_np.reshape(-1, 1)
    val_data_np = val_data_np.reshape(-1, 1, 1, val_data_np.shape[1])
    val_labels_np = val_labels_np.reshape(-1, 1)
    print(train_data_np.shape, val_data_np.shape)

    # 将 NumPy 数组转换为 PyTorch 张量
    processed_train_data_tensor = torch.tensor(train_data_np)
    processed_train_labels_tensor = torch.tensor(train_labels_np)
    processed_val_data_tensor = torch.tensor(val_data_np)
    processed_val_labels_tensor = torch.tensor(val_labels_np)
    print(processed_val_labels_tensor.dtype)

    # 创建新的 TensorDataset
    train_dataset = TensorDataset(processed_train_data_tensor, processed_train_labels_tensor)
    val_dataset = TensorDataset(processed_val_data_tensor, processed_val_labels_tensor)

    # 创建新的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)#shuffle:每次迭代打乱顺序
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)#shuffle:每次迭代打乱顺序
    return train_loader, val_loader, train_dataset, val_dataset

    # train_data = torch.tensor(train_data.values, dtype=torch.float32)  # 将数据转化为tensor，float32
    # train_labels = torch.tensor(train_labels.values, dtype=torch.float32)  # 将数据转化为tensor，float32
    # val_data = torch.tensor(val_data.values, dtype=torch.float32)  # 将数据转化为tensor，float32
    # val_labels = torch.tensor(val_labels.values, dtype=torch.float32)  # 将数据转化为tensor，float32


if __name__ == '__main__':
    #查看数据类型
    train_loader_B, val_loader_B, train_dataset, val_dataset= ShuJu()
    batch_data, batch_labels = next(iter(train_loader_B))
    x = batch_data
    # 检查张量的数据类型
    print("Batch Data Type:", batch_data.shape)
    print("Batch Labels Type:", batch_labels.shape)
    print(type(batch_data))
