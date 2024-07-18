import torch
from torch import nn

from Data_cnn_TWK0 import ShuJu
from model_ICM_MLP import MLP_Mixer


def main():
    name = '基础模型_数据_1_D'
    Dropout = [0.5]
    # 设置参数
    batch_size = 1  # 批量大小,注意！！该值应与数据叠加倍数相同
    weight_decay = 0.01  # L2正则化参数
    lr_1 = 0.01  # 学习率/model_Attention_Res模型推荐0.0001
    lr_2 = 0.001  # 第二阶段学习率
    lr_epoch = 600  # 更换学习率的迭代次数
    epochs = 600  # 迭代次数
    n = 80

    high_size, wide0, wide1, wide2, hidden_dim, num_classes = batch_size, 31, 6, 12, 48, batch_size  # wide:特征数, hidden_dim:隐藏层尺寸, n是网络层数

    # 指定GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 导入数据集
    train_loader, test_loader, train_set, test_set = ShuJu(batch_size)

    train_data_size = len(train_set)
    test_data_size = len(test_set)
    print("训练集长度：{}".format(train_data_size))
    print("测试集长度：{}".format(test_data_size))

    # 导入模型
    net = MLP_Mixer(high_size, wide0, wide1, wide2, hidden_dim, num_classes, n)
    net.to(device)

    # 损失函数
    loss_fn = nn.MSELoss(reduction='mean')
    loss_fn.to(device)

    # 定义优化器
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.01)
    # optimizer_1 = torch.optim.Adadelta(net.parameters(), lr=lr_1,  weight_decay=weight_decay)#weight_decay=0.01表示L2系数betas=(0.9, 0.999), eps=1e-08,, amsgrad=False  #Adagrad
    optimizer_1 = torch.optim.Adadelta(net.parameters(), rho=0.9, eps=1e-8, weight_decay=0.06)

    # optimizer_2 = torch.optim.Adadelta(net.parameters(), lr=lr_2,  weight_decay=weight_decay)#weight_decay=0.01表示L2系数betas=(0.9, 0.999), eps=1e-08,, amsgrad=False

    # 计算MSE、RMSE、MAE、R^2
    def calculate_metrics(predictions, targets, num_features=27):  # num_features：特征数量
        predictions = torch.stack(predictions).view(-1)  # 将包含tensor的list更改为1维tensor类型
        targets = torch.stack(targets).view(-1)

        n = len(targets)  # 样本数量，adj_r2需要
        def adjusted_r2(num_features, r2):  # 计算adj_r2
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))
            return adj_r2
        # 计算各类值
        mse = torch.mean((predictions - targets) ** 2)  # 计算MSE
        rmse = torch.sqrt(mse)  # 计算RMSE
        mae = torch.mean(torch.abs(predictions - targets))  # 计算MAE
        mape = torch.mean(torch.abs((targets - predictions) / targets)) * 100
        vaf = 100 * (1 - torch.var(targets - predictions) / torch.var(targets))
        rse = rmse / torch.std(targets)
        ev = 1 - (torch.var(targets - predictions) / torch.var(targets))

        # 计算R^2
        ssr = torch.sum((predictions - targets) ** 2)
        sst = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ssr / sst)
        adj_r2 = adjusted_r2(num_features, r2)
        # 计算

        return mae.item(), mse.item(), rmse.item(), r2.item(), adj_r2.item(), mape.item(), vaf.item(), rse.item(), ev.item()

    # 初始化列表储存指标值
    train_output = []  # 每条数据，训练集输出值
    train_targets = []  # 每条数据，真实值
    train_MAE = []  # 每个epoch，训练集MAE值
    train_MSE = []  # 每个epoch，训练集MSE值
    train_RMSE = []  # 每个epoch，训练集RMSE值
    train_R2 = []  # 每个epoch，训练集R^2值
    train_adj_r2 = []
    train_mape = []
    train_vaf = []
    train_rse = []
    train_ev = []

    test_output = []  # 测试集值
    test_MAE = []
    test_MSE = []
    test_RMSE = []
    test_R2 = []
    test_adj_r2 = []
    test_mape = []
    test_vaf = []
    test_rse = []
    test_ev = []
    test_targets = []
    min_loss = float('inf')  # 初始化为正无穷大，用以寻找最小损失时的权重

    # 开始训练
    for epoch in range(epochs):
        net.train()
        print("---------第{}轮训练开始---------".format(epoch + 1))

        for data in train_loader:
            imgs, targets = data
            output = net(imgs.to(device))
            targets = targets.reshape(1, -1).to(device)
            loss = loss_fn(output, targets)  # MSEloss，按该损失反向传播
            # 更换步长的迭代次数
            if epoch < lr_epoch:
                optimizer_1.zero_grad()
                loss.backward()
                optimizer_1.step()
            # else:
            #     optimizer_2.zero_grad()
            #     loss.backward()
            #     optimizer_2.step()
            # 储存预测值与标签值，用以计算各种损失
            train_output.append(output)
            train_targets.append(targets)

        # 切换为评估模型
        net.eval()
        with torch.no_grad():  # 禁止求梯度
            # print("开始测试")
            for data in test_loader:
                imgs, targets = data
                output = net(imgs.to(device))
                targets = targets.reshape(1, -1).to(device)
                # 储存预测值与标签值，用以计算各种损失
                test_output.append(output)
                test_targets.append(targets)

            # 调用calculate_metrics，计算train和test的各种指标
            t_MAE, t_MSE, t_RMSE, t_R2, t_adj_r2, t_mape, t_vaf, t_rse, t_ev = calculate_metrics(train_output, train_targets, 27)
            train_MAE.append(t_MAE)
            train_MSE.append(t_MSE)
            train_RMSE.append(t_RMSE)
            train_R2.append(t_R2)
            train_adj_r2.append(t_adj_r2)
            train_mape.append(t_mape)
            train_vaf.append(t_vaf)
            train_rse.append(t_rse)
            train_ev.append(t_ev)

            t2_MAE, t2_MSE, t2_RMSE, t2_R2, t2_adj_r2, t2_mape, t2_vaf, t2_rse, t2_ev = calculate_metrics(test_output, test_targets, 27)
            test_MAE.append(t2_MAE)
            test_MSE.append(t2_MSE)
            test_RMSE.append(t2_RMSE)
            test_R2.append(t2_R2)
            test_adj_r2.append(t2_adj_r2)
            test_mape.append(t2_mape)
            test_vaf.append(t2_vaf)
            test_rse.append(t2_rse)
            test_ev.append(t2_ev)

            # 清空该epoch的预测输出与真实标签
            train_output.clear()
            train_targets.clear()
            test_output.clear()
            test_targets.clear()
        # 保存MSEloss最小时的权重
        if t2_MSE < min_loss:
            min_loss = t2_MSE
            min_weights = net.state_dict()
            min_loss_epoch = epoch

        print("本轮MSE训练损失：{}\n本轮MSE测试损失：{}".format(t_MSE, t2_MSE))
        print("该轮训练结束")
    torch.save(min_weights, 'min_weights.pth')
    CC = [n, name, Dropout, weight_decay, batch_size, lr_1, lr_2, lr_epoch, epochs]
    print("{}层模型：{}，Dropout：{}\nL2正则化参数：{} batch_size：{} 步长1：{} 步长2：{} 迭代次数1：{} 迭代次数2：{}".format(
        CC[0], CC[1], CC[2], CC[3], CC[4], CC[5], CC[6], CC[7], CC[8]))
    # Fanhui2 = train_adj_r2, train_mape, train_vaf, train_rse, train_ev, test_adj_r2, test_mape, test_vaf, test_rse, test_ev
    Fanhui = train_MAE, train_MSE, train_RMSE, train_R2, test_MAE, test_MSE, test_RMSE, test_R2, min_loss_epoch, CC, hidden_dim, train_adj_r2, train_mape, train_vaf, train_rse, train_ev, test_adj_r2, test_mape, test_vaf, test_rse, test_ev
    # 训练集和测试集的MAE、MSE、RMSE、R2，测试MSE最小时的批次
    return Fanhui


if __name__ == '__main__':
    main()


