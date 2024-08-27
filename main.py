# %%
# https://www.cnblogs.com/lugendary/p/16192669.html
# https://blog.csdn.net/qq_42365109/article/details/115140450

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# %%
import time
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class NetTrainer:
    """
    这是一个简易的nn训练器。仅支持前馈神经网络，暂不支持RNN等含有隐藏状态的网络。

    你可以使用下面2个代码快速上手：

    [快速上手-1.回归]:
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        import torch

        from easier_nn.train_net import NetTrainer

        np.random.seed(42)
        data = np.random.rand(1000, 10)
        target = np.sum(data, axis=1) + np.random.normal(0, 0.1, 1000)  # target is sum of features with some noise

        class RegressionNet(nn.Module):
            def __init__(self):
                super(RegressionNet, self).__init__()
                self.fc1 = nn.Linear(10, 50)
                self.fc2 = nn.Linear(50, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # 创建模型、损失函数和优化器
        net = RegressionNet()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        # 训练模型
        trainer = NetTrainer(data, target, net, loss_fn, optimizer, epochs=200)
        trainer.train_net()
        trainer.evaluate_net()

        # 查看模型的层与参数(以train_loss_list为例)
        # nn.Module 对象不能直接进行迭代，需要通过访问它的 modules() 或 children() 方法来迭代它的层。
        # modules() 方法返回模块和它所有的子模块，而 children() 方法仅返回模块的直接子模块。
        for layer in net.children():
            print(layer)
            # if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
            #     print('Weight:', layer.weight)
            #     print('Bias:', layer.bias)
            # print('-----------------')
        print(trainer.train_loss_list)

    [快速上手-分类，请确保使用了参数net_type="acc"]:
        print("--------分类---------")
        # 生成分类数据
        np.random.seed(42)
        data = np.random.rand(1000, 20)  # 20个特征
        target = (np.sum(data, axis=1) > 10).astype(int)  # 如果特征和大于10，则类别为1，否则为0

        class ClassificationNet(nn.Module):
            def __init__(self):
                super(ClassificationNet, self).__init__()
                self.fc1 = nn.Linear(20, 50)
                self.fc2 = nn.Linear(50, 2)  # 2个类别

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        net = ClassificationNet()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # 训练模型
        trainer = NetTrainer(data, target, net, loss_fn, optimizer, epochs=50, eval_type="acc")
        trainer.train_net()
        trainer.evaluate_net()

        print(trainer.test_acc_list)
    """
    def __init__(self, data, target, net, loss_fn, optimizer,          # 必要参数，数据与网络的基本信息
                 test_size=0.2, batch_size=64, epochs=100,             # 可选参数，用于训练
                 eval_type="loss",                                     # 比较重要的参数，用于选择训练的类型（与评估指标有关）
                 eval_during_training=True,                            # 可选参数，训练时是否进行评估（与显存有关）
                                                                       # 补充：经过优化，目前即使训练时评估也不需要额外太多的显存了
                 rnn_input_size=None, rnn_seq_len=None, rnn_hidden_size=None,  # 可选参数，当net是RNN类型时需要传入这些参数
                                                                               # Bug：对RNN的train,test划分不太行，建议传入tuple
                                                                               # batch_size不是1的时候测试集的损失会有问题
                 print_interval=20,  # 其他参数，训练时的输出间隔
                 device=None,  # 其他参数，设备选择
                 **kwargs):
        """
        初始化模型。

        :param data: 数据或训练集，X or (X_train, X_test) or train_loader
        :param target: 目标或验证集，y or (y_train, y_test) or test_loader
        :param net: 支持 net=nn.Sequential() or class Net(nn.Module)
        :param loss_fn: 损失函数，例如：
            nn.MSELoss()  # 回归，y的维度应该是(batch,)
            nn.CrossEntropyLoss()  # 分类，y的维度应该是(batch,)，并且网络的最后一层不需要加softmax
            nn.BCELoss()  # 二分类，y的维度应该是(batch,)，并且网络的最后一层需要加sigmoid
        :param optimizer: 优化器
        :param test_size: 测试集大小，支持浮点数或整数。该参数在data和target是tuple时无效
        :param batch_size: 批量大小
        :param epochs: 训练轮数
        :param eval_type: 模型类型，只可以是"loss"(回归-损失)或"acc"(分类-准确率)
        :param print_interval: 打印间隔，请注意train_loss_list等间隔也是这个
        :param eval_during_training: 训练时是否进行评估，当显存不够时，可以设置为False，等到训练结束之后再进行评估
          设置为False时，不会影响训练集上的Loss的输出，但是无法输出验证集上的loss、训练集与验证集上的acc，此时默认输出"No eval"
        :param rnn_input_size: RNN的输入维度
        :param rnn_seq_len: RNN的序列长度
        :param rnn_hidden_size: RNN的隐藏层大小
          以上三个参数同时设置时，自动判断网络类型为RNN
        :param device: 设备，支持"cuda"或"cpu"，默认为None，自动优先选择"cuda"
        :param kwargs: 其他参数，包括：
          target_reshape_1D: 是否将y的维度转换为1维，默认为True，用于_target_reshape_1D函数，
            仅在为True且self.eval_type == "acc"且y.dim() > 1时才会转换并发出警告
          drop_last: 是否丢弃最后一个batch，默认为False，用于DataLoader
        """
        self.target_reshape_1D = kwargs.get("target_reshape_1D", True)
        self.drop_last = kwargs.get("drop_last", False)

        # 设备参数
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 数据参数
        self.data = data  # X or (X_train, X_test) or train_loader
        self.target = target  # y or (y_train, y_test) or test_loader
        self.test_size = test_size
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.train_loader, self.test_loader = None, None
        # 网络参数
        self.net = net.to(self.device)
        # self.net = torch.compile(self.net)  # RuntimeError: Windows not yet supported for torch.compile 哈哈哈！
        self.net_type = "FNN"  # 默认是前馈神经网络
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # 训练参数
        self.batch_size = batch_size
        self.epochs = epochs
        # 训练输出参数
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_loss_list = []
        self.test_acc_list = []
        self.time_list = []
        self.print_interval = print_interval  # 打印间隔
        # 使用loss还是acc参数
        self.eval_type = eval_type
        # 训练时是否进行评估
        self.eval_during_training = eval_during_training
        self.no_eval_msg = '"No eval"'  # 不在训练时评估的输出
        # self.original_dataset_to_device = False  # False表示数据还没有转移到设备上
        # 是否是RNN类型
        self.rnn_seq_len = rnn_seq_len  # 该参数暂时没有使用，如果代码写完了还没用就删了得了
        self.rnn_hidden_size = rnn_hidden_size  # 同上
        self.rnn_input_size = rnn_input_size  # 在计算损失时需要用到
        self.hidden = None
        if self.rnn_input_size:
            self.net_type = "RNN"
        # 初始化
        self.init_loader()

    # [init]初始化训练数据
    def init_loader(self):
        # 如果传入的是DataLoader实例，则直接赋值
        if isinstance(self.data, DataLoader) and isinstance(self.target, DataLoader):
            self.train_loader = self.data
            self.test_loader = self.target
            print("[init_loader]传入的data与target是DataLoader实例，直接赋值train_loader和test_loader。")
            # 从DataLoader中获取数据
            self.X_train, self.y_train = self._dataloader_to_tensor(self.train_loader)
            self.X_test, self.y_test = self._dataloader_to_tensor(self.test_loader)
        else:
            # 如果传入的就是tuple，则表示已经划分好了训练集和测试集
            if isinstance(self.data, tuple) and isinstance(self.target, tuple):
                self.X_train, self.X_test = self.data
                self.y_train, self.y_test = self.target
                print("[init_loader]因为传入的data与target是元组，所以默认已经划分好了训练集和测试集。"
                      "默认元组第一个是train，第二个为test。")
            # 否则，需要划分训练集和测试集
            else:
                if self.data.shape[0] != self.target.shape[0]:
                    raise ValueError(f"data和target的shape[0](样本数)不相同: "
                                     f"data({self.data.shape[0]}) and target({self.target.shape[0]}).")
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(self.data, self.target, test_size=self.test_size)
                print(f"[init_loader]传入的data与target是X, y，则按照test_size={self.test_size}划分训练集和测试集")

            # if self.net_type == "RNN":
            #     self.X_train, self.y_train = self._prepare_rnn_data(self.X_train, self.y_train)
            #     self.X_test, self.y_test = self._prepare_rnn_data(self.X_test, self.y_test)
            #     print(f"[init_loader]RNN数据准备完毕，seq_len={self.rnn_seq_len}, hidden_size={self.rnn_hidden_size}")

            # 创建DataLoaders
            self.train_loader = self.create_dataloader(self.X_train, self.y_train)
            self.test_loader = self.create_dataloader(self.X_test, self.y_test, train=False)
        # 将数据变成tensor，并且dtype依据data的类型而定
        self.X_train = self._dataframe_to_tensor(self.X_train)
        self.X_test = self._dataframe_to_tensor(self.X_test)
        self.y_train = self._dataframe_to_tensor(self.y_train)
        self.y_test = self._dataframe_to_tensor(self.y_test)
        self.y_train = self._target_reshape_1D(self.y_train)
        self.y_test = self._target_reshape_1D(self.y_test)

    # [子函数]创建dataloader
    def create_dataloader(self, data, target, train=True):
        # dtype依据data的类型而定
        data = self._dataframe_to_tensor(data)
        target = self._dataframe_to_tensor(target)
        target = self._target_reshape_1D(target)

        # print(target)
        dataset = TensorDataset(data, target)
        if train:
            if self.net_type == "RNN":
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_last)
            else:
                return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)
        else:
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_last)

    # [主函数]训练模型
    def train_net(self, hidden=None):
        if hidden is not None:
            self.hidden = hidden
        print(f"[train_net]开始训练模型，总共epochs={self.epochs}，batch_size={self.batch_size}，"
              f"当前设备为{self.device}，网络类型为{self.net_type}，评估类型为{self.eval_type}。")
        current_gpu_memory = self._log_gpu_memory()
        print(current_gpu_memory)
        self.net.train()

        for epoch in range(self.epochs):
            start_time = time.time()
            loss_sum = 0.0
            for X, y in self.train_loader:
                # 初始化数据
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # 前向传播
                if self.net_type == "RNN":
                    if self.hidden is not None:
                        self.hidden.detach_()
                    #     print(self.hidden.shape)
                    # print(X.shape, y.shape)
                    # print("----------上面是TRAIN的hidden, X, y的shape---------")
                    outputs, self.hidden = self.net(X, self.hidden)
                    loss = self.loss_fn(outputs, y)
                else:
                    # print(X.shape, y.shape)
                    # print("----------上面是TRAIN的hidden, X, y的shape---------")
                    outputs = self.net(X)
                    # print(X.shape, y.shape, outputs.shape)
                    loss = self.loss_fn(outputs, y)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                # 计算损失
                loss_sum += loss.item()
                # 计算当前GPU显存
                current_gpu_memory = self._log_gpu_memory()
                # 释放显存。如果不释放显存，直到作用域结束时才会释放显存（这部分一直在reserve的显存里面）
                del X, y, outputs, loss
                torch.cuda.empty_cache()
            loss_epoch = loss_sum / len(self.train_loader)
            self.time_list.append(time.time() - start_time)
            # 打印训练信息
            if epoch % self.print_interval == 0:
                if self.eval_type == "loss":
                    self.train_loss_list.append(loss_epoch)
                    self.test_loss_list.append(self.evaluate_net())
                    print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {loss_epoch}, '
                          f'Test Loss: {self.test_loss_list[-1]}, '
                          f'Time: {self.time_list[-1]:.2f}s, '
                          f'GPU: {current_gpu_memory}')
                elif self.eval_type == "acc":
                    self.train_acc_list.append(self.evaluate_net(eval_type="train"))
                    self.test_acc_list.append(self.evaluate_net())
                    print(f'Epoch {epoch + 1}/{self.epochs}, Train Loss: {loss_epoch}, '
                          f'Train Acc: {self.train_acc_list[-1]}, '
                          f'Test Acc: {self.test_acc_list[-1]}, '
                          f'Time: {self.time_list[-1]:.2f}s, '
                          f'GPU: {current_gpu_memory}')
                else:
                    raise ValueError("eval_type must be 'loss' or 'acc'")
        print(f"[train_net]训练结束，总共花费时间: {sum(self.time_list)}秒")
        print(self._log_gpu_memory())
        self.eval_during_training = True  # 训练完成后，可以进行评估

    # [主函数]评估模型(暂不支持RNN的评估)
    def evaluate_net(self, eval_type="test", delete_train=False):
        """
        评估模型
        :param eval_type: 评估类型，支持"test"和"train"
        :param delete_train: delete_train=True表示删除训练集，只保留测试集，这样可以释放显存
        :return: 损失或准确率，依据self.net_type而定
        """
        if delete_train:
            del self.X_train, self.y_train
            torch.cuda.empty_cache()
        # if self.eval_during_training:
        #     self.__original_dataset_to_device()  # 如果要在训练时评估，需要将数据转移到设备上
        # else:
        #     return self.no_eval_msg  # 不在训练时评估
        if not self.eval_during_training:
            return self.no_eval_msg  # 不在训练时评估
        self.net.eval()
        with torch.no_grad():  # 在评估时禁用梯度计算，节省内存
            if self.eval_type == "loss":
                if self.net_type == "RNN":
                    if eval_type == "test":
                        output = self._cal_rnn_output(self.net, self.X_test[0], self.hidden[:, -1], len(self.y_test))
                        loss = self.loss_fn(output, self.y_test).item()
                    else:
                        # 事实上一般不调用这个，因为训练集的loss在训练时已经计算了
                        output = self._cal_rnn_output(self.net, self.X_train[0], self.hidden[:, 0], len(self.y_train))
                        loss = self.loss_fn(output, self.y_train).item()
                else:
                    if eval_type == "test":
                        loss = self._cal_fnn_loss(self.net, self.loss_fn, self.X_test, self.y_test)
                        # loss = self.loss_fn(self.net(self.X_test), self.y_test).item()
                    else:
                        # 事实上一般不调用这个，因为训练集的loss在训练时已经计算了
                        loss = self._cal_fnn_loss(self.net, self.loss_fn, self.X_train, self.y_train)
                        # loss = self.loss_fn(self.net(self.X_train), self.y_train).item()
                self.net.train()
                return loss
            elif self.eval_type == "acc":
                if self.net_type == "RNN":
                    if eval_type == "test":
                        acc = self._cal_rnn_acc(self.net, self.X_test, self.y_test)
                        # predictions = torch.argmax(self.net(self.X_test, self.hidden), dim=1).type(self.y_test.dtype)
                        # correct = (predictions == self.y_test).sum().item()
                        # n = self.y_test.numel()
                        # acc = correct / n
                    else:
                        acc = self._cal_rnn_acc(self.net, self.X_train, self.y_train)
                        # predictions = torch.argmax(self.net(self.X_train, self.hidden), dim=1).type(self.y_train.dtype)
                        # correct = (predictions == self.y_train).sum().item()
                        # n = self.y_train.numel()
                        # acc = correct / n
                else:
                    if eval_type == "test":
                        acc = self._cal_fnn_acc(self.net, self.X_test, self.y_test)
                        # predictions = torch.argmax(self.net(self.X_test), dim=1).type(self.y_test.dtype)
                        # correct = (predictions == self.y_test).sum().item()
                        # n = self.y_test.numel()
                        # acc = correct / n
                    else:
                        acc = self._cal_fnn_acc(self.net, self.X_train, self.y_train)
                        # predictions = torch.argmax(self.net(self.X_train), dim=1).type(self.y_train.dtype)
                        # correct = (predictions == self.y_train).sum().item()
                        # n = self.y_train.numel()
                        # acc = correct / n
                self.net.train()
                return acc
        # total, correct = 0, 0
        # with torch.no_grad():
        #     for inputs, labels in self.test_loader:
        #         inputs, labels = inputs.to(self.device), labels.to(self.device)
        #         outputs = self.net(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()
        #
        # print(f'Accuracy: {100 * correct / total}%')

    # [主函数]查看模型参数。使用Netron(需要安装)可视化更好，这里只是简单的查看
    def view_parameters(self, view_net_struct=True, view_params_count=True, view_params_details=False):
        # if view_layers:
        #     for layer in self.net.children():
        #         print(layer)
        if view_net_struct:
            print("网络结构如下：")
            print(self.net)
        if view_params_count:
            count = 0
            for p in self.net.parameters():
                if view_params_details:
                    print("该层的参数：" + str(list(p.size())))
                count += p.numel()
            print(f"总参数量: {count}")
            # print(f"Total params: {sum(p.numel() for p in self.net.parameters())}")

        # params = list(self.net.parameters())
        # k = 0
        # for i in params:
        #     l = 1
        #     print(f"该层的名称：{i.size()}")
        #     print("该层的结构：" + str(list(i.size())))
        #     for j in i.size():
        #         l *= j
        #     print("该层参数和：" + str(l))
        #     k = k + l
        # print("总参数数量和：" + str(k))

    # [子函数]评估FNN的loss
    def _cal_fnn_loss(self, net, criterion, x, y):
        net.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                if len(X_batch) == 0:
                    warnings.warn(f"[_cal_fn_loss]最后一个batch的长度为0，理论上不会出现这个情况吧")
                    continue
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                del X_batch, y_batch, outputs, loss
                torch.cuda.empty_cache()

        average_loss = total_loss / len(x)
        return average_loss

    # [子函数]评估RNN的loss
    def _cal_rnn_loss(self, net, criterion, x, y):
        net.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                if len(X_batch) == 0:
                    warnings.warn(f"[_cal_rnn_loss]最后一个batch的长度为0，理论上不会出现这个情况吧")
                    continue
                hidden = self.hidden.detach()
                outputs, _ = net(X_batch, hidden)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item() * y_batch.size(0)
                del X_batch, y_batch, outputs, loss
                torch.cuda.empty_cache()

        average_loss = total_loss / len(x)
        return average_loss

    # [子函数]评估RNN的loss（该函数暂时有问题）
    def _cal_rnn_output(self, net, x, hidden, pred_steps):
        hidden.to(self.device)
        pred_list = []
        # 输出x的shape
        # print(x.shape)
        # print(x)
        # 调整输入形状为 [batch_size, seq_len, input_size]
        # inp = x.view(self.batch_size, self.rnn_seq_len, self.rnn_input_size).to(self.device)

        inp = x.view(-1, self.rnn_input_size).to(self.device)
        # print(x.shape, inp.shape, hidden.shape)
        # print("----------上面是EVAL的x, inp, hidden的shape---------")
        for i in range(pred_steps):
            pred, hidden = net(inp, hidden)
            pred_list.append(pred.detach())
            inp = pred
        return torch.cat(pred_list, dim=0).view(-1)

    # [子函数]评估FNN的acc
    def _cal_fnn_acc(self, net, x, y):
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                if len(X_batch) == 0:
                    warnings.warn(f"[_cal_accuracy]最后一个batch的长度为0，理论上不会出现这个情况吧")
                    continue
                outputs = net(X_batch)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                del X_batch, y_batch, outputs, predictions
                torch.cuda.empty_cache()

        accuracy = correct / total
        return accuracy

    # [子函数]评估RNN的acc
    def _cal_rnn_acc(self, net, x, y):
        net.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                X_batch = x[i:i + self.batch_size].to(self.device)
                y_batch = y[i:i + self.batch_size].to(self.device)
                # 如果X_batch的长度不等于batch_size，说明是最后一个batch
                if len(X_batch) != self.batch_size:
                    warnings.warn(f"[_cal_rnn_acc]最后一个batch的长度为{len(X_batch)}≠{self.batch_size}，"
                                  f"暂时的处理方法是跳过，可能会影响准确率的计算")
                    break
                hidden = self.hidden.detach()
                outputs, _ = net(X_batch, hidden)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
                del X_batch, y_batch, outputs, predictions
                torch.cuda.empty_cache()

        accuracy = correct / total
        return accuracy

    # # [子函数]准备RNN数据
    # def _prepare_rnn_data(self, data, target):
    #     seq_len = self.rnn_seq_len
    #     data_len = len(data)
    #     num_sequences = data_len // (seq_len + 1) * (seq_len + 1)
    #     data = np.array(data[:num_sequences]).reshape((-1, seq_len + 1, 1))
    #     target = np.array(target[:num_sequences]).reshape((-1, seq_len + 1, 1))
    #     return data[:, :seq_len], data[:, 1:seq_len + 1]

    # 将df转换为tensor，并保持数据类型的一致性

    # [log函数]打印GPU显存
    def _log_gpu_memory(self):
        if not self.eval_during_training:
            return self.no_eval_msg  # 不在训练时评估
        log = None

        # 获取self.device的设备索引
        self_device_index = None
        # 如果是cuda
        if self.device.type == "cuda":
            self_device_index = self.device.index

        # 获取当前设备索引
        current_device_index = torch.cuda.current_device()
        if current_device_index is None:
            log = "当前没有GPU设备"
            return log
        elif self_device_index is not None and current_device_index != self_device_index:
            warnings.warn(f"[_log_gpu_memory]当前设备为{current_device_index}，与{self_device_index}不一致")
        else:
            log = ""

        props = torch.cuda.get_device_properties(current_device_index)  # 获取设备属性
        used_memory = torch.cuda.memory_allocated(current_device_index)  # 已用显存（字节）
        reserved_memory = torch.cuda.memory_reserved(current_device_index)  # 保留显存（字节）
        total_memory = props.total_memory  # 总显存（字节）
        used_memory_gb = used_memory / (1024 ** 3)  # 已用显存（GB）
        reserved_memory_gb = reserved_memory / (1024 ** 3)  # 保留显存（GB）
        total_memory_gb = total_memory / (1024 ** 3)  # 总显存（GB）
        log += (f"设备{current_device_index}的显存："
                f"已用{used_memory_gb:.2f}+保留{reserved_memory_gb:.2f}/总{total_memory_gb:.2f}(GB)")

        return log

    @staticmethod
    def _dataframe_to_tensor(df, float_dtype=torch.float16, int_dtype=torch.int64):
        """
        PyTorch's tensors are homogenous, ie, each of the elements are of the same type.
        将df转换为tensor，并保持数据类型的一致性
        :param df: pd.DataFrame
        :param float_dtype: torch.dtype, default=torch.float32
        :param int_dtype: torch.dtype, default=torch.int32
        :return: torch.Tensor
        """
        # 先判断df是不是dataframe
        if not isinstance(df, pd.DataFrame):
            if isinstance(df, torch.Tensor):
                return df
            elif isinstance(df, np.ndarray):
                return torch.tensor(df)
            else:
                raise ValueError("既不是dataframe又不是tensor")
        # 检查df中的数据类型
        dtypes = []
        for col in df.column:
            if pd.api.types.is_float_dtype(df[col]):
                dtypes.append(float_dtype)
            elif pd.api.types.is_integer_dtype(df[col]):
                dtypes.append(int_dtype)
            else:
                raise ValueError(f"[_dataframe_to_tensor]Unsupported data type in column {col}: {df[col].dtype}")
        # print(dtypes)
        # 将df中的每一列转换为tensor
        # 对于多维的data
        if len(dtypes) > 1:
            tensors = [torch.as_tensor(df[col].values, dtype=dtype) for col, dtype in zip(df.columns, dtypes)]
            return torch.stack(tensors, dim=1)  # 使用torch.stack将多个tensor堆叠在一起
        # 对于一维的target
        elif len(dtypes) == 1:
            return torch.as_tensor(df.values, dtype=dtypes[0])
        else:
            raise ValueError(f"[_dataframe_to_tensor]数据长度有误{len(dtypes)}")

    @staticmethod
    def _dataloader_to_tensor(dataloader):
        data_list = []
        target_list = []
        for data, target in dataloader:
            data_list.append(data)
            target_list.append(target)
        return torch.cat(data_list), torch.cat(target_list)

    # 将y的维度转换为1维
    def _target_reshape_1D(self, y):
        """
        将y的维度转换为1维
        :param y: torch.Tensor
        :return: torch.Tensor
        """
        if self.target_reshape_1D and self.eval_type == "acc" and y.dim() > 1:
            warnings.warn(f"[_target_reshape_1D]请注意：y的维度为{y.dim()}: {y.shape}，将被自动转换为1维\n"
                          "如需保持原有维度，请设置 target_reshape_1D=False ")
            return y.view(-1)
        else:
            return y

    # 将原始数据转移到设备上，暂被弃用
    # def __original_dataset_to_device(self):
    #     # 暂时不知道只使用self.original_dataset_to_device是否会有问题，或许可以直接检查self.X_train.device(有问题再改吧)
    #     if not self.original_dataset_to_device:
    #         # 将数据转移到设备上
    #         self.X_train, self.X_test, self.y_train, self.y_test = self.X_train.to(self.device), self.X_test.to(
    #             self.device), self.y_train.to(self.device), self.y_test.to(self.device)
    #         self.original_dataset_to_device = True



# %%
df_train = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')
df_test = pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/test.tsv.zip', sep='\t')
print(df_train.head())
print(df_train.shape)  # (156060, 4)


import os
import zipfile
import urllib.request

# 设置路径和URL
GLOVE_ZIP_PATH = '/kaggle/working/glove.6B.zip'
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_PATH = '/kaggle/working/glove.6B.50d.txt'
# 下载GloVe压缩包
if not os.path.exists(GLOVE_ZIP_PATH):
    urllib.request.urlretrieve(glove_url, GLOVE_ZIP_PATH)
# 解压文件
with zipfile.ZipFile(GLOVE_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working/")
# 检查解压后的文件
if os.path.exists(GLOVE_PATH):
    print(f"'{GLOVE_PATH}' file is ready for use.")
else:
    print("Error: The GloVe file was not found after extraction.")

# %%
# # 去除"Phrase"里小于等于3个字符的短语
# df_train = df_train[df_train["Phrase"].apply(lambda x: len(x.split()) > 3)]
# print(df_train.shape)  # (92549, 4)

# df的列名和含义如下：
# PhraseId（短语ID）：每个短语的唯一标识符。
# SentenceId（句子ID）：短语所在句子的标识符，一个句子可能包含多个短语。
# Phrase（短语）：需要进行情感分析的文本短语。
# Sentiment（情感）：情感标签，表示该短语的情感倾向。从0到4的整数，表示从非常消极（0）到非常积极（4）的情感范围。

# 提取出短语和情感标签
X = df_train["Phrase"].values
y = df_train["Sentiment"].values
print(X[:5])
print(y[:5])

# 查看y的取值的分布：
# 对于整个数据集，情感标签的分布如下：
# (array([0, 1, 2, 3, 4], dtype=int64), array([ 7072, 27273, 79582, 32927,  9206], dtype=int64))
# 如果模型全预测为2，那么准确率为79582/156060=50.994%，因此模型的准确率应该要高于50.994%
# 对于去除了较短短语的数据集，情感标签的分布如下：
# (array([0, 1, 2, 3, 4], dtype=int64), array([ 5979, 19785, 36795, 22651,  7339], dtype=int64))
# 如果模型全预测为2，那么准确率为36795/92549=39.76%，因此模型的准确率应该要高于39.76%
print(np.unique(y, return_counts=True))

# %%
# 加载GloVe词向量
class LoadGlove:
    def __init__(self, glove_path):
        self.glove_dict = {}
        self.glove_path = glove_path

# 加载GloVe词向量
    def load_glove_vectors(self):
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                self.glove_dict[word] = vector

# 将句子转换为二维矩阵
    def sentence_to_matrix(self, sentence, embedding_dim=50):
        words = sentence.split()
        matrix = np.zeros((len(words), embedding_dim))
        for i, word in enumerate(words):
            if word in self.glove_dict:
                matrix[i] = self.glove_dict[word]
            else:
                # 对于未找到的单词，使用1e-3的常量，这样可以避免全零的情况
                matrix[i] = 1e-3 * np.ones(embedding_dim)
        return matrix.astype(np.float32)  # 避免nn的时候出现float与int混用的问题

# %%
# 将句子变为embedding矩阵
load_glove = LoadGlove(GLOVE_PATH)
load_glove.load_glove_vectors()
# 转小写是因为GloVe词向量是小写的
X_embedding = [load_glove.sentence_to_matrix(sentence.lower()) for sentence in X]

for i in range(8):
    print(f"Sentence {i}: {np.array(X_embedding[i]).shape}")

print(f"Total sentences: {len(X_embedding)}")

# 基本模型--mean：将句子中所有词的词向量取平均作为句子的表示
X_mean = np.array([sentence.mean(axis=0) for sentence in X_embedding])
# 基本模型--padding：将所有句子的embedding矩阵展平成一个向量
max_length = max(len(sentence) for sentence in X_embedding)
X_padded = np.zeros((len(X_embedding), max_length, 50)).astype(np.float32)
for i, sentence in enumerate(X_embedding):
    X_padded[i, :len(sentence)] = sentence
X_flatten = X_padded.reshape(X_padded.shape[0], -1)

# %%
class BaseModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(cv, dim=2)[0] for cv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, hidden = self.rnn(x)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        return self.fc(hidden)


class TransformerModel(nn.Module):
    # embedding_dim要和X_embedding的最后一个维度一致，也就是glove的维度
    def __init__(self, embedding_dim, num_heads, num_layers, output_dim, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        encoded = self.transformer_encoder(x)
        encoded = self.dropout(encoded.mean(dim=0))  # (batch_size, embedding_dim)
        return self.fc(encoded)


def Train_BaseModel(X, y):
    net = BaseModel(X.shape[1], 128, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=20, eval_type="acc", batch_size=16, print_interval=1)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


def Train_CNNModel(X, y):
    net = CNNModel(50, 100, [2, 3, 4], 5, 0.5)
    # class_weights = 1 / torch.tensor([7072, 27273, 79582, 32927, 9206], dtype=torch.float)
    # weights = class_weights / class_weights.sum()
    # criterion = nn.CrossEntropyLoss(weight=weights.to('cuda'))  # 类别不平衡
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=20, eval_type="acc", batch_size=32, print_interval=1,
                             )
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


def Train_RNNModel(X, y):
    net = RNNModel(50, 128, 5, 2, False, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=20, eval_type="acc", batch_size=16, print_interval=1)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")


def Train_TransformerModel(X, y):
    net = TransformerModel(50, 2, 2, 5, 0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    net_trainer = NetTrainer(X, y, net, criterion, optimizer,
                             epochs=20, eval_type="acc", batch_size=16, print_interval=1)
    net_trainer.train_net()
    acc = net_trainer.evaluate_net(delete_train=True)
    print(f"Accuracy: {acc}")

# %%
# 模型1：该模型无法收敛，loss一直是nan，说明mean丢失的信息太多了
# Train_BaseModel(X_mean, y)

# 模型2：该模型测试集的acc可以增加(10个epoch从0.6增加到了0.75)，但是测试集的acc始终在58%左右，说明泛化能力不强
# Train_BaseModel(X_flatten, y)

# 模型3：似乎epoch=10太小了，此时的acc是0.626。因为loss一直在稳步下降，所以可以尝试增加epoch来提高acc，但是一个epoch要训练半分钟。。
# 设置epoch=50的时候acc是0.645。
# Train_CNNModel(X_padded, y)

# 模型4：0.6467，怀疑过拟合了，epoch=9的时候acc=0.6532
# Train_RNNModel(X_padded, y)

# 模型5：0.6496，稍微有点过拟合的倾向，之后可以设置epoch=20看看
# Train_TransformerModel(X_padded, y)

# %%
Train_BaseModel(X_flatten, y)

# %%
Train_CNNModel(X_padded, y)

# %%
Train_RNNModel(X_padded, y)

# %%
Train_TransformerModel(X_padded, y)

# %% [markdown]
# **使用pack_padded_sequence**

# %%
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from sklearn.model_selection import train_test_split

# 划分数据集，80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X_embedding, y, test_size=0.2, random_state=42, stratify=y)
# 数据集与DataLoader
class SentimentDataset(Dataset):
    def __init__(self, X_embedding, y):
        self.X_embedding = X_embedding
        self.y = y
    
    def __len__(self):
        return len(self.X_embedding)
    
    def __getitem__(self, idx):
        return self.X_embedding[idx], self.y[idx]

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    padded_texts = torch.zeros(len(texts), max(lengths), 50)
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = torch.tensor(text)
    return padded_texts, torch.tensor(labels), torch.tensor(lengths)
# 创建Dataset
train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)
# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# %%
class SentimentRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        packed_embedded = pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

# %%
# 模型训练
INPUT_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = 5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3

model = SentimentRNN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 20

# %%
def evaluate(model, data_loader):
    model.eval()
    epoch_acc = 0
    valid_batches = 0
    
    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            # 过滤掉长度小于等于0的情况
            if any(length <= 0 for length in lengths):
                continue
            
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts, lengths)
            acc = (predictions.argmax(dim=1) == labels).float().mean()
            epoch_acc += acc.item()
            valid_batches += 1
    
    return epoch_acc / valid_batches if valid_batches > 0 else 0

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for texts, labels, lengths in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss/len(train_loader)}', end="  ")
    test_acc = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_acc}')

# %%


# %% [markdown]
# **利用了多层GRU进行序列建模，使用CNN提取局部特征，并结合残差连接和Transformer层来增强模型的表达能力**
# 
# 设计思路：
# - GRU: 利用GRU捕获序列信息，并处理变长序列。双向GRU有助于捕获前后文的信息。
# - 残差CNN: CNN层用于提取局部特征，残差连接有助于保留特征并缓解梯度消失问题。将GRU的输出作为输入，这有助于捕捉句子中词的局部依赖性。
# - Transformer: Transformer层增强了全局依赖关系的建模能力。由于Transformer在处理自注意力机制时能够捕捉到更长距离的依赖关系，它补充了GRU和CNN对局部特征的捕捉。
# - 全连接层: 最后通过全连接层将特征映射到情感分类任务的5个类别上。

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EnhancedSentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(EnhancedSentimentModel, self).__init__()
        
        # GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        # 残差CNN层
        self.cnn = nn.Conv1d(in_channels=hidden_dim * 2 if bidirectional else hidden_dim, 
                             out_channels=hidden_dim, kernel_size=3, padding=1)
        self.residual_cnn = nn.Conv1d(in_channels=hidden_dim, 
                                      out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # Transformer层
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        # GRU部分
        packed_embedded = pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # CNN部分
        cnn_output = F.relu(self.cnn(output.permute(0, 2, 1)))
        residual_output = F.relu(self.residual_cnn(cnn_output)) + cnn_output
        
        # Transformer部分
        transformer_output = self.transformer_encoder(residual_output.permute(2, 0, 1))
        
        # 取Transformer输出的最后一个时间步
        final_output = transformer_output[-1]
        
        # 全连接层
        final_output = self.dropout(final_output)
        return self.fc(final_output)


# %%

# 评估过程
def evaluate(model, data_loader):
    model.eval()
    epoch_acc = 0
    valid_batches = 0
    
    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            # 跳过长度为0的样本
            if any(length <= 0 for length in lengths):
                continue
            
            texts, labels = texts.to(device), labels.to(device)
            
            # 前向传播
            predictions = model(texts, lengths)
            
            # 计算准确率
            acc = (predictions.argmax(dim=1) == labels).float().mean()
            epoch_acc += acc.item()
            valid_batches += 1
    
    return epoch_acc / valid_batches if valid_batches > 0 else 0


INPUT_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = 5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
N_EPOCHS = 20

# 初始化模型
model = EnhancedSentimentModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# 训练过程
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for texts, labels, lengths in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 记录损失
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss/len(train_loader)}', end='  ')
    # 测试集评估
    test_acc = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_acc}')


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import lr_scheduler

# 相对于之前的模型，增加BatchNorm与Dropout
class EnhancedSentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(EnhancedSentimentModel, self).__init__()
        
        # GRU层
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=bidirectional, dropout=dropout, batch_first=True)
        
        # 残差CNN层
        self.cnn = nn.Conv1d(in_channels=hidden_dim * 2 if bidirectional else hidden_dim, 
                             out_channels=hidden_dim, kernel_size=3, padding=1)
        self.residual_cnn = nn.Conv1d(in_channels=hidden_dim, 
                                      out_channels=hidden_dim, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 归一化层
        self.dropout_cnn = nn.Dropout(dropout)  # dropout层
        
        # Transformer层
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text, text_lengths):
        # GRU部分
        packed_embedded = pack_padded_sequence(text, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # CNN部分
        cnn_output = F.relu(self.cnn(output.permute(0, 2, 1)))
        residual_output = F.relu(self.residual_cnn(cnn_output)) + cnn_output
        
        # 归一化和dropout
        residual_output = self.batch_norm(residual_output)
        residual_output = self.dropout_cnn(residual_output)
        
        # Transformer部分
        transformer_output = self.transformer_encoder(residual_output.permute(2, 0, 1))
        
        # 取Transformer输出的最后一个时间步
        final_output = transformer_output[-1]
        
        # 全连接层
        final_output = self.dropout(final_output)
        return self.fc(final_output)

# 评估过程
def evaluate(model, data_loader):
    model.eval()
    epoch_acc = 0
    valid_batches = 0
    
    with torch.no_grad():
        for texts, labels, lengths in data_loader:
            # 跳过长度为0的样本
            if any(length <= 0 for length in lengths):
                continue
            
            texts, labels = texts.to(device), labels.to(device)
            
            # 前向传播
            predictions = model(texts, lengths)
            
            # 计算准确率
            acc = (predictions.argmax(dim=1) == labels).float().mean()
            epoch_acc += acc.item()
            valid_batches += 1
    
    return epoch_acc / valid_batches if valid_batches > 0 else 0


INPUT_DIM = 50
HIDDEN_DIM = 128
OUTPUT_DIM = 5
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.3
N_EPOCHS = 20


# 初始化模型
model = EnhancedSentimentModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 设置预热和余弦退火学习率调度器
warmup_epochs = 4
warmup_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1e-5 / 1e-3, total_iters=warmup_epochs)
cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS - warmup_epochs)


# 训练过程
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for texts, labels, lengths in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 记录损失
        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss/len(train_loader)}', end='  ')
    
    # 更新学习率
    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()
        
    # 测试集评估
    test_acc = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_acc}')



