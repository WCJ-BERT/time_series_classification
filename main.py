import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
import torch.optim as optim
from module.transformer import Transformer
from module.loss import Myloss
from time import time
from tqdm import tqdm
from utils.random_seed import setup_seed
from utils.visualization import result_visualization
from utils.data import data_process
from torch.utils.tensorboard import SummaryWriter

# 用于记录准确率变化
correct_on_train = []
correct_on_test = []
# 用于记录损失变化
loss_list = []
time_cost = 0


class MyDataset(Dataset):
    def __init__(self, csv_file , d_step):
        # 读取CSV文件

        df = data_process(d_step , csv_file)  ###25是时间步数量
        self.features = (df[: , : , 2:7])
        normal_labels = (df[: , : , 7:8])
        self.data_len = len(df)
        self.labels = np.zeros((self.data_len,))
        for i in range(self.data_len):
            self.labels[i] = normal_labels[i][0]
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train():
    net.train()
    writer = SummaryWriter('logs/' + str(BATCH_SIZE))
    max_accuracy = 0
    pbar = tqdm(total=EPOCH)
    begin = time()
    for index in range(EPOCH):
        for i, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            y_pre, _, _, _, _, _, _ = net(x.to(DEVICE), 'train')

            loss = loss_function(y_pre, y.to(DEVICE))

            print(f'Epoch:{index + 1}:\t\tloss:{loss.item()}')
            loss_list.append(loss.item())

            loss.backward()

            optimizer.step()

        if ((index + 1) % test_interval) == 0:
            current_accuracy = test(test_dataloader)
            test(train_dataloader, 'train_set')
            print(f'当前最大准确率\t测试集:{max(correct_on_test)}%\t 训练集:{max(correct_on_train)}%')
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                torch.save(net, f'saved_model/{file_name} batch={BATCH_SIZE}.pkl')

        pbar.update()
        writer.add_scalar('Loss', loss, EPOCH)

    os.rename(f'saved_model/{file_name} batch={BATCH_SIZE}.pkl',
              f'saved_model/{file_name} {max_accuracy} batch={BATCH_SIZE}.pkl')

    end = time()
    time_cost = round((end - begin) / 60, 2)
    print('用时：',time_cost)
    writer.close()

    # # 结果图
    # result_visualization(loss_list=loss_list, correct_on_test=correct_on_test, correct_on_train=correct_on_train,
    #                      test_interval=test_interval,
    #                      d_model=d_model, q=q, v=v, h=h, N=N, dropout=dropout, DATA_LEN=DATA_LEN, BATCH_SIZE=BATCH_SIZE,
    #                      time_cost=time_cost, EPOCH=EPOCH, draw_key=draw_key, reslut_figure_path=reslut_figure_path,
    #                      file_name=file_name,
    #                      optimizer_name=optimizer_name, LR=LR, pe=pe, mask=mask)

def test(dataloader, flag='test_set'):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pre, _, _, _, _, _, _ = net(x, 'test')
            _, label_index = torch.max(y_pre.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        if flag == 'test_set':
            correct_on_test.append(round((100 * correct / total), 2))
        elif flag == 'train_set':
            correct_on_train.append(round((100 * correct / total), 2))
        print(f'Accuracy on {flag}: %.2f %%' % (100 * correct / total))

        return round((100 * correct / total), 2)



if __name__ == '__main__':
    test_interval = 5  # 测试间隔 单位：epoch
    draw_key = 1  # 大于等于draw_key才会保存图像
    path = './data/all_tf.csv'
    file_name = path.split('/')[-1][0:path.split('/')[-1].index('.')]  # 获得文件名字
    # file_name = 'GGG'

    # 超参数设置
    EPOCH = 100
    BATCH_SIZE = 1024 #####
    LR = 1e-4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
    print(f'use device: {DEVICE}')

    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    N = 8
    dropout = 0.2
    pe = True  # # 设置的是双塔中 score=pe score=channel默认没有pe
    mask = True  # 设置的是双塔中 score=input的mask score=channel默认没有mask
    # 优化器选择
    optimizer_name = 'Adagrad'

    d_step = 40
    ####加载训练、验证数据集
    dataset = MyDataset("./data/all_tf.csv",d_step)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    testdataset = MyDataset('./data/test.csv',d_step)
    test_dataloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

    DATA_LEN = dataset.data_len  # 训练集样本数量
    # d_input = dataset.data_len  # 时间步数量
    d_input = d_step  # 时间步数量
    d_channel = 5  # 时间序列维度
    d_output = 5  # 分类类别

    # 创建Transformer模型
    net = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                      q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=DEVICE).to(DEVICE)
    # 创建loss函数 此处使用 交叉熵损失
    loss_function = Myloss()
    if optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=LR)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=LR)


    train()

