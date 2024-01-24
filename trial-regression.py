import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
    def __init__(self, csv_file):
        # 读取CSV文件

        df = data_process(25,csv_file)  ###25是时间步数量
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


class RegressionModel(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim):
        super(RegressionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim * seq_length, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, output_dim)
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, seq_length, output_dim,DEVICE,num_heads=5, num_layers=3):
        super(TransformerRegressionModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                device=DEVICE
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(input_dim * seq_length, output_dim)

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.transformer(x)
        x = x.flatten(1)  # 将输出展平为一维向量
        x = self.fc(x)
        return x


def train():
    pbar = tqdm(total=EPOCH)
    for epoch in range(EPOCH):
        for i , (x,y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(x.to(DEVICE))
            loss = loss_function(output, y.to(DEVICE))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{EPOCH}, Loss: {loss.item()}')
        pbar.update()

if __name__ == '__main__':
    test_interval = 5  # 测试间隔 单位：epoch
    draw_key = 1  # 大于等于draw_key才会保存图像
    path = './data/all_tf.csv'
    file_name = path.split('/')[-1][0:path.split('/')[-1].index('.')]  # 获得文件名字
    # file_name = 'GGG'

    # 超参数设置
    EPOCH = 100
    BATCH_SIZE = 32 #####
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

    ####加载训练、验证数据集
    dataset = MyDataset("./data/25_step/all_tf.csv")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    testdataset = MyDataset('./data/25_step/test.csv')
    test_dataloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

    DATA_LEN = dataset.data_len  # 训练集样本数量
    d_input = 25  # 时间步数量
    d_channel = 5  # 时间序列维度
    d_output = 1  #回归问题，输出维度为1

    # model = RegressionModel(d_channel, d_input, d_output).to(DEVICE)
    model = TransformerRegressionModel(d_channel, d_input, d_output,DEVICE).to(DEVICE)

    # 创建loss函数 此处使用 交叉熵损失
    loss_function = nn.MSELoss()
    if optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=LR)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LR)

    train()

