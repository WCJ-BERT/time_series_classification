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


correct_on_test = []
correct_on_train = []
class MyDataset(Dataset):
    def __init__(self, csv_file):
        # 读取CSV文件
        df = pd.read_csv(csv_file).values
        normal_features = (df[:, 2:7])
        normal_labels = (df[:, 7:8])
        normal_labels = normal_labels.reshape((len(df),))
        # 将特征数据重新组织为三维数组
        num_features = 5
        self.data_len = len(df) // num_features
        self.labels = np.zeros((self.data_len,))
        self.features = normal_features.reshape((self.data_len, num_features, -1))
        for i in range(self.data_len):
            self.labels[i] = normal_labels[i * 5]
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

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

    BATCH_SIZE = 3
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
    print(f'use device: {DEVICE}')

    # 选择要验证的模型
    save_model_path = 'saved_model/all_tf 88.24 batch=3.pkl'
    file_name = save_model_path.split('/')[-1].split(' ')[0]


    net = torch.load(save_model_path, map_location = DEVICE)

    # 加载验证数据集
    testdataset = MyDataset('./data/test.csv')
    test_dataloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

    test(test_dataloader)