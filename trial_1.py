import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, csv_file):
        # 读取CSV文件
        df = pd.read_csv(csv_file).values
        normal_features = (df[:, 2:7])
        normal_labels = (df[:, 7:8])
        # 将特征数据重新组织为三维数组
        num_features = 5
        self.data_len = len(df) // num_features
        self.labels = np.zeros((self.data_len,1))
        self.features = normal_features.reshape((self.data_len, num_features, -1))
        for i in range(self.data_len):
            self.labels[i] = normal_labels[i * 5]
        self.features = self.features.astype(np.float32)
        self.labels = self.labels.astype(np.float32)
        print(self.features.dtype)
        print(self.labels.dtype)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, num_layers, num_heads, hidden_dim, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 将时间步维度放在第一维
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # 恢复原始维度顺序
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 定义超参数
    input_size = 5
    num_classes = 2
    num_layers = 2
    num_heads = 2
    hidden_dim = 32
    dropout = 0.2
    batch_size = 32
    epochs = 10

    # 创建模型实例
    model = TransformerModel(input_size, num_classes, num_layers, num_heads, hidden_dim, dropout)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建数据集实例
    dataset = MyDataset("./data/all_tf.csv")



    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")