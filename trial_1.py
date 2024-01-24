import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 生成示例训练数据，实际应替换为你的数据加载逻辑
def generate_sample_data(num_samples, input_dim, output_dim):
    input_data = torch.rand(num_samples, input_dim)
    target_data = torch.rand(num_samples, output_dim)
    return input_data, target_data

# 模型定义
class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers):
        super(TransformerRegressionModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        output = self.transformer(src)
        output = self.fc(output[-1, :, :])
        return output

# 示例使用
input_dim = 5  # 输入特征维度
output_dim = 1  # 输出维度
nhead = 1  # 多头注意力的头数
num_layers = 3  # 编码器和解码器层数

model = TransformerRegressionModel(input_dim, output_dim, nhead, num_layers)

# 数据准备
num_samples = 1000  # 替换为实际的训练集大小
train_input, train_target = generate_sample_data(num_samples, input_dim, output_dim)

# 转换为 PyTorch 的 TensorDataset
train_dataset = TensorDataset(train_input, train_target)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10  # 替换为实际的训练轮数
for epoch in range(num_epochs):
    for batch_input, batch_target in train_loader:
        optimizer.zero_grad()
        output = model(batch_input)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
