import torch
import torch.nn as nn

# 假设输入矩阵是一个5x5的矩阵，包含5个特征
input_matrix = torch.rand(5, 5, 5)

# 将输入矩阵展开为一个序列
input_sequence = input_matrix.view(-1)

# 将输入序列转换为整数类型
input_sequence = input_sequence.long()

# 定义嵌入层
embedding_dim = 10
embedding = nn.Embedding(25, embedding_dim)

# 将输入序列转换为嵌入向量
embedded_sequence = embedding(input_sequence)

# 将嵌入向量转换回矩阵形式
embedded_matrix = embedded_sequence.view(5, 5, embedding_dim)

print(embedded_matrix.shape)  # 输出: torch.Size([5, 5, 10])