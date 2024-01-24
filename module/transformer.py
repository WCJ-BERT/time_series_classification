import numpy as np
from torch.nn import Module
import torch
from torch.nn import ModuleList
from module.encoder import Encoder
import math
import torch.nn.functional as F


class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(Transformer, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)
        self.final_output_linear = torch.nn.Linear(d_output,1)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model
    def regression(self,weight,label_index):
        output = np.zeros((len(weight)))
        for i in range(len(weight)):
            output[i] = (label_index[i] - 0.5) + weight[i]
            if output[i] >= 4:
                output[i] = 4
            elif output[i] <= 0:
                output[i] = 0
        output = torch.tensor(output,dtype=torch.float32)
        return output

    def forward(self, x, stage):
        """
        前向传播
        :param x: 输入
        :param stage: 用于描述此时是训练集的训练过程还是测试集的测试过程  测试过程中均不在加mask机制
        :return: 输出，gate之后的二维向量，step-wise encoder中的score矩阵，channel-wise encoder中的score矩阵，step-wise embedding后的三维矩阵，channel-wise embedding后的三维矩阵，gate
        """
        # step-wise 学习时间步间的，例如我有5个时间步为一个sequence单位，5个一起学
        # score矩阵为 input， 默认加mask 和 pe
        '''STEP-WISE'''
        encoding_1 = self.embedding_channel(x) ###5维变512维
        input_to_gather = encoding_1

        if self.pe:
            pe = torch.ones_like(encoding_1[0]) ###创建一个相同形状但初始化为 1 的张量
            position = torch.arange(0, self._d_input).unsqueeze(-1) ###生成等差序列，为每个时间步进行位置编码
            temp = torch.Tensor(range(0, self._d_model, 2))###生成间隔为2的等差数列，终止值为512
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0) ###e为底，做指数乘法
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        '''CHANNEL-WISE'''
        encoding_2 = self.embedding_input(x.transpose(-1, -2)) ###一个维度内，包含25步的所有特征值
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 三维变二维
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1) ##512*25=12800 摊开
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1) ##512*5=2560 摊开

        # gate
        a = self.gate(torch.cat([encoding_1, encoding_2], dim=-1))
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        # gate = F.sigmoid(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)))
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # 输出
        output = self.output_linear(encoding)
        output = F.relu(output)
        output = self.final_output_linear(output)
        # output = F.sigmoid(output)
        # output = self.final_output_linear(output)
        # weight,label_index = torch.max(output.data, dim=-1)
        # weight = weight.cpu().numpy()
        # label_index = label_index.cpu().numpy()
        # final_output = self.regression(weight,label_index)


        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
