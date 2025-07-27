import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sigmoid
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch import Tensor
from torch.nn import Parameter, Linear, Dropout, LayerNorm, ReLU


class Att(nn.Module):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 4,
            concat: bool = True,
            dropout: float = 0.0,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.bias = bias

        self.linear = Linear(in_channels, out_channels, bias=bias)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)

        return x


IMG_SIZE = 32
OUT = 1
featurelength = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GCNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Att(IMG_SIZE * IMG_SIZE, featurelength)
        self.conv2 = ResGatedGraphConv(featurelength, featurelength)
        self.channel = 4

        self.fc1 = nn.Linear(featurelength * self.channel // 2, OUT)
        self.bn = nn.BatchNorm1d(featurelength)

        self.dropout = nn.Dropout(0.15)
        self.pool = nn.MaxPool1d(2)

    def forward_1(self, x, train):
        batch_size = x.shape[0]

        x_1 = torch.reshape(x, [batch_size, IMG_SIZE * IMG_SIZE])

        edge_index = torch.tensor([[i for i in range(batch_size)] for j in range(batch_size)]).to(device)
        edge_index = edge_index.reshape(1, -1)
        edge_index = torch.cat((edge_index, torch.flip(edge_index, [0])), dim=0)

        x_2 = self.conv1(x_1)

        if train:
            x_2 = self.dropout(x_2)

        x_3 = F.relu(x_2)
        # x_3 = torch.reshape(x_3,[batch_size,-1,channel])

        x_4 = self.conv2(x_3, edge_index)
        # x_4 = torch.reshape(x_4,[batch_size,channel,-1])

        x_4 = F.relu(x_4)
        x_4 = self.pool(self.bn(x_4))

        attention = torch.matmul(x_4, torch.transpose(x_4, 0, 1))
        attention = sigmoid(attention)
        attention = attention / torch.sum(attention, dim=1).unsqueeze(1)

        x_5 = torch.matmul(attention, x_4)

        x_6 = x_5 + x_4
        # output = self.fc1(x_6)
        # output = torch.sigmoid(output)
        return x_6

    def forward(self, x1, train=True):
        batch_size = x1.shape[0]
        x = []
        for i in range(self.channel):
            x2 = x1[:, i, ...]
            x_i = self.forward_1(x2, train=train)
            x.append(x_i)

        x = torch.stack(x, dim=1)
        x = x.view(batch_size, -1)
        output = self.fc1(x)

        return output



