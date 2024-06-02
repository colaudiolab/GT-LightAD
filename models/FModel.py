import torch
import torch.nn as nn
from layer.RevIN import RevIN

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, feature_dimension, stride=1):
        super(moving_avg, self).__init__()
        self.avg = nn.Conv1d(in_channels=feature_dimension, out_channels=feature_dimension,
                             kernel_size=kernel_size, stride=stride, groups=feature_dimension)
        #纯卷积
        # self.avg = nn.Conv1d(in_channels=feature_dimension, out_channels=feature_dimension,
        #                      kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        # no padding
        x = self.avg(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

class FModel(nn.Module):
    def __init__(self, args):
        super(FModel, self).__init__()
        self.revin_layer = RevIN()
        self.seq_len = args.seq_len
        self.pre_len = args.pre_len
        self.layer = args.layer
        self.MA_factor = args.MA_factor
        self.start = self.seq_len
        self.end = self.seq_len - (self.MA_factor-1) * (self.layer-1)
        self.input_len = [i for i in range(self.start, self.end-1, -(self.MA_factor-1))]
       # 移动平均

        self.Moving_Average_block = nn.ModuleList(
            [moving_avg(self.MA_factor, args.feature_dimension) for i in range(self.layer-1)])
        self.Linear = nn.ModuleList([nn.Linear(self.input_len[i], self.pre_len) for i in range(len(self.input_len))])

        # 初始化权重和偏置
        # for linear in self.Linear:
        #     nn.init.normal_(linear.weight, 0, 0.1)  # 使用正态分布初始化权重
        #     if linear.bias is not None:
        #         nn.init.normal_(linear.weight, 0, 0.1)  # 初始化偏置为0

        # self.predict = nn.Linear(self.pre_len, self.pre_len)
        # nn.init.normal_(self.predict.weight, mean=0.1, std=0.1)
        # nn.init.normal_(self.predict.bias, mean=0.1, std=0.1)
        # self.predict = nn.Linear(100, 100)
        self.mix = nn.Linear(self.layer, 1)
        self.dropout = nn.Dropout(args.dropout_rate)

    def forward(self, x):
        # 消除自身波动
        x = self.revin_layer(x, 'norm')
        x_input = [x.transpose(1, 2)]
        for block in self.Moving_Average_block:
            x = block(x)
            x_input.append(x.transpose(1, 2))

        #实际预测
        y = []
        for i in range(len(x_input)):
            x = x_input[i]
            x = self.Linear[i](x)
            x = self.dropout(x)
            y.append(x)
        y = torch.stack(y, -1)
        y = self.mix(y).squeeze(-1)

        # 还原自身波动
        y = self.revin_layer(y.transpose(1, 2), 'denorm')
        return y
