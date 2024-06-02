import torch
import torch.nn as nn
from layer.RevIN import RevIN
from layer.Encoder import Encoder
from utils.tools import positional_encoding

class PatchEmbed(nn.Module):
    def __init__(self, window_size, in_chans, out_chans):
        super().__init__()
        self.padding = window_size//2
        self.proj = nn.Conv1d(in_chans, out_chans, kernel_size=window_size, stride=window_size//2)

    def forward(self, x):
        x = x.transpose(1, 2)
        padding_patch_layer = nn.ReplicationPad1d((0, self.padding))
        x = padding_patch_layer(x)
        x = self.proj(x)
        return x.transpose(1, 2)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.revin_layer = RevIN()
        # predict
        self.pre_len = args.pre_len
        self.feature_dimension = args.feature_dimension
        # patch
        self.patch_len = args.patch_len
        self.stride_len = args.stride_len
        self.padding_len = args.padding_len
        self.patch_num = args.patch_num

        # 位置编码
        self.d_model = args.d_model
        self.dropout_rate = args.dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.W_pos = positional_encoding('zeros', True, self.patch_num, self.d_model)
        self.W_P = nn.Linear(self.patch_len, self.d_model)

        # model
        self.backbone = nn.Sequential(Encoder(args), Encoder(args), Encoder(args))
        self.head = nn.Linear(self.patch_num*self.d_model, self.pre_len)


    def forward(self, x):  # [B,seq_len,var]
        # 消除自身波动
        x = self.revin_layer(x, 'norm')

        # patch
        padding_patch_layer = nn.ReplicationPad1d((0, self.padding_len))
        x = padding_patch_layer(x.transpose(1, 2))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride_len)  # [batch,var,patch_num,patch_len]

        # 位置编码
        x = self.W_P(x)
        x = x.reshape(-1, self.patch_num, self.d_model)  # [batch*var,patch_num,patch_len]
        x = self.dropout(x + self.W_pos)


        # model
        prev = None
        for model in self.backbone:
            x, attn_scores = model(x, prev)
            prev = attn_scores
        x = x.reshape(-1, self.feature_dimension, self.patch_num, self.d_model)
        x = x.permute(0, 1, 3, 2)
        x = self.head(x.reshape(-1, self.patch_num*self.d_model))
        x = x.reshape(-1, self.feature_dimension, self.pre_len)

        # 还原波动
        x = self.revin_layer(x.transpose(1, 2), 'denorm')
        return x.transpose(1, 2)