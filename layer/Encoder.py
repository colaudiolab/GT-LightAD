import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.patch_num = args.patch_num
        self.num_heads = args.num_heads
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.dropout_rate = args.dropout_rate
        self.pre_norm = True

        # 自注意力融合
        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)

        # others
#        self.norm = nn.LayerNorm(self.d_model)
        self.norm_attn = nn.BatchNorm1d(self.d_model)
        self.norm_ffn = nn.BatchNorm1d(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.to_out = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Dropout(self.dropout_rate))
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                nn.GELU(),
                                nn.Dropout(self.dropout_rate),
                                nn.Linear(self.d_ff, self.d_model))

    def attention(self, x, prev):  # [B, C, self.window_size, self.embed_dims]
        num_heads = self.num_heads
        n_demension = int(x.size(-1)/num_heads)
        scale = nn.Parameter(torch.tensor(n_demension ** -0.5))
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.reshape(-1, self.patch_num, num_heads, n_demension).transpose(1, 2)
        K = K.reshape(-1, self.patch_num, num_heads, n_demension).transpose(1, 2)
        V = V.reshape(-1, self.patch_num, num_heads, n_demension).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(2, 3)) * scale
        if prev is not None:
            attn_scores = attn_scores + prev
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 注意力融合
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).reshape(-1,  self.patch_num, num_heads*n_demension)
        output = self.to_out(output)
        return output, attn_scores

    def forward(self, x, prev):  # [B*C, self.window_num,self.embed_dims]
        if self.pre_norm:
            x = x.transpose(1, 2)
            x = self.norm_attn(x)
            x = x.transpose(1, 2)
        new_x, attn_scores = self.attention(x, prev)
        x = x + self.dropout(new_x)
        if not self.pre_norm:
            x = x.transpose(1, 2)
            x = self.norm_attn(x)
            x = x.transpose(1, 2)

        #  feed_forward
        if self.pre_norm:
            x = x.transpose(1, 2)
            x = self.norm_ffn(x)
            x = x.transpose(1, 2)
        new_x = self.ff(x)
        x = x + self.dropout(new_x)
        if not self.pre_norm:
            x = x.transpose(1, 2)
            x = self.norm_ffn(x)
            x = x.transpose(1, 2)

        return x, attn_scores






