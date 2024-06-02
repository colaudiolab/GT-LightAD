import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(RevIN, self).__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)#得到均值方差
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))

        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


    def _normalize(self, x):

        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x