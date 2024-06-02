import os
import torch
import numpy as np
from models.Model import Model
from models.FModel import FModel
import torch.nn as nn
from thop import profile

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # net = self.model
        # inputs = torch.randn(1, 100, 51).to(self.device)
        # flops, params = profile(net, (inputs,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    def _build_model(self):
        model = FModel(self.args)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
