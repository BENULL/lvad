#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午4:52
"""
import torch
import torch.nn as nn

from models.lstm_autoencoder import AutoEncoderRNN
from models.agcn import Model as AGCN

class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.gcn = AGCN(in_channels=3, headless=False, seg_len=args.seg_len).cuda(0)
        self.lstm_ae = AutoEncoderRNN(input_size=54, hidden_size=18, num_layers=2).cuda(0)

        self.linear = nn.Linear(54, 54)
        # self.fcn = nn.Conv1d(12, 12, kernel_size=203)

    def forward(self, x):
        N, C, T, V = x.size()
        gcn_out = self.gcn(x)

        # gcn_out = x.reshape(N, T, C*V)

        out = self.lstm_ae(gcn_out)
        # out = self.fcn(lstm_out)

        out = self.linear(out)

        out = out.reshape(N, C, T, V)
        return out


if __name__ == '__main__':
    model = Model().cuda(0)

    data = torch.ones((2, 3, 12, 18))
    data = data.to(0)
    out = model(data)
    print(out.shape)
