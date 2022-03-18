#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午4:52
"""
import torch
import torch.nn as nn

from models.lstm_autoencoder import TwoBranchAutoEncoderRNN
from models.agcn import Model as AGCN

class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.gcn = AGCN(in_channels=3, headless=False, seg_len=args.seg_len).cuda(0)
        self.lstm_ae = TwoBranchAutoEncoderRNN(input_size=54, hidden_size=16, num_layers=4).cuda(0)

    def forward(self, x):
        N, C, T, V = x.size()
        gcn_out = self.gcn(x)

        # gcn_out = x.reshape(N, T, C*V)

        rec_out, pre_out = self.lstm_ae(gcn_out)

        # out = self.fcn(lstm_out)

        # out = self.linear(out)

        rec_out = rec_out.reshape(N, C, T, V)
        pre_out = pre_out.reshape(N, C, T//2, V)
        # out = torch.cat((rec_out, pre_out), dim=2)
        return rec_out, pre_out


if __name__ == '__main__':
    model = Model().cuda(0)

    data = torch.ones((2, 3, 12, 18))
    data = data.to(0)
    out = model(data)
    print(out.shape)
