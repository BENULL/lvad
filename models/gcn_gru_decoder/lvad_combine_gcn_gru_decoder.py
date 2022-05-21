#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午4:52
"""
import torch
import torch.nn as nn

from models.gcn_gru_decoder.gcn_gru_decoder import AutoEncoderRNN
from models.gcn_gru_decoder.agcn_one_sample import Model as AGCN
from models.memory_module import MemModule


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.in_channels = 3 #args.in_channels
        self.headless = False #args.headless


        self.gcn = AGCN(in_channels=self.in_channels, headless=self.headless, seg_len=6)
        # self.gcn = MSG3D()
        self.mlp_input_size = self.in_channels * (14 if self.headless else 18)
        self.ae_hidden_size = 324  # self.mlp_input_size
        # self.lstm_ae = AutoEncoderRNN()

        self.mlp_input_size = self.in_channels * (14 if self.headless else 18)
        self.ae_hidden_size = self.mlp_input_size

        self.gcn_lstm_ae = AutoEncoderRNN(num_layers=2)

        self.rec_local_fcn = nn.Conv2d(3, self.in_channels, kernel_size=1)
        self.pre_local_fcn = nn.Conv2d(3, self.in_channels, kernel_size=1)
        self.perceptual_fcn = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

        self.mem_bank1 = MemModule(mem_dim=2000, fea_dim=18)
        self.mem_bank2 = MemModule(mem_dim=2000, fea_dim=18)

    def forward(self, x):
        N, C, T, V = x.size()
        gcn_out = self.gcn(x)  # N, C, 1, V
        # ae_in = gcn_out.unsqueeze(0)

        gcn_out = gcn_out.reshape(-1, 18, 1, 3).contiguous()

        gcn_out, _ = self.mem_bank1(gcn_out)
        gcn_out = gcn_out.reshape(N, 3, 1, -1).contiguous()

        ae_in = torch.stack((gcn_out, gcn_out), dim=0)

        rec_out, pre_out = self.gcn_lstm_ae(ae_in)

        rec_out = self.rec_local_fcn(rec_out)
        pre_out = self.pre_local_fcn(pre_out)

        rec_out = torch.flip(rec_out, dims=[2])
        local_out = torch.cat((rec_out, pre_out), dim=2)
        perceptual_out = self.perceptual_fcn(local_out)

        # local_out = local_out.reshape(N, C, -1, V)
        # perceptual_out = perceptual_out.reshape(N, C, -1, V)

        return local_out, perceptual_out


if __name__ == '__main__':

    model = Model({'in_channels':3}).cuda(3)

    data = torch.ones((256, 3, 6, 18))
    data = data.to(3)
    out = model(data)
    print(out.shape)
