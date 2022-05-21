#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午4:52
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcn_gru.gcn_autoencoder import AutoEncoderRNN


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.in_channels = args.in_channels
        self.headless = args.headless
        self.mlp_input_size = self.in_channels * (14 if self.headless else 18)
        self.ae_hidden_size = self.mlp_input_size
        self.gcn_lstm_ae = AutoEncoderRNN()

        # self.mlp_in = nn.Sequential(
        #     nn.Linear(self.mlp_input_size, 256),
        #     nn.ReLU(),
        #     # nn.Linear(128, 256),
        #     # nn.ReLU(),
        #     nn.Linear(256, self.mlp_input_size),
        # )
        #
        # self.mlp_out = nn.Sequential(
        #     nn.Linear(self.ae_hidden_size, 256),
        #     nn.ReLU(),
        #     # nn.Linear(128, 256),
        #     # nn.ReLU(),
        #     # nn.Linear(256, 128),
        #     # nn.ReLU(),
        #     nn.Linear(256, self.mlp_input_size),
        # )
        # self.perceptual_linear = nn.Sequential(
        #     nn.Linear(self.mlp_input_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.mlp_input_size),
        # )
        #nn.Linear(self.mlp_input_size, self.mlp_input_size)
        self.local_fcn = nn.Conv2d(64, self.in_channels, kernel_size=1)
        # self.pre_local_fcn = nn.Conv2d(64, self.in_channels, kernel_size=1)
        self.perceptual_fcn = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        rec_out, pre_out = self.gcn_lstm_ae(x)
        # x = F.avg_pool2d(rec_out, rec_out.size()[2:])
        rec_out = self.local_fcn(rec_out)
        pre_out = self.local_fcn(pre_out)

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
