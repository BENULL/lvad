#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午4:52
"""
import torch
import torch.nn as nn

from models.gcn_lstmdecoder.gru_two_decoder import AutoEncoderRNN
from models.gcn_lstmdecoder.agcn_with_tcn import Model as AGCN


class Model(nn.Module):
    def __init__(self, args=None):
        super(Model, self).__init__()
        self.in_channels = args.in_channels
        self.headless = args.headless
        self.gcn = AGCN(in_channels=self.in_channels, headless=self.headless, seg_len=6)
        # self.gcn = MSG3D()
        self.mlp_input_size = self.in_channels * (14 if self.headless else 18)
        self.ae_hidden_size = 324 #self.mlp_input_size
        self.lstm_ae = AutoEncoderRNN(input_size=self.mlp_input_size,
                                               hidden_size=self.ae_hidden_size,
                                               num_layers=2)

        self.mlp_in = nn.Sequential(
            nn.Linear(self.mlp_input_size, 256),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            nn.Linear(256, self.mlp_input_size),
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(self.ae_hidden_size, 256),
            nn.ReLU(),
            # nn.Linear(128, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(256, self.mlp_input_size),
        )
        self.perceptual_linear = nn.Sequential(
            nn.Linear(self.mlp_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.mlp_input_size),
        )
        #nn.Linear(self.mlp_input_size, self.mlp_input_size)

    def forward(self, x):
        N, C, T, V = x.size()
        gcn_out = self.gcn(x) # N, -1, (T*C*V(324))
        ae_in = torch.stack((gcn_out, gcn_out), dim=0)
        # gcn_out = x.reshape(N, T, C*V)
        # gcn_out = self.mlp_in(gcn_out)

        rec_out, pre_out = self.lstm_ae(ae_in)
        rec_out = self.mlp_out(rec_out)
        pre_out = self.mlp_out(pre_out)
        rec_out = torch.flip(rec_out, dims=[1])
        local_out = torch.cat((rec_out, pre_out), dim=1)
        perceptual_out = self.perceptual_linear(local_out)

        local_out = local_out.reshape(N, C, -1, V)
        perceptual_out = perceptual_out.reshape(N, C, -1, V)

        return local_out, perceptual_out


if __name__ == '__main__':

    model = Model({'in_channels':3}).cuda(0)

    data = torch.ones((2, 3, 6, 18))
    data = data.to(0)
    out = model(data)
    print(out.shape)
