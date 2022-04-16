#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午3:44
"""
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=0.2, bidirectional=bidirectional)

        # self.relu = nn.ReLU()

        # initialize weights
        # nn.init.xavier_uniform_(self.gru.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.xavier_uniform_(self.gru.weight_hh_l0, gain=np.sqrt(2))

        # nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward propagate lstm
        _, h_n = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return h_n
        # return out[:, -1, :].unsqueeze(1)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, output_size, num_layers, batch_first=True,
                          dropout=0.2, bidirectional=bidirectional)

        # initialize weights
        nn.init.xavier_uniform_(self.gru.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.gru.weight_hh_l0, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x, h_0):

        # set initial hidden and cell states

        h0 = h_0
        # h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(x.device)

        # forward propagate lstm
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        # self.sequence_length = args['seg_len']
        self.hidden_size = hidden_size
        # self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional)

        self.rec_decoder = DecoderRNN(hidden_size, hidden_size, num_layers, bidirectional)
        self.pre_decoder = DecoderRNN(hidden_size, hidden_size, num_layers, bidirectional)




    def forward(self, x):
        _, N, K = x.size()
        T = K//3//18
        encoder_h_n = x
        # encoder_h_n = self.encoder(x)
        rec_decoder_h_0 = encoder_h_n.clone()
        pre_decoder_h_0 = encoder_h_n.clone()

        rec_input = torch.zeros((N, T, self.hidden_size)).to(x.device)
        pre_input = rec_input.clone()

        rec_out = self.rec_decoder(rec_input, rec_decoder_h_0)
        pre_out = self.pre_decoder(pre_input, pre_decoder_h_0)


        return rec_out, pre_out

if __name__ == '__main__':
    args = {'seg_len': 12}
    # N, T, V*C
    data = torch.randn((256, 6, 54))
    # data = torch.ones((2, 2, 18) 12,)
    data = data.to(0)
    model = AutoEncoderRNN(input_size=54, hidden_size=20).cuda(0)
    out = model(data)
    print(out)
    print(out.shape)