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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)
        self.relu = nn.ReLU()

        # initialize weights
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

        # nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out[:, -1, :].unsqueeze(1)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True,
                            dropout=0.2, bidirectional=bidirectional)

        # initialize weights
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        # nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.output_size).to(x.device)

        # forward propagate lstm
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super(AutoEncoderRNN, self).__init__()
        # self.sequence_length = args['seg_len']
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, bidirectional)

    def forward(self, x):
        N, T, K = x.size()
        encoded_x = self.encoder(x).expand(-1, T, -1)

        # decoded_x = encoded_x
        decoded_x = self.decoder(encoded_x)

        return decoded_x

if __name__ == '__main__':
    args = {'seg_len': 12}
    # N, T, V*C
    data = torch.randn((32, 12, 54))
    # data = torch.ones((2, 2, 18) 12,)
    data = data.to(0)
    model = AutoEncoderRNN(input_size=54, hidden_size=20).cuda(0)
    out = model(data)
    print(out)
    print(out.shape)