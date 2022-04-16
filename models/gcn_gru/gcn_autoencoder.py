#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/3/7 下午3:44
"""
import torch
import torch.nn as nn
import numpy as np

from models.gcn_gru.gcn_lstm import GraphConvolutionGRU


class EncoderRNN(nn.Module):

    def __init__(self, in_channels=(3, 64), out_channels=(64, 64), num_layers=2):
        super(EncoderRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.graph_convolution_gru = GraphConvolutionGRU(self.in_channels, self.out_channels, num_layers=2)

    def forward(self, x):
        _, h_n = self.graph_convolution_gru(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        return h_n


class DecoderRNN(nn.Module):

    def __init__(self, in_channels=(3, 64), out_channels=(64, 64), num_layers=2):
        super(DecoderRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.graph_convolution_gru = GraphConvolutionGRU(self.in_channels, self.out_channels, num_layers=2)

    def forward(self, x, h=None):
        out, _ = self.graph_convolution_gru(x, h)
        return out


class AutoEncoderRNN(nn.Module):
    def __init__(self, in_channels=(64, 64), out_channels=(64, 64), num_layers=2):
        super(AutoEncoderRNN, self).__init__()
        self.encoder = EncoderRNN()
        self.rec_decoder = DecoderRNN()
        self.pre_decoder = DecoderRNN()

    def forward(self, x):
        N, C, T, V = x.size()
        encoder_h_n = self.encoder(x)
        rec_decoder_h_0 = encoder_h_n.clone()
        pre_decoder_h_0 = encoder_h_n.clone()

        rec_input = torch.zeros((N, C, T, V)).to(x.device)
        pre_input = rec_input.clone()

        rec_out = self.rec_decoder(rec_input, rec_decoder_h_0)
        pre_out = self.pre_decoder(pre_input, pre_decoder_h_0)

        return rec_out, pre_out


if __name__ == '__main__':
    args = {'seg_len': 12}
    # N, T, V*C
    data = torch.randn((256, 3, 6, 18))
    # data = torch.ones((2, 2, 18) 12,)
    data = data.to(3)
    model = AutoEncoderRNN().cuda(3)
    out = model(data)
    print(out)
