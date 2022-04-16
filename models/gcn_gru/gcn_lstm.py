#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/4/15 下午3:07
"""
import torch
import numpy as np
from torch import nn

from models.gcn_gru.gcn import SimpleGCN as GCN


class GraphConvolutionGRU(nn.Module):

    def __init__(self, in_channels=(3, 64), out_channels=(64, 64), num_layers=2, dropout=0.2):
        super(GraphConvolutionGRU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.layer_list = nn.ModuleList([GraphConvolutionGRUCell(self.in_channels[i], self.out_channels[i]) for i in range(self.num_layers)])

    def forward(self, x, hidden_state=None):
        N, C, T, V = x.size()
        if hidden_state is None:
            hidden_state = torch.empty((self.num_layers, N, self.out_channels[0], 1, V)).to(x.device)
            hidden_state = nn.init.xavier_uniform_(hidden_state, gain=np.sqrt(2))


        last_state_list = []
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h = self.layer_list[layer_idx](cur_layer_input[:, :, t, :], h)
                output_inner.append(h)

            layer_output = torch.cat(output_inner, dim=2)
            cur_layer_input = layer_output

            # layer_output_list.append(layer_output)
            last_state_list.append(h)

        # if not self.return_all_layers:
        #     layer_output_list = layer_output_list[-1:]
        #     last_state_list = last_state_list[-1:]
        last_states = torch.stack(last_state_list, dim=0)
        return layer_output, last_states

class GraphConvolutionGRUCell(nn.Module):
    """Our GRUCell"""

    def __init__(self, in_channels=3, out_channels=64):
        super(GraphConvolutionGRUCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.xz_gcn = GCN(in_channels, out_channels)
        self.hz_gcn = GCN(out_channels, out_channels)
        self.xr_gcn = GCN(in_channels, out_channels)
        self.hr_gcn = GCN(out_channels, out_channels)
        self.xh_gcn = GCN(in_channels, out_channels)
        self.hh_gcn = GCN(out_channels, out_channels)


    def forward(self, x, h=None):
        if len(x.size()) == 3:
            x = x.unsqueeze(2)

        N, C, T, V = x.size()
        if h is None:
            h = torch.empty((N, self.out_channels, 1, V)).to(x.device)
            h = nn.init.xavier_uniform_(h, gain=np.sqrt(2))

        z = torch.sigmoid(self.xz_gcn(x)+self.hz_gcn(h))
        r = torch.sigmoid(self.xr_gcn(x)+self.hr_gcn(h))
        h_wave = torch.tanh(self.xh_gcn(x) + self.hh_gcn(torch.mul(r,h)))
        h_next = torch.mul((1 - z), h_wave) + torch.mul(z, h)
        return h_next

if __name__ == '__main__':
    ggc = GraphConvolutionGRU().to(3)
    x = torch.ones((256, 3, 6, 18)).to(3)  # N, C, T, V
    out, h = ggc(x)
    print(out)
