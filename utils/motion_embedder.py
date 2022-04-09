#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/4/9 上午9:38
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rayleigh
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import os

class MotionEmbedder:
    def __init__(self, method="norm", options=None):
        self.method = method
        self.stat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/motion_stat_12.npy')
        self.motion_stat = np.load(self.stat_path)
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        if self.method == "norm":
            self.loc, self.scale = stats.norm.fit_loc_scale(self.motion_stat)
            self.pdf = stats.norm.pdf(self.motion_stat, self.loc, self.scale)
            self.scaler.fit(self.pdf.reshape(-1, 1))
        elif self.method == "rayleigh":
            self.loc, self.scale = stats.rayleigh.fit_loc_scale(self.motion_stat)
            self.pdf = stats.rayleigh.pdf(self.motion_stat, self.loc, self.scale)
            self.scaler.fit(self.pdf.reshape(-1, 1))
        else:
            raise ("Invalid method {}".format(self.method))

    def score(self, x):
        if self.method == "norm":
            pdf = stats.norm.pdf(x, self.loc, self.scale)
            return self.scaler.transform(pdf.reshape(-1, 1)).reshape(x.shape)
        elif self.method == "rayleigh":
            pdf = stats.rayleigh.pdf(x, self.loc, self.scale)
            return self.scaler.transform(pdf.reshape(-1, 1)).reshape(x.shape)
        else:
            raise ("Invalid method {}".format(self.method))

    transform = score

if __name__ == '__main__':
    x = np.array([[[0.02],[0.01],[1.1]],[[0.03],[0.04],[0.05]]])
    me = MotionEmbedder()
    x_t = me.transform(x)
    print(x_t)


